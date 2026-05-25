#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import sysconfig
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Iterable

from packaging.tags import sys_tags
from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import InvalidVersion, Version


DEFAULT_FLASH_ATTN_VERSION = "2.7.3"
OFFICIAL_REPO = "Dao-AILab/flash-attention"
COMMUNITY_REPO = "mjun0812/flash-attention-prebuild-wheels"
DEFAULT_SOURCES = ("official", "community")


@dataclass(frozen=True)
class WheelAsset:
    name: str
    url: str
    source: str
    tag: str
    score: tuple


@dataclass(frozen=True)
class SystemInfo:
    python_tag: str
    torch_raw: str
    torch_major_minor: str
    torch_exact: str
    cuda_raw: str
    cuda_major: str
    cuda_strict: str
    cxx11_abi: str
    supported_tags: frozenset[str]
    platform_name: str
    gpu_name: str | None
    gpu_capability: tuple[int, int] | None


def request_json(url: str):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "trellis2-flash-attn-wheel-installer",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def paged_api(url: str, max_pages: int | None = None):
    page = 1
    while True:
        sep = "&" if "?" in url else "?"
        page_url = f"{url}{sep}per_page=100&page={page}"
        data = request_json(page_url)
        if not data:
            break
        yield from data
        if len(data) < 100:
            break
        page += 1
        if max_pages is not None and page > max_pages:
            break


def get_python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def normalize_machine(machine: str) -> str:
    machine = machine.lower()
    if machine in {"amd64", "x86-64"}:
        return "x86_64"
    if machine in {"aarch64", "arm64"}:
        return "aarch64"
    return machine.replace("-", "_")


def get_platform_name() -> str:
    platform_name = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    if sys.platform.startswith("linux") and platform_name.startswith("linux_"):
        return f"linux_{normalize_machine(platform_name.removeprefix('linux_'))}"
    return platform_name


def get_system_info() -> SystemInfo:
    print("Checking system environment...")
    try:
        import torch
    except ImportError:
        print("   [!] Error: PyTorch is not installed.")
        sys.exit(1)

    torch_raw = torch.__version__
    torch_version = Version(torch_raw.split("+", 1)[0])
    cuda_raw = torch.version.cuda
    if not cuda_raw:
        print("   [!] Error: PyTorch is not compiled with CUDA support.")
        sys.exit(1)

    cuda_parts = cuda_raw.split(".")
    cuda_major = cuda_parts[0]
    cuda_minor = cuda_parts[1] if len(cuda_parts) > 1 else "0"
    cuda_strict = f"cu{cuda_major}{cuda_minor}"
    cxx11_abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()

    gpu_name = None
    gpu_capability = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_capability = tuple(torch.cuda.get_device_capability(0))
        except Exception:
            pass

    info = SystemInfo(
        python_tag=get_python_tag(),
        torch_raw=torch_raw,
        torch_major_minor=f"{torch_version.major}.{torch_version.minor}",
        torch_exact=f"{torch_version.major}.{torch_version.minor}.{torch_version.micro}",
        cuda_raw=cuda_raw,
        cuda_major=f"cu{cuda_major}",
        cuda_strict=cuda_strict,
        cxx11_abi=cxx11_abi,
        supported_tags=frozenset(str(tag) for tag in sys_tags()),
        platform_name=get_platform_name(),
        gpu_name=gpu_name,
        gpu_capability=gpu_capability,
    )

    print(f"   Python: {sys.version_info.major}.{sys.version_info.minor} ({info.python_tag})")
    print(f"   PyTorch: {info.torch_raw} (torch{info.torch_major_minor})")
    print(f"   CUDA from PyTorch: {info.cuda_raw} ({info.cuda_strict} / {info.cuda_major})")
    print(f"   CXX11 ABI: {info.cxx11_abi}")
    print(f"   Platform: {info.platform_name}")
    if info.gpu_name and info.gpu_capability:
        cap = ".".join(str(part) for part in info.gpu_capability)
        print(f"   GPU: {info.gpu_name} (sm_{cap})")
        if info.gpu_capability < (8, 0):
            print(
                "   [!] FlashAttention-2 official CUDA kernels target Ampere/Ada/Hopper; "
                "Turing/Volta users should prefer ATTN_BACKEND=xformers."
            )
    return info


def target_version_matches(filename: str, target_version: Version) -> bool:
    try:
        _, wheel_version, _, _ = parse_wheel_filename(filename)
    except InvalidWheelFilename:
        return False
    return wheel_version.public == target_version.public


def torch_tokens(info: SystemInfo) -> tuple[str, ...]:
    if info.torch_exact.endswith(".0"):
        return (info.torch_major_minor, info.torch_exact)
    return (info.torch_exact, info.torch_major_minor)


def asset_score(filename: str, source: str, info: SystemInfo) -> tuple | None:
    try:
        distribution, _, _, wheel_tags = parse_wheel_filename(filename)
    except InvalidWheelFilename:
        return None

    if distribution != "flash-attn":
        return None

    if not {str(tag) for tag in wheel_tags}.intersection(info.supported_tags):
        return None

    lowered = filename.lower()
    cuda_score = -1
    if source == "official":
        # Official wheels normalize CUDA 12.x to cu12 and CUDA 11.x to cu11.
        cuda_order = (info.cuda_major, info.cuda_strict)
    else:
        # Community wheels usually encode the exact PyTorch CUDA minor.
        cuda_order = (info.cuda_strict, info.cuda_major)
    for index, cuda_token in enumerate(cuda_order):
        if f"+{cuda_token.lower()}torch" in lowered:
            cuda_score = len(cuda_order) - index
            break
    if cuda_score < 0:
        return None

    torch_score = -1
    for index, torch_token in enumerate(torch_tokens(info)):
        if re.search(rf"torch{re.escape(torch_token)}(?=cxx11abi|-|\+)", lowered):
            torch_score = len(torch_tokens(info)) - index
            break
    if torch_score < 0:
        return None

    abi_score = 1
    abi_match = re.search(r"cxx11abi(true|false)", lowered)
    if abi_match:
        if abi_match.group(1).upper() != info.cxx11_abi:
            return None
        abi_score = 2

    platform_score = 2 if info.platform_name in lowered else 1
    return (cuda_score, torch_score, abi_score, platform_score, filename)


def matching_assets(
    assets: Iterable[dict],
    source: str,
    tag: str,
    info: SystemInfo,
    target_version: Version,
):
    for asset in assets:
        name = asset.get("name", "")
        if not name.endswith(".whl"):
            continue
        if not target_version_matches(name, target_version):
            continue
        score = asset_score(name, source, info)
        if score is None:
            continue
        url = asset.get("browser_download_url")
        if not url:
            continue
        yield WheelAsset(name=name, url=url, source=source, tag=tag, score=score)


def assets_from_release(release: dict):
    assets = release.get("assets") or []
    assets_count = release.get("assets_count")
    if assets and (assets_count is None or len(assets) >= assets_count):
        return assets
    assets_url = release["assets_url"].split("{", 1)[0]
    return paged_api(assets_url)


def official_release_assets(version: str):
    tag = f"v{version}"
    url = f"https://api.github.com/repos/{OFFICIAL_REPO}/releases/tags/{urllib.parse.quote(tag)}"
    release = request_json(url)
    return tag, assets_from_release(release)


def community_release_assets(max_release_pages: int | None = None):
    url = f"https://api.github.com/repos/{COMMUNITY_REPO}/releases"
    for release in paged_api(url, max_pages=max_release_pages):
        tag = release.get("tag_name", "unknown")
        yield tag, assets_from_release(release)


def find_official_wheel(info: SystemInfo, target_version: Version):
    version = target_version.public
    print(f"\n[1/2] Querying official {OFFICIAL_REPO} release v{version}...")
    try:
        tag, assets = official_release_assets(version)
        matches = sorted(
            matching_assets(assets, "official", tag, info, target_version),
            key=lambda asset: asset.score,
            reverse=True,
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(f"   [-] No official release tag v{version}.")
            return None
        print(f"   [!] Official release query failed: HTTP {exc.code}")
        return None
    except Exception as exc:
        print(f"   [!] Official release query failed: {exc}")
        return None

    if not matches:
        print("   [-] No compatible official wheel found.")
        return None
    print(f"   [+] Matched {matches[0].name}")
    return matches[0]


def find_community_wheel(info: SystemInfo, target_version: Version, max_release_pages: int | None):
    print(f"\n[2/2] Querying community {COMMUNITY_REPO} releases...")
    best = None
    try:
        for tag, assets in community_release_assets(max_release_pages=max_release_pages):
            matches = sorted(
                matching_assets(assets, "community", tag, info, target_version),
                key=lambda asset: asset.score,
                reverse=True,
            )
            if matches:
                best = matches[0]
                break
    except urllib.error.HTTPError as exc:
        print(f"   [!] Community release query failed: HTTP {exc.code}")
        return None
    except Exception as exc:
        print(f"   [!] Community release query failed: {exc}")
        return None

    if best is None:
        print("   [-] No compatible community wheel found.")
        return None
    print(f"   [+] Matched {best.name} from {best.tag}")
    return best


def find_wheel(info: SystemInfo, version: str, sources: tuple[str, ...], max_release_pages: int | None):
    try:
        target_version = Version(version)
    except InvalidVersion:
        print(f"[!] Invalid FlashAttention version: {version}")
        sys.exit(2)

    for source in sources:
        if source == "official":
            wheel = find_official_wheel(info, target_version)
        elif source == "community":
            wheel = find_community_wheel(info, target_version, max_release_pages)
        else:
            print(f"   [!] Ignoring unknown wheel source: {source}")
            continue
        if wheel:
            return wheel
    return None


def install_wheel(wheel: WheelAsset, dry_run: bool):
    print(f"\n[+] Compatible wheel: {wheel.name}")
    print(f"    Source: {wheel.source} ({wheel.tag})")
    print(f"    URL: {wheel.url}")
    cmd = [sys.executable, "-m", "pip", "install", "--no-deps", wheel.url]
    if dry_run:
        print("\nDry run; not installing.")
        print("Command:")
        print(" ".join(cmd))
        return
    print("\nInstalling prebuilt FlashAttention wheel...")
    subprocess.check_call(cmd)
    print("\n[+] FlashAttention wheel installation complete.")


def source_build(version: str, dry_run: bool):
    env = os.environ.copy()
    env.setdefault("MAX_JOBS", "4")
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        f"flash-attn=={version}",
        "--no-build-isolation",
    ]
    print("\n[!] Falling back to source build because --allow-source-build was set.")
    print("    This can take a long time in Colab/Vast-style environments.")
    if dry_run:
        print("Dry run; not building.")
        print("Command:")
        print(f"MAX_JOBS={env['MAX_JOBS']} " + " ".join(cmd))
        return
    subprocess.check_call(cmd, env=env)


def parse_sources(value: str) -> tuple[str, ...]:
    sources = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    return sources or DEFAULT_SOURCES


def main():
    parser = argparse.ArgumentParser(
        description="Install FlashAttention from a matching prebuilt wheel when available."
    )
    parser.add_argument(
        "--version",
        default=os.environ.get("FLASH_ATTN_VERSION", DEFAULT_FLASH_ATTN_VERSION),
        help=f"FlashAttention version to install (default: {DEFAULT_FLASH_ATTN_VERSION}).",
    )
    parser.add_argument(
        "--sources",
        default=os.environ.get("FLASH_ATTN_SOURCES", ",".join(DEFAULT_SOURCES)),
        help="Comma-separated wheel sources to try: official,community.",
    )
    parser.add_argument(
        "--prefer-community",
        action="store_true",
        help="Try community wheels before official wheels.",
    )
    parser.add_argument(
        "--max-community-release-pages",
        type=int,
        default=int(os.environ.get("FLASH_ATTN_MAX_COMMUNITY_RELEASE_PAGES", "0")),
        help="Limit community GitHub release pages scanned; 0 scans all pages.",
    )
    parser.add_argument(
        "--allow-source-build",
        action="store_true",
        default=os.environ.get("FLASH_ATTN_ALLOW_SOURCE_BUILD", "").lower() in {"1", "true", "yes"},
        help="Build from source only if no compatible prebuilt wheel exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the install command without installing.",
    )
    args = parser.parse_args()

    sources = parse_sources(args.sources)
    if args.prefer_community:
        sources = tuple(sorted(sources, key=lambda source: 0 if source == "community" else 1))
    max_release_pages = args.max_community_release_pages or None

    info = get_system_info()
    wheel = find_wheel(info, args.version, sources, max_release_pages)
    if wheel:
        install_wheel(wheel, args.dry_run)
        return

    print("\n[-] No compatible prebuilt FlashAttention wheel was found.")
    print("    Checked:")
    for source in sources:
        print(f"    - {source}")
    print(
        "    Required match: "
        f"flash-attn {args.version}, Python {info.python_tag}, "
        f"torch{info.torch_major_minor}, {info.cuda_strict}/{info.cuda_major}, "
        f"cxx11abi{info.cxx11_abi}, platform tag compatible with {info.platform_name}."
    )
    print("    Not building from source by default.")
    if args.allow_source_build:
        source_build(args.version, args.dry_run)
        return
    print("    To opt into a source build, rerun with --allow-source-build.")
    sys.exit(1)


if __name__ == "__main__":
    main()

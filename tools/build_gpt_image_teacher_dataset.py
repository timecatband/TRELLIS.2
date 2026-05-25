#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import logging
import os
import re
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from trellis2 import models
from trellis2.modules.sparse import SparseTensor
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap, PbrMeshRenderer
from trellis2.utils.random_utils import sphere_hammersley_sequence
from trellis2.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics


DEFAULT_TEACHER_PROMPT = """Use the first input image as the current albedo/base-color render of a generated 3D asset view.
Generate a new improved albedo/base-color image only: sharper, more coherent texture detail, but still pixel-aligned to the first image.
Preserve silhouette, camera viewpoint, object boundaries, geometry, material identity, and colors.
Do not add text, background, new parts, lighting effects, shadows, or view changes.
Return only the repaired albedo/base-color view."""

DEFAULT_TEACHER_PROMPT_WITH_SOURCE = """Use the first input image as the current albedo/base-color render of a generated 3D asset view.
Use the second input image as the original source/reference image that the 3D asset was generated from.
Generate a new improved albedo/base-color image only: make the visible texture look more like the source/reference image while staying pixel-aligned to the generated render.
Preserve the generated render's silhouette, camera viewpoint, object boundaries, geometry, and visible part layout.
Do not add text, background, new parts, lighting effects, shadows, geometry changes, or view changes.
Return only the repaired albedo/base-color view."""

DEFAULT_TEACHER_PROMPT_WITH_NORMAL = """Use the first input image as the current albedo/base-color render of a generated 3D asset view.
Use the second input image as a camera-space normal map that defines the exact visible geometry.
Generate a new improved albedo/base-color image only: sharper, more coherent texture detail, but still pixel-aligned to the first image and geometrically consistent with the normal map.
Preserve silhouette, camera viewpoint, object boundaries, material identity, and colors.
Do not add text, background, new parts, lighting effects, shadows, geometry changes, or normal-map colors.
Return only the repaired albedo/base-color view."""

DEFAULT_TEACHER_PROMPT_WITH_NORMAL_AND_SOURCE = """Use the first input image as the current albedo/base-color render of a generated 3D asset view.
Use the second input image as a camera-space normal map that defines the exact visible geometry.
Use the third input image as the original source/reference image that the 3D asset was generated from.
Generate a new improved albedo/base-color image only: make the visible texture look more like the source/reference image while staying pixel-aligned to the generated render and geometrically consistent with the normal map.
Preserve the generated render's silhouette, camera viewpoint, object boundaries, geometry, and visible part layout.
Do not add text, background, new parts, lighting effects, shadows, geometry changes, normal-map colors, or view changes.
Return only the repaired albedo/base-color view."""

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS


@dataclass
class InputRecord:
    item_id: str
    safe_id: str
    image_path: str
    seed: int
    split: str = "train"
    notes: str = ""


def setup_logging(output_dir: str, verbose: bool = False):
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    logger = logging.getLogger("teacher_dataset")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG if verbose else logging.INFO)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(os.path.join(output_dir, "logs", "build_teacher_dataset.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def append_jsonl(path: str, record: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fp:
        fp.write(json.dumps(record) + "\n")


def sanitize_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value[:160] or hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def read_inputs(args) -> List[InputRecord]:
    records = []
    if args.input_manifest:
        path = Path(args.input_manifest)
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            rows = pd.read_csv(path).to_dict("records")
        for index, row in enumerate(rows):
            item_id = str(row.get("id") or Path(str(row["image_path"])).stem)
            image_path = str(row["image_path"])
            if not os.path.isabs(image_path):
                image_path = str((path.parent / image_path).resolve())
            records.append(InputRecord(
                item_id=item_id,
                safe_id=sanitize_id(item_id),
                image_path=image_path,
                seed=int(row.get("seed", args.seed + index)),
                split=str(row.get("split", "train")),
                notes=str(row.get("notes", "")),
            ))
    elif args.input_image_dir:
        image_dir = Path(args.input_image_dir)
        image_paths = sorted([
            p for p in image_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ])
        for index, image_path in enumerate(image_paths):
            records.append(InputRecord(
                item_id=image_path.stem,
                safe_id=sanitize_id(image_path.stem),
                image_path=str(image_path),
                seed=args.seed + index,
            ))
    else:
        raise ValueError("Provide --input_manifest or --input_image_dir")
    return records


def ensure_dirs(output_dir: str, shape_name: str, teacher_name: str):
    for rel in [
        "source_images",
        "intermediate_images",
        "mesh_npz",
        "pbr_voxels/original",
        "pbr_voxels/fused",
        f"shape_latents/{shape_name}",
        f"pbr_latents/{teacher_name}",
        f"loss_weights/{teacher_name}",
        "teacher_views",
        "renders",
    ]:
        os.makedirs(os.path.join(output_dir, rel), exist_ok=True)


def pil_to_chw(image: Image.Image, device="cuda", include_alpha=False) -> torch.Tensor:
    if include_alpha:
        image = image.convert("RGBA")
    else:
        image = image.convert("RGB")
    arr = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).to(device)


def chw_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().float().clamp(0, 1)
    if tensor.ndim == 2:
        arr = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")
    arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def image_copy(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.copy()


def fit_image_tile(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    image = ImageOps.contain(image.convert("RGBA"), size, RESAMPLE_LANCZOS)
    tile = Image.new("RGB", size, (255, 255, 255))
    tile.paste(image.convert("RGB"), ((size[0] - image.width) // 2, (size[1] - image.height) // 2), image.getchannel("A"))
    return tile


def save_image_sheet(
    items: List[Tuple[str, Image.Image]],
    output_path: str,
    *,
    tile_size: Tuple[int, int] = (256, 256),
    header_height: int = 28,
):
    if not items:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    font = ImageFont.load_default()
    tiles = []
    for label, image in items:
        tile = Image.new("RGB", (tile_size[0], tile_size[1] + header_height), (242, 242, 242))
        tile.paste(fit_image_tile(image, tile_size), (0, header_height))
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, tile.width - 1, header_height - 1), fill=(232, 232, 232))
        draw.text((8, 8), label[:64], fill=(20, 20, 20), font=font)
        tiles.append(tile)

    sheet = Image.new("RGB", (tile_size[0] * len(tiles), tile_size[1] + header_height), (255, 255, 255))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (index * tile_size[0], 0))
    sheet.save(output_path)


def save_vertical_stack(image_paths: List[str], output_path: str, *, gap: int = 8):
    images = [image_copy(path).convert("RGB") for path in image_paths if os.path.exists(path)]
    if not images:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width = max(image.width for image in images)
    height = sum(image.height for image in images) + gap * (len(images) - 1)
    stack = Image.new("RGB", (width, height), (255, 255, 255))
    y = 0
    for image in images:
        stack.paste(image, ((width - image.width) // 2, y))
        y += image.height + gap
    stack.save(output_path)


def intermediate_dir(output_dir: str, safe_id: str) -> str:
    return os.path.join(output_dir, "intermediate_images", safe_id)


def save_sparse(path: str, tensor: SparseTensor):
    coords = tensor.coords[:, 1:].detach().cpu().numpy()
    dtype = np.uint8 if coords.max(initial=0) < 256 else np.uint16
    np.savez_compressed(
        path,
        feats=tensor.feats.detach().cpu().numpy().astype(np.float32),
        coords=coords.astype(dtype),
    )


def align_sparse_like(source: SparseTensor, reference: SparseTensor, fill: Optional[torch.Tensor] = None) -> SparseTensor:
    if torch.equal(source.coords, reference.coords):
        return source
    fill_feats = fill if fill is not None else torch.zeros_like(reference.feats)
    source_map = {
        tuple(coord.tolist()): feat
        for coord, feat in zip(source.coords.detach().cpu(), source.feats.detach().cpu())
    }
    aligned = fill_feats.detach().cpu().clone()
    for i, coord in enumerate(reference.coords.detach().cpu()):
        feat = source_map.get(tuple(coord.tolist()))
        if feat is not None:
            aligned[i] = feat
    return SparseTensor(aligned.to(reference.feats.device), reference.coords.detach().clone())


def load_envmap() -> EnvMap:
    import cv2

    image = cv2.cvtColor(
        cv2.imread("assets/hdri/forest.exr", cv2.IMREAD_UNCHANGED),
        cv2.COLOR_BGR2RGB,
    )
    return EnvMap(torch.tensor(image, dtype=torch.float32, device="cuda"))


def get_teacher_views(num_views: int, radius: float, fov: float, offset: Tuple[float, float]):
    yaws = []
    pitchs = []
    for i in range(num_views):
        yaw, pitch = sphere_hammersley_sequence(i, num_views, offset=offset)
        yaws.append(yaw)
        pitchs.append(pitch)
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, radius, fov)
    return [
        {
            "index": i,
            "yaw": float(yaws[i]),
            "pitch": float(pitchs[i]),
            "radius": float(radius),
            "fov": float(fov),
            "extrinsics": extrinsics[i],
            "intrinsics": intrinsics[i],
        }
        for i in range(num_views)
    ]


def render_view(renderer, envmap, mesh, view):
    return renderer.render(mesh, view["extrinsics"], view["intrinsics"], envmap=envmap)


def teacher_prompt(args, has_normal_reference: bool, has_source_reference: bool) -> str:
    if args.teacher_prompt is not None:
        return args.teacher_prompt
    if has_normal_reference and has_source_reference:
        return DEFAULT_TEACHER_PROMPT_WITH_NORMAL_AND_SOURCE
    if has_normal_reference:
        return DEFAULT_TEACHER_PROMPT_WITH_NORMAL
    if has_source_reference:
        return DEFAULT_TEACHER_PROMPT_WITH_SOURCE
    return DEFAULT_TEACHER_PROMPT


def call_gpt_image_teacher(
    input_path: str,
    output_path: str,
    args,
    normal_path: Optional[str] = None,
    source_path: Optional[str] = None,
):
    if args.skip_teacher:
        Image.open(input_path).save(output_path)
        return

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the openai package or pass --skip_teacher for a dry run") from exc

    client = OpenAI(api_key=args.openai_api_key or None)
    with ExitStack() as stack:
        image_file = stack.enter_context(open(input_path, "rb"))
        image_input = [image_file]
        if normal_path is not None:
            normal_file = stack.enter_context(open(normal_path, "rb"))
            image_input.append(normal_file)
        if source_path is not None:
            source_file = stack.enter_context(open(source_path, "rb"))
            image_input.append(source_file)
        if len(image_input) == 1:
            image_input = image_input[0]
        request = {
            "model": args.teacher_model,
            "image": image_input,
            "prompt": teacher_prompt(args, normal_path is not None, source_path is not None),
            "size": args.teacher_size,
            "quality": args.teacher_quality,
            "output_format": "png",
        }
        try:
            result = client.images.edit(**request)
        except TypeError:
            request.pop("output_format", None)
            result = client.images.edit(**request)

    image_data = result.data[0]
    if getattr(image_data, "b64_json", None):
        decoded = base64.b64decode(image_data.b64_json)
        with open(output_path, "wb") as out:
            out.write(decoded)
    elif getattr(image_data, "url", None):
        raise RuntimeError("Teacher returned a URL; configure the client for base64 image output.")
    else:
        raise RuntimeError("Teacher image response did not include b64_json")


def image_alignment_score(original_rgb: torch.Tensor, original_mask: torch.Tensor, teacher: Image.Image, args):
    teacher_rgba = pil_to_chw(teacher.resize((original_rgb.shape[-1], original_rgb.shape[-2])), include_alpha=True)
    teacher_rgb = teacher_rgba[:3]
    teacher_alpha = teacher_rgba[3:4]
    teacher_has_alpha = teacher.mode == "RGBA" and np.any(np.array(teacher.getchannel("A")) < 250)

    mask = original_mask.float().clamp(0, 1).unsqueeze(0) if original_mask.ndim == 2 else original_mask.float().clamp(0, 1)
    if teacher_has_alpha:
        inter = torch.minimum(mask, teacher_alpha).sum()
        union = torch.maximum(mask, teacher_alpha).sum().clamp_min(1e-8)
        iou = (inter / union).item()
    else:
        iou = 1.0

    delta = (teacher_rgb - original_rgb).abs().mul(mask).sum() / (mask.sum().clamp_min(1e-8) * 3)
    delta = float(delta.item())
    accepted = iou >= args.min_mask_iou and delta <= args.max_teacher_delta
    score = max(0.0, min(1.0, 1.0 - delta / max(args.max_teacher_delta, 1e-6))) * iou
    return accepted, score, teacher_rgb, teacher_alpha


def save_teacher_view_intermediates(
    out_dir: str,
    record: InputRecord,
    view: dict,
    source: Image.Image,
    render_path: str,
    mask_path: str,
    normal_path: Optional[str],
    teacher_path: str,
    accepted: bool,
    score: float,
):
    inspect_dir = intermediate_dir(out_dir, record.safe_id)
    os.makedirs(inspect_dir, exist_ok=True)
    source.save(os.path.join(inspect_dir, "source.png"))

    prefix = f"view_{view['index']:02d}"
    render_image = image_copy(render_path)
    mask_image = image_copy(mask_path)
    teacher_image = image_copy(teacher_path)
    teacher_image.save(os.path.join(inspect_dir, f"{prefix}_gpt_output.png"))
    render_image.save(os.path.join(inspect_dir, f"{prefix}_gpt_input_base_color.png"))
    mask_image.save(os.path.join(inspect_dir, f"{prefix}_mask.png"))

    sheet_items = [
        ("source", source),
        ("GPT input albedo", render_image),
    ]
    if normal_path is not None and os.path.exists(normal_path):
        normal_image = image_copy(normal_path)
        normal_image.save(os.path.join(inspect_dir, f"{prefix}_normal_reference.png"))
        sheet_items.append(("normal reference", normal_image))
    sheet_items.extend([
        ("mask", mask_image),
        (f"GPT output {'accepted' if accepted else 'rejected'} {score:.3f}", teacher_image),
    ])
    save_image_sheet(sheet_items, os.path.join(inspect_dir, f"{prefix}_gpt_compare.png"))


def save_fused_intermediates(
    out_dir: str,
    record: InputRecord,
    mesh,
    fused_attrs: torch.Tensor,
    views: List[dict],
    view_records: List[dict],
    renderer,
    envmap,
    source: Image.Image,
):
    inspect_dir = intermediate_dir(out_dir, record.safe_id)
    os.makedirs(inspect_dir, exist_ok=True)

    fused_mesh = mesh.to(mesh.device)
    fused_mesh.attrs = fused_attrs.detach().to(mesh.attrs.device)
    view_records_by_index = {int(row["view_index"]): row for row in view_records}
    comparison_paths = []

    for view in views:
        prefix = f"view_{view['index']:02d}"
        buffers = render_view(renderer, envmap, fused_mesh, view)
        fused_base = chw_to_pil(buffers["base_color"])
        fused_shaded = chw_to_pil(buffers["shaded"])
        fused_base_path = os.path.join(inspect_dir, f"{prefix}_fused_base_color.png")
        fused_shaded_path = os.path.join(inspect_dir, f"{prefix}_fused_shaded.png")
        fused_base.save(fused_base_path)
        fused_shaded.save(fused_shaded_path)

        render_path = os.path.join(out_dir, "renders", record.safe_id, f"{prefix}_base_color.png")
        teacher_path = os.path.join(out_dir, "teacher_views", record.safe_id, f"{prefix}_teacher.png")
        view_record = view_records_by_index.get(int(view["index"]), {})
        status = "accepted" if view_record.get("accepted") else "rejected"
        score = float(view_record.get("alignment_score", 0.0))

        sheet_items = [("source", source)]
        if os.path.exists(render_path):
            sheet_items.append(("GPT input albedo", image_copy(render_path)))
        if os.path.exists(teacher_path):
            sheet_items.append((f"GPT output {status} {score:.3f}", image_copy(teacher_path)))
        sheet_items.extend([
            ("fused albedo", fused_base),
            ("fused shaded", fused_shaded),
        ])
        compare_path = os.path.join(inspect_dir, f"{prefix}_projection_compare.png")
        save_image_sheet(sheet_items, compare_path)
        comparison_paths.append(compare_path)

    save_vertical_stack(comparison_paths, os.path.join(inspect_dir, "summary.png"))


def sample_chw(chw: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    if chw.ndim == 2:
        chw = chw.unsqueeze(0)
    sampled = F.grid_sample(
        chw.unsqueeze(0),
        grid.view(1, -1, 1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled[0, :, :, 0].transpose(0, 1)


def project_teacher_to_voxels(mesh, teacher_rgb, teacher_alpha, buffers, view, args, alignment_score: float):
    coords = mesh.coords.float().cuda()
    xyz = mesh.origin.to(coords.device) + (coords + 0.5) * mesh.voxel_size
    xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
    cam = xyz_h @ view["extrinsics"].transpose(0, 1)
    z = cam[:, 2:3]
    intr = view["intrinsics"]
    grid = torch.cat([
        cam[:, 0:1] * intr[0, 0] / z.clamp_min(1e-6) + intr[0, 2],
        cam[:, 1:2] * intr[1, 1] / z.clamp_min(1e-6) + intr[1, 2],
    ], dim=1)
    in_frame = (z[:, 0] > 0) & (grid.abs() <= 1).all(dim=1)

    mask = sample_chw(buffers["mask"].float(), grid)[:, :1].clamp(0, 1)
    alpha = sample_chw(teacher_alpha.float(), grid)[:, :1].clamp(0, 1)
    rgb = sample_chw(teacher_rgb.float(), grid)[:, :3].clamp(0, 1)

    visible = in_frame.float().unsqueeze(1)
    if "depth" in buffers:
        depth = sample_chw(buffers["depth"].float(), grid)[:, :1]
        visible = visible * (depth - z).abs().lt(args.depth_tolerance).float()

    normal_conf = torch.ones_like(visible)
    if "normal" in buffers:
        normal = sample_chw(buffers["normal"].float(), grid)[:, :3]
        cam_normal_z = -((normal[:, 2:3] - 0.5) * 2.0)
        normal_conf = cam_normal_z.abs().clamp(args.min_grazing_weight, 1.0)

    weight = visible * mask * alpha * normal_conf * float(alignment_score)
    return rgb, weight.clamp_min(0.0)


def fuse_teacher_views(mesh, projections, base_color_slice):
    attrs = mesh.attrs.detach().clone()
    numerator = torch.zeros(attrs.shape[0], 3, dtype=attrs.dtype, device=attrs.device)
    denominator = torch.zeros(attrs.shape[0], 1, dtype=attrs.dtype, device=attrs.device)
    for rgb, weight in projections:
        numerator += rgb.to(attrs.dtype) * weight.to(attrs.dtype)
        denominator += weight.to(attrs.dtype)
    observed = denominator[:, 0] > 1e-6
    if observed.any():
        attrs[observed, base_color_slice] = (numerator[observed] / denominator[observed].clamp_min(1e-6)).clamp(0, 1)
    confidence = denominator / denominator.max().clamp_min(1e-6)
    return attrs, confidence.clamp(0, 1)


def latent_weights_from_voxels(
    voxel_coords: torch.Tensor,
    voxel_conf: torch.Tensor,
    latent_coords: torch.Tensor,
    *,
    resolution: int,
    visible_weight: float,
) -> torch.Tensor:
    if voxel_conf.max() <= 0:
        return torch.ones(latent_coords.shape[0], 1, dtype=torch.float32)

    spatial = int(latent_coords.max().item() + 1)
    factor = max(1, int(round(resolution / spatial)))
    voxel_keys = (voxel_coords.long() // factor).clamp(min=0, max=spatial - 1)
    latent_keys = latent_coords.long().clamp(min=0, max=spatial - 1)
    flat_mul = torch.tensor([spatial * spatial, spatial, 1], device=voxel_keys.device, dtype=torch.long)
    voxel_flat = (voxel_keys * flat_mul).sum(dim=1)
    latent_flat = (latent_keys * flat_mul).sum(dim=1)
    conf = voxel_conf[:, 0].float().to(voxel_flat.device)

    try:
        unique, inverse = torch.unique(voxel_flat, return_inverse=True)
        aggregated = torch.zeros(unique.shape[0], dtype=torch.float32, device=voxel_flat.device)
        aggregated.scatter_reduce_(0, inverse, conf, reduce="amax", include_self=False)
        order = torch.argsort(unique)
        unique = unique[order]
        aggregated = aggregated[order]
        pos = torch.searchsorted(unique, latent_flat)
        matched = (pos < unique.numel()) & (unique[pos.clamp(max=unique.numel() - 1)] == latent_flat)
        latent_conf = torch.zeros(latent_flat.shape[0], dtype=torch.float32, device=voxel_flat.device)
        latent_conf[matched] = aggregated[pos[matched]]
    except Exception:
        conf_map = {}
        for key, value in zip(voxel_flat.cpu().tolist(), conf.cpu().tolist()):
            conf_map[key] = max(conf_map.get(key, 0.0), value)
        latent_conf = torch.tensor([conf_map.get(k, 0.0) for k in latent_flat.cpu().tolist()], dtype=torch.float32)

    weights = 1.0 + (float(visible_weight) - 1.0) * latent_conf.cpu().clamp(0, 1)
    return weights.reshape(-1, 1)


@torch.no_grad()
def process_record(record: InputRecord, pipeline, tex_encoder, renderer, envmap, args, logger):
    start_time = time.time()
    out_dir = args.output_dir
    final_latent_path = os.path.join(out_dir, "pbr_latents", args.teacher_latent_name, f"{record.safe_id}.npz")
    if args.skip_existing and os.path.exists(final_latent_path):
        final_intermediate_path = os.path.join(intermediate_dir(out_dir, record.safe_id), "summary.png")
        if args.no_save_intermediate_images or os.path.exists(final_intermediate_path):
            logger.info("skip_existing id=%s latent=%s", record.safe_id, final_latent_path)
            return None
        logger.info(
            "skip_existing_latent_without_intermediate_images id=%s latent=%s",
            record.safe_id,
            final_latent_path,
        )

    source = Image.open(record.image_path)
    source_save_path = os.path.join(out_dir, "source_images", f"{record.safe_id}.png")
    source.save(source_save_path)
    if not args.no_save_intermediate_images:
        inspect_dir = intermediate_dir(out_dir, record.safe_id)
        os.makedirs(inspect_dir, exist_ok=True)
        source.save(os.path.join(inspect_dir, "source.png"))

    logger.info("generate_start id=%s seed=%s image=%s", record.safe_id, record.seed, record.image_path)
    try:
        meshes, latents = pipeline.run(
            source,
            seed=record.seed,
            pipeline_type="512",
            return_latent=True,
            preprocess_image=not args.no_preprocess_image,
            tex_slat_sampler_params={"steps": args.tex_sampling_steps, "guidance_strength": args.tex_guidance_scale},
        )
    except Exception as exc:
        logger.exception("generate_failed id=%s", record.safe_id)
        append_jsonl(os.path.join(out_dir, "logs", "failures.jsonl"), {
            "id": record.safe_id,
            "stage": "generate",
            "error": repr(exc),
        })
        if args.keep_going:
            return None
        raise
    shape_slat, original_tex_slat, resolution = latents
    mesh = meshes[0].to("cuda")

    save_sparse(os.path.join(out_dir, "shape_latents", args.shape_latent_name, f"{record.safe_id}.npz"), shape_slat)
    save_sparse(os.path.join(out_dir, "pbr_voxels", "original", f"{record.safe_id}.npz"), SparseTensor(
        mesh.attrs.detach().cpu(),
        torch.cat([torch.zeros_like(mesh.coords[:, :1]), mesh.coords.cpu()], dim=1),
    ))
    np.savez_compressed(
        os.path.join(out_dir, "mesh_npz", f"{record.safe_id}.npz"),
        vertices=mesh.vertices.detach().cpu().numpy().astype(np.float32),
        faces=mesh.faces.detach().cpu().numpy().astype(np.int32),
    )

    views = get_teacher_views(args.num_views, args.radius, args.fov, tuple(args.view_offset))
    projections = []
    view_records = []
    for view in views:
        view_start = time.time()
        buffers = render_view(renderer, envmap, mesh, view)
        view_dir = os.path.join(out_dir, "teacher_views", record.safe_id)
        render_dir = os.path.join(out_dir, "renders", record.safe_id)
        os.makedirs(view_dir, exist_ok=True)
        os.makedirs(render_dir, exist_ok=True)

        render_path = os.path.join(render_dir, f"view_{view['index']:02d}_base_color.png")
        mask_path = os.path.join(render_dir, f"view_{view['index']:02d}_mask.png")
        normal_path = os.path.join(render_dir, f"view_{view['index']:02d}_normal.png")
        teacher_path = os.path.join(view_dir, f"view_{view['index']:02d}_teacher.png")
        chw_to_pil(buffers["base_color"]).save(render_path)
        chw_to_pil(buffers["mask"]).save(mask_path)
        if args.include_normal_reference:
            chw_to_pil(buffers["normal"]).save(normal_path)
        else:
            normal_path = None

        source_reference_path = None if args.no_source_reference else source_save_path
        try:
            call_gpt_image_teacher(
                render_path,
                teacher_path,
                args,
                normal_path=normal_path,
                source_path=source_reference_path,
            )
        except Exception as exc:
            logger.exception("teacher_failed id=%s view=%s", record.safe_id, view["index"])
            append_jsonl(os.path.join(out_dir, "logs", "failures.jsonl"), {
                "id": record.safe_id,
                "stage": "teacher",
                "view_index": view["index"],
                "error": repr(exc),
            })
            if args.keep_going:
                view_records.append({
                    "view_index": view["index"],
                    "yaw": view["yaw"],
                    "pitch": view["pitch"],
                    "accepted": False,
                    "alignment_score": 0.0,
                    "normal_reference": bool(args.include_normal_reference),
                    "source_reference": not bool(args.no_source_reference),
                    "error": repr(exc),
                    "elapsed_sec": time.time() - view_start,
                })
                continue
            raise
        teacher = Image.open(teacher_path)
        accepted, score, teacher_rgb, teacher_alpha = image_alignment_score(
            buffers["base_color"],
            buffers["mask"],
            teacher,
            args,
        )
        if not args.no_save_intermediate_images:
            save_teacher_view_intermediates(
                out_dir,
                record,
                view,
                source,
                render_path,
                mask_path,
                normal_path,
                teacher_path,
                accepted,
                score,
            )
        if accepted:
            projections.append(project_teacher_to_voxels(mesh, teacher_rgb, teacher_alpha, buffers, view, args, score))
        view_records.append({
            "view_index": view["index"],
            "yaw": view["yaw"],
            "pitch": view["pitch"],
            "accepted": bool(accepted),
            "alignment_score": float(score),
            "normal_reference": bool(args.include_normal_reference),
            "source_reference": not bool(args.no_source_reference),
            "elapsed_sec": time.time() - view_start,
        })
        logger.info(
            "view_done id=%s view=%s accepted=%s score=%.4f elapsed=%.1fs",
            record.safe_id,
            view["index"],
            accepted,
            score,
            time.time() - view_start,
        )

    if projections:
        fused_attrs, voxel_conf = fuse_teacher_views(mesh, projections, mesh.layout["base_color"])
    else:
        fused_attrs = mesh.attrs.detach().clone()
        voxel_conf = torch.zeros(mesh.attrs.shape[0], 1, dtype=mesh.attrs.dtype, device=mesh.attrs.device)

    if not args.no_save_intermediate_images:
        save_fused_intermediates(
            out_dir,
            record,
            mesh,
            fused_attrs,
            views,
            view_records,
            renderer,
            envmap,
            source,
        )

    fused_voxel = SparseTensor(
        feats=fused_attrs.detach().float() * 2.0 - 1.0,
        coords=torch.cat([torch.zeros_like(mesh.coords[:, :1]), mesh.coords], dim=1).int(),
    )
    teacher_z = tex_encoder(fused_voxel.cuda())
    teacher_z = align_sparse_like(teacher_z, original_tex_slat, fill=original_tex_slat.feats)
    save_sparse(final_latent_path, teacher_z)

    weights = latent_weights_from_voxels(
        mesh.coords.detach(),
        voxel_conf.detach(),
        teacher_z.coords[:, 1:].detach(),
        resolution=args.resolution,
        visible_weight=args.visible_weight,
    )
    np.savez_compressed(
        os.path.join(out_dir, "loss_weights", args.teacher_latent_name, f"{record.safe_id}.npz"),
        feats=weights.numpy().astype(np.float32),
        coords=teacher_z.coords[:, 1:].detach().cpu().numpy().astype(np.uint8),
    )
    np.savez_compressed(
        os.path.join(out_dir, "pbr_voxels", "fused", f"{record.safe_id}.npz"),
        feats=fused_attrs.detach().cpu().numpy().astype(np.float32),
        coords=mesh.coords.detach().cpu().numpy().astype(np.uint16),
        confidence=voxel_conf.detach().cpu().numpy().astype(np.float32),
    )

    with open(os.path.join(out_dir, "teacher_views", record.safe_id, "views.json"), "w") as fp:
        json.dump(view_records, fp, indent=2)

    accepted_views = sum(1 for view in view_records if view["accepted"])
    coverage = float((voxel_conf[:, 0] > 1e-6).float().mean().item())
    row = {
        "sha256": record.safe_id,
        "id": record.item_id,
        "image_path": record.image_path,
        "source_image_path": os.path.relpath(source_save_path, out_dir),
        "intermediate_image_dir": None if args.no_save_intermediate_images else os.path.relpath(
            intermediate_dir(out_dir, record.safe_id),
            out_dir,
        ),
        "seed": record.seed,
        "split": record.split,
        "notes": record.notes,
        "shape_latent_encoded": True,
        "teacher_latent_encoded": True,
        "pbr_latent_encoded": True,
        "shape_latent_tokens": int(shape_slat.coords.shape[0]),
        "pbr_latent_tokens": int(teacher_z.coords.shape[0]),
        "teacher_accepted_views": int(accepted_views),
        "teacher_projected_voxel_fraction": coverage,
        "build_elapsed_sec": time.time() - start_time,
    }
    append_jsonl(os.path.join(out_dir, "logs", "records.jsonl"), row)
    logger.info(
        "record_done id=%s accepted_views=%s/%s coverage=%.4f tokens=%s elapsed=%.1fs",
        record.safe_id,
        accepted_views,
        len(view_records),
        coverage,
        int(teacher_z.coords.shape[0]),
        row["build_elapsed_sec"],
    )
    return row


def write_metadata(output_dir: str, rows: List[dict]):
    path = os.path.join(output_dir, "metadata.csv")
    existing = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
    by_id = {row["sha256"]: row for row in existing if "sha256" in row}
    for row in rows:
        by_id[row["sha256"]] = row
    pd.DataFrame.from_records(list(by_id.values())).to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def write_run_config(output_dir: str, args, records: List[InputRecord]):
    payload = {
        "argv": sys.argv,
        "num_inputs": len(records),
        "args": {
            key: value
            for key, value in vars(args).items()
            if key not in {"openai_api_key"}
        },
    }
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    with open(os.path.join(output_dir, "logs", "run_config.json"), "w") as fp:
        json.dump(payload, fp, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Build a multi-view GPT Image 2 teacher dataset for TRELLIS texture LoRA.")
    parser.add_argument("--input_manifest", type=str, default=None)
    parser.add_argument("--input_image_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="microsoft/TRELLIS.2-4B")
    parser.add_argument("--tex_encoder", type=str, default="microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16")
    parser.add_argument("--teacher_model", type=str, default="gpt-image-2")
    parser.add_argument("--teacher_size", type=str, default="auto")
    parser.add_argument("--teacher_quality", type=str, default="high")
    parser.add_argument("--teacher_prompt", type=str, default=None)
    parser.add_argument("--include_normal_reference", action="store_true",
                        help="Send the rendered normal map with the albedo render to the GPT Image teacher.")
    parser.add_argument("--no_source_reference", action="store_true",
                        help="Do not send the original input/source image to the GPT Image teacher.")
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--skip_teacher", action="store_true", help="Dry-run by copying the original render as the teacher.")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--no_save_intermediate_images", action="store_true",
                        help="Disable per-view GPT input/output images, projection previews, and comparison sheets.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--view_offset", type=float, nargs=2, default=[0.0, 0.0])
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--fov", type=float, default=40.0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--render_resolution", type=int, default=512)
    parser.add_argument("--shape_latent_name", type=str, default="shape_enc_next_dc_f16c32_fp16_512")
    parser.add_argument("--teacher_latent_name", type=str, default="tex_teacher_multiview_512")
    parser.add_argument("--visible_weight", type=float, default=4.0)
    parser.add_argument("--min_mask_iou", type=float, default=0.9)
    parser.add_argument("--max_teacher_delta", type=float, default=0.65)
    parser.add_argument("--depth_tolerance", type=float, default=0.04)
    parser.add_argument("--min_grazing_weight", type=float, default=0.15)
    parser.add_argument("--tex_sampling_steps", type=int, default=12)
    parser.add_argument("--tex_guidance_scale", type=float, default=7.5)
    parser.add_argument("--no_preprocess_image", action="store_true")
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--keep_going", action="store_true", help="Log per-record failures and continue building the rest of the dataset.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs(args.output_dir, args.shape_latent_name, args.teacher_latent_name)
    logger = setup_logging(args.output_dir, args.verbose)
    records = read_inputs(args)
    if args.max_items is not None:
        records = records[:args.max_items]
    write_run_config(args.output_dir, args, records)
    logger.info(
        "run_start inputs=%s output_dir=%s skip_teacher=%s normal_reference=%s source_reference=%s",
        len(records),
        args.output_dir,
        args.skip_teacher,
        args.include_normal_reference,
        not args.no_source_reference,
    )

    torch.set_grad_enabled(False)
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model_path)
    pipeline.cuda()
    tex_encoder = models.from_pretrained(args.tex_encoder).eval().cuda()

    renderer = PbrMeshRenderer({
        "resolution": args.render_resolution,
        "near": 1,
        "far": 100,
        "ssaa": 2,
        "peel_layers": 8,
        "return_geometry_buffers": True,
    })
    envmap = load_envmap()

    rows = []
    for index, record in enumerate(records, 1):
        logger.info("record_start index=%s/%s id=%s", index, len(records), record.safe_id)
        row = process_record(record, pipeline, tex_encoder, renderer, envmap, args, logger)
        if row is not None:
            rows.append(row)
            write_metadata(args.output_dir, rows)
    write_metadata(args.output_dir, rows)
    logger.info("run_done processed=%s output_dir=%s", len(rows), args.output_dir)


if __name__ == "__main__":
    main()

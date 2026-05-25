#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap
from trellis2.utils import render_utils
from trellis2.utils.lora_utils import apply_lora_from_config, load_lora_checkpoint


def read_inputs(input_manifest=None, input_image_dir=None, max_items=None):
    if input_manifest:
        path = Path(input_manifest)
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            rows = pd.read_csv(path).to_dict("records")
        items = []
        for row in rows:
            image_path = str(row["image_path"])
            if not os.path.isabs(image_path):
                image_path = str((path.parent / image_path).resolve())
            items.append((str(row.get("id") or Path(image_path).stem), image_path, int(row.get("seed", 42))))
    else:
        image_paths = sorted([
            p for p in Path(input_image_dir).iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ])
        items = [(p.stem, str(p), 42 + i) for i, p in enumerate(image_paths)]
    return items[:max_items] if max_items is not None else items


def load_envmap():
    import cv2

    image = cv2.cvtColor(
        cv2.imread("assets/hdri/forest.exr", cv2.IMREAD_UNCHANGED),
        cv2.COLOR_BGR2RGB,
    )
    return EnvMap(torch.tensor(image, dtype=torch.float32, device="cuda"))


def save_render_grid(mesh, path: str, envmap, nviews: int):
    frames = render_utils.make_pbr_vis_frames(
        render_utils.render_video(mesh, envmap=envmap, num_frames=nviews)
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(frames[0]).save(path)


def load_lora(pipeline, lora_ckpt):
    payload = torch.load(lora_ckpt, map_location="cpu", weights_only=False)
    summary = apply_lora_from_config(pipeline.models["tex_slat_flow_model_512"], payload.get("lora_config", {}))
    load_lora_checkpoint({"denoiser": pipeline.models["tex_slat_flow_model_512"]}, lora_ckpt, map_location=pipeline.device)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Render baseline vs texture-LoRA TRELLIS outputs for visual QA.")
    parser.add_argument("--input_manifest", type=str, default=None)
    parser.add_argument("--input_image_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--lora_ckpt", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="microsoft/TRELLIS.2-4B")
    parser.add_argument("--max_items", type=int, default=8)
    parser.add_argument("--nviews", type=int, default=8)
    parser.add_argument("--tex_sampling_steps", type=int, default=12)
    parser.add_argument("--tex_guidance_scale", type=float, default=7.5)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input_manifest and not args.input_image_dir:
        raise ValueError("Provide --input_manifest or --input_image_dir")
    items = read_inputs(args.input_manifest, args.input_image_dir, args.max_items)
    envmap = load_envmap()

    rows = []
    sampler_params = {
        "steps": args.tex_sampling_steps,
        "guidance_strength": args.tex_guidance_scale,
    }

    baseline = Trellis2ImageTo3DPipeline.from_pretrained(args.model_path)
    baseline.cuda()
    for item_id, image_path, seed in items:
        image = Image.open(image_path)
        base_mesh = baseline.run(image, seed=seed, pipeline_type="512", tex_slat_sampler_params=sampler_params)[0]
        base_path = os.path.join(args.output_dir, item_id, "baseline_preview.png")
        save_render_grid(base_mesh, base_path, envmap, args.nviews)
        rows.append({
            "id": item_id,
            "image_path": image_path,
            "seed": seed,
            "baseline_preview": base_path,
            "lora_preview": os.path.join(args.output_dir, item_id, "lora_preview.png"),
        })
    del baseline
    torch.cuda.empty_cache()

    lora_pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model_path)
    lora_pipeline.cuda()
    load_lora(lora_pipeline, args.lora_ckpt)
    for row in rows:
        image = Image.open(row["image_path"])
        lora_mesh = lora_pipeline.run(image, seed=int(row["seed"]), pipeline_type="512", tex_slat_sampler_params=sampler_params)[0]
        lora_path = row["lora_preview"]
        save_render_grid(lora_mesh, lora_path, envmap, args.nviews)

    pd.DataFrame.from_records(rows).to_csv(os.path.join(args.output_dir, "eval_manifest.csv"), index=False)


if __name__ == "__main__":
    main()

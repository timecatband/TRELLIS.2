#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap
from trellis2.utils import render_utils
from trellis2.utils.lora_utils import apply_lora_from_config, load_lora_checkpoint


def load_adapter_into_pipeline(pipeline, lora_ckpt: str, model_key: str = "tex_slat_flow_model_512"):
    payload = torch.load(lora_ckpt, map_location="cpu", weights_only=False)
    lora_config = payload.get("lora_config", {})
    summary = apply_lora_from_config(pipeline.models[model_key], lora_config)
    load_lora_checkpoint({"denoiser": pipeline.models[model_key]}, lora_ckpt, map_location=pipeline.device)
    return summary


def load_envmap():
    import cv2

    image = cv2.cvtColor(
        cv2.imread("assets/hdri/forest.exr", cv2.IMREAD_UNCHANGED),
        cv2.COLOR_BGR2RGB,
    )
    return EnvMap(torch.tensor(image, dtype=torch.float32, device="cuda"))


def save_preview(mesh, output_path: str):
    envmap = load_envmap()
    preview = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap, num_frames=4))[0]
    Image.fromarray(preview).save(output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Load a TRELLIS.2 texture LoRA into the 512 texture pipeline.")
    parser.add_argument("--model_path", type=str, default="microsoft/TRELLIS.2-4B")
    parser.add_argument("--lora_ckpt", type=str, required=True)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--output_preview", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tex_sampling_steps", type=int, default=12)
    parser.add_argument("--tex_guidance_scale", type=float, default=7.5)
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model_path)
    pipeline.cuda()
    summary = load_adapter_into_pipeline(pipeline, args.lora_ckpt)
    print(json.dumps({
        "wrapped_modules": summary.wrapped_modules,
        "trainable_params": summary.trainable_params,
        "total_params": summary.total_params,
    }, indent=2))

    if args.image:
        meshes = pipeline.run(
            Image.open(args.image),
            seed=args.seed,
            pipeline_type="512",
            tex_slat_sampler_params={
                "steps": args.tex_sampling_steps,
                "guidance_strength": args.tex_guidance_scale,
            },
        )
        if args.output_preview:
            os.makedirs(os.path.dirname(args.output_preview) or ".", exist_ok=True)
            save_preview(meshes[0], args.output_preview)


if __name__ == "__main__":
    main()

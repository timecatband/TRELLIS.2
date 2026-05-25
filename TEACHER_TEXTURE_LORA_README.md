# Multi-View GPT Image Teacher Texture LoRA

This guide explains how to build a self-generated teacher dataset and train a LoRA adapter for TRELLIS.2 texture generation. The v1 setup uses TRELLIS.2 512-mode assets, renders four views per asset, asks GPT Image 2 to repair each visible albedo view, projects the repaired views back into PBR voxels, encodes teacher texture latents, and trains a LoRA on `tex_slat_flow_model_512`.

## 1. Prepare Input Images

Start with the same kind of reference images you would normally feed to TRELLIS.2. Use clean, centered object images where possible. Alpha-masked PNG/WebP inputs are ideal; RGB images also work because the pipeline can preprocess/remove backgrounds.

Create either an image directory:

```text
datasets/input_images/
  chair_001.png
  toy_car_001.webp
  ...
```

or a manifest CSV/JSONL. CSV columns:

```csv
id,image_path,seed,split,notes
chair_001,/abs/path/chair_001.png,42,train,clean front view
toy_car_001,/abs/path/toy_car_001.webp,43,train,
lamp_holdout,/abs/path/lamp.png,9001,val,held out
```

Required columns are `id` and `image_path`. Optional columns are `seed`, `split`, and `notes`. Relative `image_path` values are resolved relative to the manifest file.

## 2. Build The Teacher Dataset

Run the builder from the repo root. For a dry run that does not call OpenAI, use `--skip_teacher`; it copies the original rendered albedo and validates the TRELLIS/render/projection/encoding path.

```sh
python tools/build_gpt_image_teacher_dataset.py \
  --input_manifest datasets/input_images.csv \
  --output_dir datasets/trellis_teacher_512 \
  --include_normal_reference \
  --max_items 20 \
  --skip_teacher
```

For the real teacher pass, remove `--skip_teacher` and ensure `OPENAI_API_KEY` is set or pass `--openai_api_key`.

```sh
export OPENAI_API_KEY=...

python tools/build_gpt_image_teacher_dataset.py \
  --input_manifest datasets/input_images.csv \
  --output_dir datasets/trellis_teacher_512 \
  --include_normal_reference \
  --teacher_model gpt-image-2 \
  --teacher_quality high
```

What the builder writes:

```text
datasets/trellis_teacher_512/
  metadata.csv
  logs/run_config.json
  logs/build_teacher_dataset.log
  logs/records.jsonl
  logs/failures.jsonl
  source_images/
  renders/<id>/view_00_base_color.png
  renders/<id>/view_00_normal.png
  renders/<id>/view_00_mask.png
  teacher_views/<id>/view_00_teacher.png
  teacher_views/<id>/views.json
  shape_latents/shape_enc_next_dc_f16c32_fp16_512/<id>.npz
  pbr_latents/tex_teacher_multiview_512/<id>.npz
  loss_weights/tex_teacher_multiview_512/<id>.npz
  pbr_voxels/original/<id>.npz
  pbr_voxels/fused/<id>.npz
```

Important builder options:

```sh
--num_views 4
```
Number of GPT Image teacher views per asset.

```sh
--include_normal_reference
```
Sends both the rendered albedo and camera-space normal map to GPT Image 2, asking for a geometry-aligned new albedo.

```sh
--teacher_size auto
```
Uses the GPT Image model’s default flexible sizing. The output is resized back to the render resolution before projection.

```sh
--visible_weight 4.0
```
Controls how strongly teacher-observed latent tokens are weighted during training.

```sh
--min_mask_iou 0.9 --max_teacher_delta 0.65
```
Filters teacher views that drift too far from the original silhouette/alignment.

## 3. Inspect Dataset Quality

Before training, spot-check:

```sh
open datasets/trellis_teacher_512/metadata.csv
open datasets/trellis_teacher_512/renders/<id>/
open datasets/trellis_teacher_512/teacher_views/<id>/
```

Look for:

- `teacher_accepted_views` should usually be at least `1`; higher is better.
- Teacher images should preserve silhouette, viewpoint, and part boundaries.
- Normal-reference runs should improve alignment on high-curvature or detailed geometry.
- Fused targets should not show obvious baked shadows or background colors.

If too many views are rejected, loosen `--min_mask_iou` slightly or improve the teacher prompt. If teacher images change geometry, tighten the prompt or use `--include_normal_reference`.

Builder observability:

```sh
tail -f datasets/trellis_teacher_512/logs/build_teacher_dataset.log
cat datasets/trellis_teacher_512/logs/run_config.json
tail -n 20 datasets/trellis_teacher_512/logs/records.jsonl
tail -n 20 datasets/trellis_teacher_512/logs/failures.jsonl
```

Use `--keep_going` for long runs where one failed asset or teacher call should not stop the whole dataset build. Each failure is written to `logs/failures.jsonl` with the failed stage and view index when applicable.

## 4. Train The Texture LoRA

Use the supplied config:

```sh
python tools/train_texture_lora.py \
  --config configs/gen/slat_flow_imgshape2tex_lora_teacher_512.json \
  --teacher_data_dir datasets/trellis_teacher_512 \
  --output_dir results/texture_lora_teacher_512 \
  --max_steps 1000 \
  --num_gpus 1
```

Useful overrides:

```sh
--lr 5e-5
--batch_size_per_gpu 2
--i_save 500
--i_sample 500
```

The script loads the pretrained 512 texture denoiser, wraps LoRA adapters around sparse DiT attention/MLP linear layers, freezes base weights, and trains adapter-only parameters. Adapter checkpoints are saved under:

```text
results/texture_lora_teacher_512/lora/adapters_step0001000.pt
```

Training observability:

```text
results/texture_lora_teacher_512/logs/run_config.json
results/texture_lora_teacher_512/logs/dataset_summary.json
results/texture_lora_teacher_512/logs/lora_summary.json
results/texture_lora_teacher_512/logs/train_texture_lora.log
results/texture_lora_teacher_512/log.txt
results/texture_lora_teacher_512/tb_logs/
results/texture_lora_teacher_512/samples/
```

Watch progress from the shell:

```sh
tail -f results/texture_lora_teacher_512/logs/train_texture_lora.log
tail -f results/texture_lora_teacher_512/log.txt
```

Key logged loss terms:

- `loss/loss`: weighted training objective.
- `loss/mse_unweighted`: raw flow-matching MSE without teacher weights.
- `loss/mse_teacher_tokens`: MSE on latent tokens touched by projected teacher views.
- `loss/mse_base_tokens`: MSE on unchanged/background tokens.
- `loss/teacher_token_fraction`: fraction of sparse latent tokens receiving teacher emphasis.
- `loss/teacher_weight_mean` and `loss/teacher_weight_max`: sanity checks for loss weighting.

If `teacher_token_fraction` is near zero, the projection/fusion stage is not covering enough surface area. If `mse_teacher_tokens` explodes while `mse_base_tokens` is stable, inspect the teacher images and fused voxel outputs for that run.

Resume from an adapter checkpoint:

```sh
python tools/train_texture_lora.py \
  --config configs/gen/slat_flow_imgshape2tex_lora_teacher_512.json \
  --teacher_data_dir datasets/trellis_teacher_512 \
  --output_dir results/texture_lora_teacher_512_resume \
  --resume_lora results/texture_lora_teacher_512/lora/adapters_step0001000.pt \
  --max_steps 2000
```

## 5. Apply Or Evaluate The LoRA

Apply an adapter to the local 512 pipeline and optionally render a preview:

```sh
python tools/apply_lora_to_pipeline.py \
  --lora_ckpt results/texture_lora_teacher_512/lora/adapters_step0001000.pt \
  --image datasets/input_images/chair_001.png \
  --output_preview results/texture_lora_teacher_512/chair_preview.png
```

Compare baseline vs LoRA across multiple views:

```sh
python tools/eval_texture_lora.py \
  --input_manifest datasets/input_images.csv \
  --lora_ckpt results/texture_lora_teacher_512/lora/adapters_step0001000.pt \
  --output_dir results/texture_lora_teacher_512/eval \
  --max_items 8 \
  --nviews 8
```

Review:

```text
results/texture_lora_teacher_512/eval/eval_manifest.csv
results/texture_lora_teacher_512/eval/<id>/baseline_preview.png
results/texture_lora_teacher_512/eval/<id>/lora_preview.png
```

## 6. Practical Run Order

For a first pilot:

```sh
# 1. Dry-run dataset machinery on a few images.
python tools/build_gpt_image_teacher_dataset.py \
  --input_manifest datasets/input_images.csv \
  --output_dir datasets/trellis_teacher_512_dryrun \
  --include_normal_reference \
  --max_items 5 \
  --skip_teacher

# 2. Real teacher pass on 10-20 images.
python tools/build_gpt_image_teacher_dataset.py \
  --input_manifest datasets/input_images.csv \
  --output_dir datasets/trellis_teacher_512_pilot \
  --include_normal_reference \
  --max_items 20 \
  --keep_going

# 3. Short smoke train.
python tools/train_texture_lora.py \
  --config configs/gen/slat_flow_imgshape2tex_lora_teacher_512.json \
  --teacher_data_dir datasets/trellis_teacher_512_pilot \
  --output_dir results/texture_lora_teacher_512_pilot \
  --max_steps 100 \
  --i_save 100 \
  --i_sample 100

# 4. Visual comparison.
python tools/eval_texture_lora.py \
  --input_manifest datasets/input_images.csv \
  --lora_ckpt results/texture_lora_teacher_512_pilot/lora/adapters_step0000100.pt \
  --output_dir results/texture_lora_teacher_512_pilot/eval \
  --max_items 8
```

## Notes And Caveats

- This is 512 texture-flow LoRA only. The 1024/cascade texture path is intentionally left for a later experiment.
- V1 distills base color only. Metallic, roughness, and alpha are preserved from TRELLIS.
- The teacher may hallucinate details. The normal-map reference and alignment filters reduce this risk, but human inspection is still important.
- GPT Image calls are per view. With the default `--num_views 4`, each asset makes four image edit requests.
- The local environment must have the full TRELLIS runtime installed, including PyTorch, CUDA dependencies, OpenAI Python SDK, and the TRELLIS model dependencies.

## Debugging Checklist

- Dataset build stops early: rerun with `--keep_going`, then inspect `logs/failures.jsonl`.
- Teacher edits look misaligned: enable `--include_normal_reference`, inspect `renders/<id>/*_normal.png`, and tighten the prompt.
- Too many rejected views: compare `renders/<id>/*_base_color.png` to `teacher_views/<id>/*_teacher.png`, then tune `--min_mask_iou` and `--max_teacher_delta`.
- Low projected coverage: increase `--num_views`, inspect `teacher_projected_voxel_fraction`, and check whether depth/mask renders look correct.
- Training loss is noisy: reduce `--lr`, reduce `--visible_weight`, or start with a smaller pilot dataset and inspect `mse_teacher_tokens` vs `mse_base_tokens`.
- Adapter has no visible effect: confirm `logs/lora_summary.json` has nonzero `wrapped_modules` and `trainable_params`, then evaluate at the same seeds used for the baseline.

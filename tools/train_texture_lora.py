#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
import torch.multiprocessing as mp
from easydict import EasyDict as edict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from trellis2 import datasets, models, trainers
from trellis2.utils.dist_utils import setup_dist
from trellis2.utils.lora_utils import apply_lora_from_config, load_lora_checkpoint


def setup_logging(output_dir: str, rank: int, verbose: bool = False):
    logger = logging.getLogger(f"texture_lora_train.rank{rank}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG if verbose else logging.INFO)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if rank == 0:
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(output_dir, "logs", "train_texture_lora.log"))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def write_json(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fp:
        json.dump(payload, fp, indent=2)


def summarize_teacher_dataset(dataset):
    summary = {
        "class": dataset.__class__.__name__,
        "length": len(dataset),
    }
    metadata = getattr(dataset, "metadata_by_instance", None)
    if metadata:
        rows = list(metadata.values())
        for column in ("teacher_accepted_views", "teacher_projected_voxel_fraction", "pbr_latent_tokens"):
            values = [float(row[column]) for row in rows if column in row and row[column] == row[column]]
            if values:
                summary[column] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                }
    return summary


def setup_rng(rank: int):
    torch.manual_seed(rank)
    torch.cuda.manual_seed_all(rank)
    np.random.seed(rank)
    random.seed(rank)


def get_model_summary(model):
    lines = []
    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
        lines.append(f"{name:<88}{str(tuple(param.shape)):<32}{str(param.dtype):<16}{param.requires_grad}")
    lines.append("")
    lines.append(f"Number of parameters: {total}")
    lines.append(f"Number of trainable parameters: {trainable}")
    return "\n".join(lines)


def build_model(model_cfg):
    if "pretrained" in model_cfg:
        if "name" in model_cfg and "args" in model_cfg:
            model = getattr(models, model_cfg.name)(**model_cfg.args).cuda()
            pretrained = models.from_pretrained(model_cfg["pretrained"]).cuda()
            model.load_state_dict(pretrained.state_dict(), strict=False)
            del pretrained
            torch.cuda.empty_cache()
            return model
        return models.from_pretrained(model_cfg["pretrained"]).cuda()
    return getattr(models, model_cfg.name)(**model_cfg.args).cuda()


def normalize_teacher_data_dir(cfg, teacher_data_dir: str):
    if teacher_data_dir:
        cfg.data_dir = teacher_data_dir
    return cfg


def apply_training_overrides(cfg, args):
    if args.max_steps is not None:
        cfg.trainer.args.max_steps = args.max_steps
    if args.lr is not None:
        cfg.trainer.args.optimizer.args.lr = args.lr
    if args.batch_size_per_gpu is not None:
        cfg.trainer.args.batch_size_per_gpu = args.batch_size_per_gpu
    if args.i_save is not None:
        cfg.trainer.args.i_save = args.i_save
    if args.i_sample is not None:
        cfg.trainer.args.i_sample = args.i_sample
    return cfg


def main_worker(local_rank: int, cfg, args):
    rank = cfg.node_rank * cfg.num_gpus + local_rank
    world_size = cfg.num_nodes * cfg.num_gpus
    logger = setup_logging(cfg.output_dir, rank, args.verbose)
    if world_size > 1:
        setup_dist(rank, local_rank, world_size, cfg.master_addr, cfg.master_port)
    setup_rng(rank)
    logger.info("worker_start rank=%s local_rank=%s world_size=%s", rank, local_rank, world_size)

    dataset = getattr(datasets, cfg.dataset.name)(cfg.data_dir, **cfg.dataset.args)
    dataset_summary = summarize_teacher_dataset(dataset)
    if rank == 0:
        write_json(os.path.join(cfg.output_dir, "logs", "dataset_summary.json"), dataset_summary)
        logger.info("dataset_summary %s", json.dumps(dataset_summary))

    model_dict: Dict[str, torch.nn.Module] = {
        name: build_model(model_cfg)
        for name, model_cfg in cfg.models.items()
    }

    lora_config = deepcopy(cfg.get("lora", {}))
    lora_summaries = {}
    for name, model in model_dict.items():
        if name == "denoiser":
            lora_summaries[name] = apply_lora_from_config(model, lora_config)
        else:
            for param in model.parameters():
                param.requires_grad = False

    resume_step = None
    if args.resume_lora:
        resume_info = load_lora_checkpoint(model_dict, args.resume_lora, map_location="cuda")
        resume_step = resume_info.get("step")
        if rank == 0:
            logger.info("loaded_lora checkpoint=%s step=%s result=%s", args.resume_lora, resume_step, json.dumps(resume_info.get("load_result", {})))

    if rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        lora_payload = {}
        for name, summary in lora_summaries.items():
            lora_payload[name] = {
                "wrapped_modules": summary.wrapped_modules,
                "trainable_params": summary.trainable_params,
                "total_params": summary.total_params,
            }
            logger.info("lora_summary model=%s %s", name, json.dumps(lora_payload[name]))
        write_json(os.path.join(cfg.output_dir, "logs", "lora_summary.json"), lora_payload)
        for name, backbone in model_dict.items():
            summary = get_model_summary(backbone)
            logger.info("model_summary_written model=%s", name)
            with open(os.path.join(cfg.output_dir, f"{name}_model_summary.txt"), "w") as fp:
                print(summary, file=fp)

    cfg.trainer.args.lora_save_adapters = True
    cfg.trainer.args.lora_save_full_checkpoint = not bool(lora_config.get("save_lora_only", True))
    cfg.trainer.args.lora_config = lora_config

    trainer = getattr(trainers, cfg.trainer.name)(
        model_dict,
        dataset,
        **cfg.trainer.args,
        output_dir=cfg.output_dir,
        load_dir="",
        step=None,
    )
    if resume_step is not None and not args.restart_step:
        trainer.step = int(resume_step)
        logger.info("resume_step set_to=%s", trainer.step)

    if args.tryrun:
        logger.info("tryrun_snapshot_dataset_start")
        trainer.snapshot_dataset(batch_size=cfg.trainer.args.get("snapshot_batch_size", 4))
        logger.info("tryrun_done")
        return
    logger.info("training_start max_steps=%s output_dir=%s", trainer.max_steps, cfg.output_dir)
    start = time.time()
    trainer.run()
    logger.info("training_done elapsed_sec=%.1f", time.time() - start)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a LoRA adapter for TRELLIS.2 texture flow from teacher latents.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--teacher_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume_lora", type=str, default=None)
    parser.add_argument("--restart_step", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size_per_gpu", type=int, default=None)
    parser.add_argument("--i_save", type=int, default=None)
    parser.add_argument("--i_sample", type=int, default=None)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=-1)
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="12345")
    parser.add_argument("--tryrun", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.num_gpus = torch.cuda.device_count() if args.num_gpus == -1 else args.num_gpus
    config = json.load(open(args.config, "r"))
    cfg = edict(config)
    cfg.output_dir = args.output_dir
    cfg.load_dir = ""
    cfg.ckpt = "none"
    cfg.node_rank = args.node_rank
    cfg.num_nodes = args.num_nodes
    cfg.num_gpus = args.num_gpus
    cfg.master_addr = args.master_addr
    cfg.master_port = args.master_port
    cfg = normalize_teacher_data_dir(cfg, args.teacher_data_dir)
    cfg = apply_training_overrides(cfg, args)

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "command.txt"), "w") as fp:
        print(" ".join(["python"] + sys.argv), file=fp)
    with open(os.path.join(cfg.output_dir, "config.json"), "w") as fp:
        json.dump(config, fp, indent=4)
    write_json(os.path.join(cfg.output_dir, "logs", "run_config.json"), {
        "argv": sys.argv,
        "teacher_data_dir": args.teacher_data_dir,
        "output_dir": args.output_dir,
        "overrides": {
            "max_steps": args.max_steps,
            "lr": args.lr,
            "batch_size_per_gpu": args.batch_size_per_gpu,
            "i_save": args.i_save,
            "i_sample": args.i_sample,
            "resume_lora": args.resume_lora,
            "restart_step": args.restart_step,
        },
    })

    if cfg.num_gpus > 1:
        mp.spawn(main_worker, args=(cfg, args), nprocs=cfg.num_gpus, join=True)
    else:
        main_worker(0, cfg, args)


if __name__ == "__main__":
    main()

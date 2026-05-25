import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..modules.sparse import SparseTensor, sparse_cat
from ..utils.data_utils import load_balanced_group_indices


def _read_roots(roots: str, *, pbr_latent_name: str, shape_latent_name: str, loss_weight_name: str):
    try:
        parsed = json.loads(roots)
    except Exception:
        parsed = roots.split(",")

    if isinstance(parsed, dict):
        if all(isinstance(value, dict) for value in parsed.values()):
            root_specs = list(parsed.values())
        else:
            root_specs = [parsed]
    else:
        root_specs = [{"base": root} for root in parsed]

    normalized = []
    for spec in root_specs:
        base = spec.get("base", spec.get("root", "."))
        normalized.append({
            "base": base,
            "metadata": spec.get("metadata", os.path.join(base, "metadata.csv")),
            "cond": spec.get("cond", os.path.join(base, "source_images")),
            "shape_latent": spec.get("shape_latent", os.path.join(base, "shape_latents", shape_latent_name)),
            "pbr_latent": spec.get("pbr_latent", os.path.join(base, "pbr_latents", pbr_latent_name)),
            "loss_weight": spec.get("loss_weight", os.path.join(base, "loss_weights", loss_weight_name)),
        })
    return normalized


class ImageConditionedTeacherSLatPbr(Dataset):
    """
    Self-generated TRELLIS texture teacher dataset.

    Expected layout:
      metadata.csv
      source_images/<id>.png
      shape_latents/<shape_latent_name>/<id>.npz
      pbr_latents/<pbr_latent_name>/<id>.npz
      loss_weights/<loss_weight_name>/<id>.npz
    """

    def __init__(
        self,
        roots: str,
        *,
        resolution: int,
        image_size: int = 512,
        pbr_latent_name: str = "tex_teacher_multiview_512",
        shape_latent_name: str = "shape_enc_next_dc_f16c32_fp16_512",
        loss_weight_name: Optional[str] = None,
        pbr_slat_normalization: Optional[dict] = None,
        shape_slat_normalization: Optional[dict] = None,
        attrs: List[str] = ["base_color", "metallic", "roughness", "alpha"],
        max_tokens: int = 8192,
        min_teacher_accepted_views: int = 1,
        split: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.resolution = resolution
        self.image_size = image_size
        self.pbr_latent_name = pbr_latent_name
        self.shape_latent_name = shape_latent_name
        self.loss_weight_name = loss_weight_name or pbr_latent_name
        self.pbr_slat_normalization = pbr_slat_normalization
        self.shape_slat_normalization = shape_slat_normalization
        self.max_tokens = max_tokens
        self.min_teacher_accepted_views = min_teacher_accepted_views
        self.value_range = (0, 1)

        self.roots = _read_roots(
            roots,
            pbr_latent_name=self.pbr_latent_name,
            shape_latent_name=self.shape_latent_name,
            loss_weight_name=self.loss_weight_name,
        )

        self.channels = {
            "base_color": 3,
            "metallic": 1,
            "roughness": 1,
            "emissive": 3,
            "alpha": 1,
        }
        self.layout = {}
        start = 0
        for attr in attrs:
            self.layout[attr] = slice(start, start + self.channels[attr])
            start += self.channels[attr]

        if self.pbr_slat_normalization is not None:
            self.pbr_slat_mean = torch.tensor(self.pbr_slat_normalization["mean"]).reshape(1, -1)
            self.pbr_slat_std = torch.tensor(self.pbr_slat_normalization["std"]).reshape(1, -1)
        if self.shape_slat_normalization is not None:
            self.shape_slat_mean = torch.tensor(self.shape_slat_normalization["mean"]).reshape(1, -1)
            self.shape_slat_std = torch.tensor(self.shape_slat_normalization["std"]).reshape(1, -1)

        self.instances: List[Tuple[Dict[str, str], str]] = []
        self.metadata_by_instance: Dict[str, pd.Series] = {}
        self._stats: Dict[str, Dict[str, int]] = {}
        for root in self.roots:
            metadata = pd.read_csv(root["metadata"])
            if "sha256" not in metadata.columns and "id" in metadata.columns:
                metadata = metadata.rename(columns={"id": "sha256"})
            if split is not None and "split" in metadata.columns:
                metadata = metadata[metadata["split"] == split]

            before = len(metadata)
            if "teacher_latent_encoded" in metadata.columns:
                metadata = metadata[metadata["teacher_latent_encoded"] == True]
            if "teacher_accepted_views" in metadata.columns:
                metadata = metadata[metadata["teacher_accepted_views"] >= min_teacher_accepted_views]
            if "pbr_latent_tokens" in metadata.columns:
                metadata = metadata[metadata["pbr_latent_tokens"] <= max_tokens]

            key = os.path.basename(os.path.abspath(root["base"])) or root["base"]
            self._stats[key] = {
                "Total": before,
                "Usable": len(metadata),
            }
            for _, row in metadata.iterrows():
                instance = str(row["sha256"])
                self.instances.append((root, instance))
                self.metadata_by_instance[instance] = row

    def __len__(self):
        return len(self.instances)

    def __str__(self):
        lines = [self.__class__.__name__, f"  - Total instances: {len(self)}", "  - Sources:"]
        for key, stats in self._stats.items():
            lines.append(f"    - {key}:")
            for stat_name, value in stats.items():
                lines.append(f"      - {stat_name}: {value}")
        return "\n".join(lines)

    def _load_sparse_npz(self, path: str, normalize: Optional[str] = None) -> SparseTensor:
        data = np.load(path)
        coords = torch.tensor(data["coords"]).int()
        coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1)
        feats = torch.tensor(data["feats"]).float()
        if normalize == "pbr" and self.pbr_slat_normalization is not None:
            feats = (feats - self.pbr_slat_mean) / self.pbr_slat_std
        if normalize == "shape" and self.shape_slat_normalization is not None:
            feats = (feats - self.shape_slat_mean) / self.shape_slat_std
        return SparseTensor(feats, coords)

    def _load_loss_weight(self, path: str, target: SparseTensor) -> SparseTensor:
        if not os.path.exists(path):
            return target.replace(torch.ones(target.feats.shape[0], 1, dtype=torch.float32))

        data = np.load(path)
        coords = torch.tensor(data["coords"]).int()
        coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1)
        feats = torch.tensor(data["feats"]).float()
        if torch.equal(coords, target.coords):
            return SparseTensor(feats, coords)

        weight_by_coord = {
            tuple(coord.tolist()): float(feat[0])
            for coord, feat in zip(coords, feats)
        }
        aligned = torch.ones(target.feats.shape[0], 1, dtype=torch.float32)
        for i, coord in enumerate(target.coords):
            aligned[i, 0] = weight_by_coord.get(tuple(coord.tolist()), 1.0)
        return target.replace(aligned)

    def _load_condition_image(self, root: Dict[str, str], instance: str) -> torch.Tensor:
        row = self.metadata_by_instance.get(instance)
        image_path = None
        if row is not None:
            for column in ("source_image_path", "cond_image_path", "image_path"):
                if column in row and isinstance(row[column], str) and row[column]:
                    candidate = row[column]
                    image_path = candidate if os.path.isabs(candidate) else os.path.join(root["base"], candidate)
                    if os.path.exists(image_path):
                        break
        if image_path is None or not os.path.exists(image_path):
            image_path = os.path.join(root["cond"], f"{instance}.png")

        image = Image.open(image_path)
        alpha = None
        if image.mode == "RGBA":
            alpha_np = np.array(image.getchannel("A"))
            if np.any(alpha_np > 0):
                ys, xs = np.nonzero(alpha_np > 0)
                bbox = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
                image = image.crop(bbox)
                alpha = image.getchannel("A")

        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        if alpha is None and image.mode == "RGBA":
            alpha = image.getchannel("A")
        image_rgb = image.convert("RGB")
        image_tensor = torch.tensor(np.array(image_rgb)).permute(2, 0, 1).float() / 255.0
        if alpha is not None:
            alpha = alpha.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            alpha_tensor = torch.tensor(np.array(alpha)).float() / 255.0
            image_tensor = image_tensor * alpha_tensor.unsqueeze(0)
        return image_tensor

    def get_instance(self, root: Dict[str, str], instance: str) -> Dict[str, Any]:
        pbr_z = self._load_sparse_npz(os.path.join(root["pbr_latent"], f"{instance}.npz"), normalize="pbr")
        shape_z = self._load_sparse_npz(os.path.join(root["shape_latent"], f"{instance}.npz"), normalize="shape")
        if not torch.equal(shape_z.coords, pbr_z.coords):
            raise ValueError(f"Shape and PBR latent coords differ for {instance}")

        loss_weight = self._load_loss_weight(os.path.join(root["loss_weight"], f"{instance}.npz"), pbr_z)
        return {
            "x_0": pbr_z,
            "concat_cond": shape_z,
            "loss_weight": loss_weight,
            "cond": self._load_condition_image(root, instance),
        }

    def __getitem__(self, index) -> Dict[str, Any]:
        try:
            root, instance = self.instances[index]
            return self.get_instance(root, instance)
        except Exception as exc:
            print(f"Error loading teacher instance {index}: {exc}")
            return self.__getitem__(np.random.randint(0, len(self)))

    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices([b["x_0"].feats.shape[0] for b in batch], split_size)
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}
            for key in sub_batch[0].keys():
                if isinstance(sub_batch[0][key], torch.Tensor):
                    pack[key] = torch.stack([b[key] for b in sub_batch])
                elif isinstance(sub_batch[0][key], SparseTensor):
                    pack[key] = sparse_cat([b[key] for b in sub_batch], dim=0)
                elif isinstance(sub_batch[0][key], list):
                    pack[key] = sum([b[key] for b in sub_batch], [])
                else:
                    pack[key] = [b[key] for b in sub_batch]
            packs.append(pack)
        return packs[0] if split_size is None else packs

    @torch.no_grad()
    def visualize_sample(self, sample):
        if isinstance(sample, dict) and "cond" in sample:
            return {"cond": sample["cond"]}
        return sample

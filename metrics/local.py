"""
Local metric backends using PyTorch.
Runs on GPU (CUDA) with CPU fallback.
"""
import logging
from pathlib import Path

import torch
import numpy as np

from .base import MetricBackend, MetricResult

logger = logging.getLogger(__name__)

BATCH_SIZE = 32
MAX_SUBMISSION_IMAGES = 1000


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_images(image_dir: Path, max_images: int = 50000):
    from PIL import Image
    supported = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in supported)[:max_images]
    if not paths:
        raise ValueError(f"No images found in {image_dir}")
    images = []
    for p in paths:
        try:
            images.append(Image.open(p).convert("RGB"))
        except Exception as e:
            logger.warning(f"Skipping {p.name}: {e}")
    if not images:
        raise ValueError(f"No valid images loaded from {image_dir}")
    return images


def _inception_transform():
    from torchvision import transforms
    return transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])


def _feed_images(metric, images, real: bool, transform, device):
    for i in range(0, len(images), BATCH_SIZE):
        tensors = torch.stack([transform(img) for img in images[i:i+BATCH_SIZE]]).to(device)
        metric.update(tensors, real=real)


def _extract_inception_features(images, device) -> np.ndarray:
    from torchvision import models, transforms
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    inception = models.inception_v3(weights="DEFAULT")
    inception.fc = torch.nn.Identity()
    inception.eval().to(device)
    features = []
    with torch.no_grad():
        for i in range(0, len(images), BATCH_SIZE):
            batch = torch.stack([transform(img) for img in images[i:i+BATCH_SIZE]]).to(device)
            features.append(inception(batch).cpu().numpy())
    return np.concatenate(features, axis=0)


def _cache_file(cache_dir: Path, name: str, ext: str) -> Path:
    return cache_dir / f"{name}{ext}"


class FIDMetric(MetricBackend):
    @property
    def name(self): return "fid"
    @property
    def display_name(self): return "FID"
    @property
    def is_higher_better(self): return False

    def cache_reference_features(self, reference_dir: Path, cache_dir: Path) -> Path:
        from torchmetrics.image.fid import FrechetInceptionDistance
        device = _get_device()
        fid = FrechetInceptionDistance(normalize=True).to(device)
        ref_images = _load_images(reference_dir)
        _feed_images(fid, ref_images, real=True, transform=_inception_transform(), device=device)
        out = _cache_file(cache_dir, "fid", ".pt")
        torch.save(fid.state_dict(), out)
        logger.info(f"FID cache saved: {len(ref_images)} images")
        return out

    def compute(self, submission_dir: Path, reference_dir: Path,
                cached_ref_features: Path | None = None) -> MetricResult:
        from torchmetrics.image.fid import FrechetInceptionDistance
        device = _get_device()
        fid = FrechetInceptionDistance(normalize=True).to(device)
        transform = _inception_transform()
        cache = _cache_file(cached_ref_features, "fid", ".pt") if cached_ref_features else None

        if cache and cache.exists():
            fid.load_state_dict(torch.load(cache, map_location=device))
            num_ref = int(fid.real_features_num_samples.item())
            logger.info(f"FID: loaded cache ({num_ref} ref images)")
        else:
            ref_images = _load_images(reference_dir)
            _feed_images(fid, ref_images, real=True, transform=transform, device=device)
            num_ref = len(ref_images)

        sub_images = _load_images(submission_dir, max_images=MAX_SUBMISSION_IMAGES)
        _feed_images(fid, sub_images, real=False, transform=transform, device=device)
        score = fid.compute().item()
        logger.info(f"FID: {score:.4f} ({len(sub_images)} sub vs {num_ref} ref)")
        return MetricResult(name=self.name, score=score, is_higher_better=self.is_higher_better,
                            metadata={"num_submission": len(sub_images), "num_reference": num_ref})


class ISMetric(MetricBackend):
    @property
    def name(self): return "is"
    @property
    def display_name(self): return "IS"
    @property
    def is_higher_better(self): return True

    def compute(self, submission_dir: Path, reference_dir: Path,
                cached_ref_features: Path | None = None) -> MetricResult:
        from torchmetrics.image.inception import InceptionScore
        device = _get_device()
        inception = InceptionScore(normalize=True).to(device)
        sub_images = _load_images(submission_dir, max_images=MAX_SUBMISSION_IMAGES)
        transform = _inception_transform()
        for i in range(0, len(sub_images), BATCH_SIZE):
            tensors = torch.stack([transform(img) for img in sub_images[i:i+BATCH_SIZE]]).to(device)
            inception.update(tensors)
        mean, std = inception.compute()
        logger.info(f"IS: {mean.item():.4f} +/- {std.item():.4f}")
        return MetricResult(name=self.name, score=mean.item(), is_higher_better=self.is_higher_better,
                            metadata={"std": std.item(), "num_images": len(sub_images)})


class KIDMetric(MetricBackend):
    @property
    def name(self): return "kid"
    @property
    def display_name(self): return "KID"
    @property
    def is_higher_better(self): return False

    def cache_reference_features(self, reference_dir: Path, cache_dir: Path) -> Path:
        from torchmetrics.image.kid import KernelInceptionDistance
        device = _get_device()
        ref_images = _load_images(reference_dir)
        kid = KernelInceptionDistance(normalize=True, subset_size=min(1000, len(ref_images))).to(device)
        _feed_images(kid, ref_images, real=True, transform=_inception_transform(), device=device)
        out = _cache_file(cache_dir, "kid", ".pt")
        torch.save(kid.state_dict(), out)
        logger.info(f"KID cache saved: {len(ref_images)} images")
        return out

    def compute(self, submission_dir: Path, reference_dir: Path,
                cached_ref_features: Path | None = None) -> MetricResult:
        from torchmetrics.image.kid import KernelInceptionDistance
        device = _get_device()
        sub_images = _load_images(submission_dir, max_images=MAX_SUBMISSION_IMAGES)
        subset_size = min(1000, len(sub_images))
        cache = _cache_file(cached_ref_features, "kid", ".pt") if cached_ref_features else None

        if cache and cache.exists():
            kid = KernelInceptionDistance(normalize=True, subset_size=subset_size).to(device)
            kid.load_state_dict(torch.load(cache, map_location=device))
            logger.info("KID: loaded cache")
        else:
            ref_images = _load_images(reference_dir)
            kid = KernelInceptionDistance(normalize=True, subset_size=min(subset_size, len(ref_images))).to(device)
            _feed_images(kid, ref_images, real=True, transform=_inception_transform(), device=device)

        _feed_images(kid, sub_images, real=False, transform=_inception_transform(), device=device)
        mean, std = kid.compute()
        logger.info(f"KID: {mean.item():.6f} +/- {std.item():.6f}")
        return MetricResult(name=self.name, score=mean.item(), is_higher_better=self.is_higher_better,
                            metadata={"std": std.item(), "num_submission": len(sub_images)})


class TopPRMetric(MetricBackend):
    @property
    def name(self): return "toppr"
    @property
    def display_name(self): return "TopPR"
    @property
    def is_higher_better(self): return True

    def cache_reference_features(self, reference_dir: Path, cache_dir: Path) -> Path:
        device = _get_device()
        ref_images = _load_images(reference_dir)
        features = _extract_inception_features(ref_images, device)
        out = _cache_file(cache_dir, "toppr", ".npy")
        np.save(str(out), features)
        logger.info(f"TopPR cache saved: {features.shape}")
        return out

    def compute(self, submission_dir: Path, reference_dir: Path,
                cached_ref_features: Path | None = None) -> MetricResult:
        from top_pr import compute_top_pr
        device = _get_device()
        sub_images = _load_images(submission_dir, max_images=MAX_SUBMISSION_IMAGES)
        fake_features = _extract_inception_features(sub_images, device)
        cache = _cache_file(cached_ref_features, "toppr", ".npy") if cached_ref_features else None

        if cache and cache.exists():
            real_features = np.load(str(cache))
            logger.info(f"TopPR: loaded cache {real_features.shape}")
        else:
            real_features = _extract_inception_features(_load_images(reference_dir), device)

        result = compute_top_pr(real_features=real_features, fake_features=fake_features, f1_score=True)
        fidelity = float(result.get("fidelity", 0.0))
        diversity = float(result.get("diversity", 0.0))
        f1 = float(result.get("Top_F1", 0.0))
        logger.info(f"TopPR: fidelity={fidelity:.4f} diversity={diversity:.4f} f1={f1:.4f}")
        return MetricResult(name=self.name, score=f1, is_higher_better=self.is_higher_better,
                            metadata={"fidelity": fidelity, "diversity": diversity,
                                      "num_submission": len(sub_images)})


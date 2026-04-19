"""
Local metric backends using PyTorch.
Runs on GPU (CUDA) with CPU fallback.
"""
import logging
from pathlib import Path

import torch

from .base import MetricBackend, MetricResult

logger = logging.getLogger(__name__)

BATCH_SIZE = 32


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
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])


def _feed_images(metric, images, real: bool, transform, device):
    for i in range(0, len(images), BATCH_SIZE):
        batch = images[i:i + BATCH_SIZE]
        tensors = torch.stack([transform(img) for img in batch]).to(device)
        if real:
            metric.update(tensors, real=True)
        else:
            metric.update(tensors, real=False)


class FIDMetric(MetricBackend):
    """Frechet Inception Distance — lower is better."""

    @property
    def name(self): return "fid"

    @property
    def display_name(self): return "FID"

    @property
    def is_higher_better(self): return False

    def cache_reference_features(self, reference_dir: Path, cache_path: Path) -> Path:
        from torchmetrics.image.fid import FrechetInceptionDistance
        device = _get_device()
        fid = FrechetInceptionDistance(normalize=True).to(device)
        ref_images = _load_images(reference_dir)
        _feed_images(fid, ref_images, real=True, transform=_inception_transform(), device=device)
        torch.save(fid.state_dict(), cache_path)
        logger.info(f"FID reference cache saved: {len(ref_images)} images → {cache_path}")
        return cache_path

    def compute(self, submission_dir: Path, reference_dir: Path,
                cached_ref_features: Path | None = None) -> MetricResult:
        from torchmetrics.image.fid import FrechetInceptionDistance
        device = _get_device()
        fid = FrechetInceptionDistance(normalize=True).to(device)
        transform = _inception_transform()

        if cached_ref_features and Path(cached_ref_features).exists():
            fid.load_state_dict(torch.load(cached_ref_features, map_location=device))
            logger.info("FID: loaded cached reference features")
            num_ref = fid.real_features_num_samples.item()
        else:
            ref_images = _load_images(reference_dir)
            _feed_images(fid, ref_images, real=True, transform=transform, device=device)
            num_ref = len(ref_images)

        sub_images = _load_images(submission_dir)
        _feed_images(fid, sub_images, real=False, transform=transform, device=device)

        score = fid.compute().item()
        logger.info(f"FID: {score:.4f} ({len(sub_images)} sub vs {num_ref} ref)")
        return MetricResult(name=self.name, score=score, is_higher_better=self.is_higher_better,
                            metadata={"num_submission": len(sub_images), "num_reference": int(num_ref)})


class ISMetric(MetricBackend):
    """Inception Score — higher is better. Reference-free."""

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
        sub_images = _load_images(submission_dir)
        transform = _inception_transform()
        for i in range(0, len(sub_images), BATCH_SIZE):
            batch = sub_images[i:i + BATCH_SIZE]
            tensors = torch.stack([transform(img) for img in batch]).to(device)
            inception.update(tensors)
        mean, std = inception.compute()
        score = mean.item()
        logger.info(f"IS: {score:.4f} +/- {std.item():.4f} ({len(sub_images)} images)")
        return MetricResult(name=self.name, score=score, is_higher_better=self.is_higher_better,
                            metadata={"std": std.item(), "num_images": len(sub_images)})


class KIDMetric(MetricBackend):
    """Kernel Inception Distance — lower is better."""

    @property
    def name(self): return "kid"

    @property
    def display_name(self): return "KID"

    @property
    def is_higher_better(self): return False

    def cache_reference_features(self, reference_dir: Path, cache_path: Path) -> Path:
        from torchmetrics.image.kid import KernelInceptionDistance
        device = _get_device()
        ref_images = _load_images(reference_dir)
        subset_size = min(1000, len(ref_images))
        kid = KernelInceptionDistance(normalize=True, subset_size=subset_size).to(device)
        _feed_images(kid, ref_images, real=True, transform=_inception_transform(), device=device)
        torch.save(kid.state_dict(), cache_path)
        logger.info(f"KID reference cache saved: {len(ref_images)} images → {cache_path}")
        return cache_path

    def compute(self, submission_dir: Path, reference_dir: Path,
                cached_ref_features: Path | None = None) -> MetricResult:
        from torchmetrics.image.kid import KernelInceptionDistance
        device = _get_device()
        sub_images = _load_images(submission_dir)
        transform = _inception_transform()

        if cached_ref_features and Path(cached_ref_features).exists():
            # subset_size must be <= min(num_real, num_fake)
            subset_size = min(1000, len(sub_images))
            kid = KernelInceptionDistance(normalize=True, subset_size=subset_size).to(device)
            state = torch.load(cached_ref_features, map_location=device)
            # Only load real_features from cache
            kid.load_state_dict(state)
            logger.info("KID: loaded cached reference features")
        else:
            ref_images = _load_images(reference_dir)
            subset_size = min(1000, len(sub_images), len(ref_images))
            kid = KernelInceptionDistance(normalize=True, subset_size=subset_size).to(device)
            _feed_images(kid, ref_images, real=True, transform=transform, device=device)

        _feed_images(kid, sub_images, real=False, transform=transform, device=device)

        mean, std = kid.compute()
        score = mean.item()
        logger.info(f"KID: {score:.6f} +/- {std.item():.6f}")
        return MetricResult(name=self.name, score=score, is_higher_better=self.is_higher_better,
                            metadata={"std": std.item(), "num_submission": len(sub_images)})

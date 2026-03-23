"""
Local metric backends using PyTorch.
Runs on bourbon's GPU (or CPU fallback).
"""
import logging
from pathlib import Path

from .base import MetricBackend, MetricResult

logger = logging.getLogger(__name__)


def _get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_images_as_tensor(image_dir: Path, max_images: int = 50000):
    """Load images from directory as a list of PIL images."""
    from PIL import Image

    supported = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in supported)
    if not paths:
        raise ValueError(f"No images found in {image_dir}")
    paths = paths[:max_images]
    images = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
        except Exception as e:
            logger.warning(f"Skipping {p.name}: {e}")
    if not images:
        raise ValueError(f"No valid images loaded from {image_dir}")
    return images


class FIDMetric(MetricBackend):
    """Frechet Inception Distance — lower is better."""

    @property
    def name(self) -> str:
        return "fid"

    @property
    def display_name(self) -> str:
        return "FID"

    @property
    def is_higher_better(self) -> bool:
        return False

    def compute(self, submission_dir: Path, reference_dir: Path,
                cached_ref_features: Path | None = None) -> MetricResult:
        import torch
        from torchmetrics.image.fid import FrechetInceptionDistance

        device = _get_device()
        fid = FrechetInceptionDistance(normalize=True).to(device)

        sub_images = _load_images_as_tensor(submission_dir)
        ref_images = _load_images_as_tensor(reference_dir)

        transform = self._get_transform()

        # Process in batches to avoid OOM
        batch_size = 32
        for i in range(0, len(ref_images), batch_size):
            batch = ref_images[i:i + batch_size]
            tensors = torch.stack([transform(img) for img in batch]).to(device)
            fid.update(tensors, real=True)

        for i in range(0, len(sub_images), batch_size):
            batch = sub_images[i:i + batch_size]
            tensors = torch.stack([transform(img) for img in batch]).to(device)
            fid.update(tensors, real=False)

        score = fid.compute().item()
        logger.info(f"FID: {score:.4f} ({len(sub_images)} vs {len(ref_images)} images)")
        return MetricResult(name=self.name, score=score, is_higher_better=self.is_higher_better,
                            metadata={"num_submission": len(sub_images), "num_reference": len(ref_images)})

    @staticmethod
    def _get_transform():
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ])


class ISMetric(MetricBackend):
    """Inception Score — higher is better."""

    @property
    def name(self) -> str:
        return "is"

    @property
    def display_name(self) -> str:
        return "IS"

    @property
    def is_higher_better(self) -> bool:
        return True

    def compute(self, submission_dir: Path, reference_dir: Path,
                cached_ref_features: Path | None = None) -> MetricResult:
        import torch
        from torchmetrics.image.inception import InceptionScore

        device = _get_device()
        inception = InceptionScore(normalize=True).to(device)

        sub_images = _load_images_as_tensor(submission_dir)
        transform = FIDMetric._get_transform()

        batch_size = 32
        for i in range(0, len(sub_images), batch_size):
            batch = sub_images[i:i + batch_size]
            tensors = torch.stack([transform(img) for img in batch]).to(device)
            inception.update(tensors)

        mean, std = inception.compute()
        score = mean.item()
        logger.info(f"IS: {score:.4f} +/- {std.item():.4f} ({len(sub_images)} images)")
        return MetricResult(name=self.name, score=score, is_higher_better=self.is_higher_better,
                            metadata={"std": std.item(), "num_images": len(sub_images)})


class LPIPSMetric(MetricBackend):
    """Learned Perceptual Image Patch Similarity — lower is better.
    Requires paired images (same filenames in submission and reference).
    """

    @property
    def name(self) -> str:
        return "lpips"

    @property
    def display_name(self) -> str:
        return "LPIPS"

    @property
    def is_higher_better(self) -> bool:
        return False

    def compute(self, submission_dir: Path, reference_dir: Path,
                cached_ref_features: Path | None = None) -> MetricResult:
        import torch
        import lpips
        from torchvision import transforms

        device = _get_device()
        loss_fn = lpips.LPIPS(net="alex").to(device)

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1, 1]
        ])

        from PIL import Image
        supported = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        sub_files = sorted(p for p in submission_dir.iterdir() if p.suffix.lower() in supported)
        ref_files = {p.stem: p for p in reference_dir.iterdir() if p.suffix.lower() in supported}

        scores = []
        for sub_path in sub_files:
            ref_path = ref_files.get(sub_path.stem)
            if ref_path is None:
                continue
            try:
                sub_img = transform(Image.open(sub_path).convert("RGB")).unsqueeze(0).to(device)
                ref_img = transform(Image.open(ref_path).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    d = loss_fn(sub_img, ref_img)
                scores.append(d.item())
            except Exception as e:
                logger.warning(f"LPIPS skip {sub_path.name}: {e}")

        if not scores:
            raise ValueError("No matched image pairs found for LPIPS computation")

        avg_score = sum(scores) / len(scores)
        logger.info(f"LPIPS: {avg_score:.4f} ({len(scores)} pairs)")
        return MetricResult(name=self.name, score=avg_score, is_higher_better=self.is_higher_better,
                            metadata={"num_pairs": len(scores)})


class KIDMetric(MetricBackend):
    """Kernel Inception Distance — lower is better."""

    @property
    def name(self) -> str:
        return "kid"

    @property
    def display_name(self) -> str:
        return "KID"

    @property
    def is_higher_better(self) -> bool:
        return False

    def compute(self, submission_dir: Path, reference_dir: Path,
                cached_ref_features: Path | None = None) -> MetricResult:
        import torch
        from torchmetrics.image.kid import KernelInceptionDistance

        device = _get_device()
        kid = KernelInceptionDistance(normalize=True, subset_size=min(50, 1000)).to(device)

        sub_images = _load_images_as_tensor(submission_dir)
        ref_images = _load_images_as_tensor(reference_dir)

        transform = FIDMetric._get_transform()

        batch_size = 32
        for i in range(0, len(ref_images), batch_size):
            batch = ref_images[i:i + batch_size]
            tensors = torch.stack([transform(img) for img in batch]).to(device)
            kid.update(tensors, real=True)

        for i in range(0, len(sub_images), batch_size):
            batch = sub_images[i:i + batch_size]
            tensors = torch.stack([transform(img) for img in batch]).to(device)
            kid.update(tensors, real=False)

        mean, std = kid.compute()
        score = mean.item()
        logger.info(f"KID: {score:.6f} +/- {std.item():.6f}")
        return MetricResult(name=self.name, score=score, is_higher_better=self.is_higher_better,
                            metadata={"std": std.item(), "num_submission": len(sub_images),
                                      "num_reference": len(ref_images)})

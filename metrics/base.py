"""
Abstract base class for metric backends.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MetricResult:
    """Result of a metric computation."""
    name: str
    score: float
    is_higher_better: bool = False
    metadata: dict = field(default_factory=dict)


class MetricBackend(ABC):
    """Abstract base for metric computation backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique metric name (e.g. 'fid', 'is', 'lpips', 'kid')."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name."""

    @property
    @abstractmethod
    def is_higher_better(self) -> bool:
        """Whether higher scores are better."""

    @abstractmethod
    def compute(self, submission_dir: Path, reference_dir: Path,
                cached_ref_features: Path | None = None) -> MetricResult:
        """Compute the metric.

        Args:
            submission_dir: Directory containing submitted images.
            reference_dir: Directory containing reference/ground-truth images.
            cached_ref_features: Optional path to cached reference features.

        Returns:
            MetricResult with the computed score.
        """

    def cache_reference_features(self, reference_dir: Path, cache_path: Path) -> Path:
        """Pre-compute and cache reference features for faster evaluation.

        Default implementation is a no-op. Override in subclasses that
        benefit from caching (e.g. FID with InceptionV3 features).

        Returns:
            Path to the cached features file.
        """
        return cache_path

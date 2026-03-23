"""
Metric registry — discover and manage metric backends.
"""
import logging
from .base import MetricBackend

logger = logging.getLogger(__name__)


class MetricRegistry:
    """Registry for metric backends, keyed by name."""

    def __init__(self):
        self._backends: dict[str, MetricBackend] = {}

    def register(self, backend: MetricBackend):
        self._backends[backend.name] = backend
        logger.info(f"Registered metric: {backend.name} ({backend.display_name})")

    def get(self, name: str) -> MetricBackend | None:
        return self._backends.get(name)

    def list_metrics(self) -> list[dict]:
        return [
            {"name": b.name, "display_name": b.display_name, "is_higher_better": b.is_higher_better}
            for b in self._backends.values()
        ]

    @property
    def names(self) -> list[str]:
        return list(self._backends.keys())


def create_default_registry() -> MetricRegistry:
    """Create a registry with all available local metrics."""
    registry = MetricRegistry()
    try:
        from .local import FIDMetric, ISMetric, LPIPSMetric, KIDMetric
        registry.register(FIDMetric())
        registry.register(ISMetric())
        registry.register(LPIPSMetric())
        registry.register(KIDMetric())
    except ImportError as e:
        logger.warning(f"Some metrics unavailable (missing dependencies): {e}")
    return registry

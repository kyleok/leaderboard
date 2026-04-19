"""
Background worker for processing submissions.
Asyncio-based in-process task queue.
"""
import asyncio
import logging
import shutil
import zipfile
from pathlib import Path

from database import (
    get_submission, get_competition, get_reference_dataset,
    update_submission_status, save_score
)
from metrics.registry import MetricRegistry

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("/app/data/uploads")
EXTRACT_DIR = Path("/app/data/extracted")
REFERENCE_DIR = Path("/app/data/references")
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class SubmissionWorker:
    """Processes submissions from an asyncio queue."""

    def __init__(self, registry: MetricRegistry):
        self.registry = registry
        self.queue: asyncio.Queue[int] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def start(self):
        self._task = asyncio.create_task(self._run())
        logger.info("Submission worker started")

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Submission worker stopped")

    async def enqueue(self, submission_id: int):
        await self.queue.put(submission_id)
        logger.info(f"Enqueued submission {submission_id} (queue size: {self.queue.qsize()})")

    async def _run(self):
        while True:
            submission_id = await self.queue.get()
            try:
                await self._process(submission_id)
            except Exception as e:
                logger.error(f"Submission {submission_id} failed: {e}", exc_info=True)
                update_submission_status(submission_id, "failed", error_message=str(e)[:500])
            finally:
                self.queue.task_done()

    async def _process(self, submission_id: int):
        submission = get_submission(submission_id)
        if not submission:
            logger.warning(f"Submission {submission_id} not found")
            return

        competition = get_competition(submission["competition_id"])
        if not competition:
            update_submission_status(submission_id, "failed", error_message="Competition not found")
            return

        ref_dataset = get_reference_dataset(competition["id"])
        if not ref_dataset:
            update_submission_status(submission_id, "failed",
                                    error_message="No reference dataset uploaded")
            return

        update_submission_status(submission_id, "processing")

        # Extract zip
        extract_path = EXTRACT_DIR / str(submission["competition_id"]) / str(submission_id)
        if extract_path.exists():
            shutil.rmtree(extract_path)
        extract_path.mkdir(parents=True)

        zip_path = Path(submission["file_path"])
        if not zip_path.exists():
            update_submission_status(submission_id, "failed", error_message="Upload file not found")
            return

        await asyncio.to_thread(self._extract_zip, zip_path, extract_path)

        # Find images in extracted dir (handle nested dirs)
        images = self._find_images(extract_path)
        if not images:
            update_submission_status(submission_id, "failed",
                                    error_message="No valid images found in zip")
            self._cleanup(extract_path)
            return

        # Create flat image dir for metrics
        flat_dir = extract_path / "_flat"
        flat_dir.mkdir(exist_ok=True)
        for i, img_path in enumerate(images):
            dest = flat_dir / f"{i:06d}{img_path.suffix.lower()}"
            shutil.copy2(img_path, dest)

        num_images = len(images)
        update_submission_status(submission_id, "processing",
                                progress_current=0, progress_total=len(competition["metrics"]))

        # Resolve reference dir
        ref_path = Path(ref_dataset["file_path"])
        cached_features = Path(ref_dataset["cached_features"]) if ref_dataset.get("cached_features") else None

        # Compute each metric
        metrics_to_run = competition["metrics"]
        for i, metric_name in enumerate(metrics_to_run):
            backend = self.registry.get(metric_name)
            if not backend:
                logger.warning(f"Metric '{metric_name}' not found in registry, skipping")
                continue

            try:
                result = await asyncio.to_thread(
                    backend.compute, flat_dir, ref_path, cached_features
                )
                save_score(submission_id, result.name, result.score, result.is_higher_better)
                logger.info(f"Submission {submission_id}: {result.name} = {result.score:.6f}")
            except Exception as e:
                logger.error(f"Metric {metric_name} failed for submission {submission_id}: {e}")
                save_score(submission_id, metric_name, -1.0, False)

            update_submission_status(submission_id, "processing",
                                    progress_current=i + 1, progress_total=len(metrics_to_run))

        # Done
        update_submission_status(submission_id, "done",
                                progress_current=len(metrics_to_run),
                                progress_total=len(metrics_to_run))
        logger.info(f"Submission {submission_id} processed ({num_images} images, "
                     f"{len(metrics_to_run)} metrics)")

        # Cleanup extracted files, keep only latest zip per team
        self._cleanup(extract_path)
        self._cleanup_old_zips(submission["team_id"], submission["competition_id"], submission["file_path"])

    @staticmethod
    def _extract_zip(zip_path: Path, extract_path: Path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Security: prevent zip slip
            for info in zf.infolist():
                if info.filename.startswith("/") or ".." in info.filename:
                    raise ValueError(f"Unsafe path in zip: {info.filename}")
            zf.extractall(extract_path)

    @staticmethod
    def _find_images(root: Path) -> list[Path]:
        """Recursively find all image files, ignoring __MACOSX and hidden files."""
        images = []
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            if any(part.startswith(".") or part == "__MACOSX" for part in p.parts):
                continue
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                images.append(p)
        return images

    @staticmethod
    def _cleanup(path: Path):
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Cleanup failed for {path}: {e}")

    @staticmethod
    def _cleanup_old_zips(team_id: int, competition_id: int, keep_path: str):
        team_dir = UPLOAD_DIR / str(competition_id) / str(team_id)
        if not team_dir.exists():
            return
        for zip_file in sorted(team_dir.glob("*.zip")):
            if str(zip_file) != keep_path:
                try:
                    zip_file.unlink()
                    logger.info(f"Deleted old ZIP: {zip_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete old ZIP {zip_file}: {e}")

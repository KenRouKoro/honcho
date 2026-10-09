import asyncio
import contextlib
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from logging import getLogger
from typing import Any, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import sentry_sdk
from sqlalchemy import exists, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.dependencies import tracked_db
from src.schemas import DreamType
from src.utils.work_unit import construct_work_unit_key, parse_work_unit_key

logger = getLogger(__name__)


_dream_scheduler: "DreamScheduler | None" = None


def set_dream_scheduler(dream_scheduler: "DreamScheduler") -> None:
    """Set the global dream scheduler reference."""
    global _dream_scheduler
    _dream_scheduler = dream_scheduler


def get_dream_scheduler() -> "DreamScheduler | None":
    """Get the global dream scheduler reference."""
    return _dream_scheduler


@dataclass(frozen=True)
class DreamWindow:
    """The daily window a collection's dreams are allowed to fire in.

    The window started as a process-wide setting, but "off-peak" is a property of the
    *place*, not of the deployment: a night-shift group or an overseas group is quiet at
    a different hour, and clamping everyone into `Asia/Shanghai 02:00–07:00` merely
    moves their load into their busiest hour. So the window is now resolved per session
    (from `Session.metadata`, §28.4) with the global setting as the default.
    """

    start_hour: int
    end_hour: int
    timezone: str
    source: str = "global"

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


def global_dream_window() -> DreamWindow | None:
    """The process-wide window, or None when no window is configured."""
    start = settings.DREAM.ALLOWED_WINDOW_START_HOUR
    end = settings.DREAM.ALLOWED_WINDOW_END_HOUR
    if start is None or end is None:
        return None
    return DreamWindow(
        start_hour=start,
        end_hour=end,
        timezone=settings.DREAM.WINDOW_TIMEZONE,
        source="global",
    )


def parse_session_dream_window(raw: Any) -> DreamWindow | None:
    """``{"start_hour": 4, "end_hour": 9, "timezone": "Asia/Tokyo"}`` → DreamWindow.

    Every key is optional *except* that both hours must be present together — half a
    window is not a window, and silently borrowing the other half from the global
    setting would produce a range nobody asked for. Anything malformed is ignored with
    a debug log; a bad override must never stop dreams, only fail to improve them.
    """
    if not isinstance(raw, dict):
        return None
    data = cast(dict[str, Any], raw)

    def _hour(key: str) -> int | None:
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value if 0 <= value <= 23 else None

    start = _hour("start_hour")
    end = _hour("end_hour")
    if start is None or end is None:
        logger.debug("Ignoring incomplete session dream window: %r", raw)
        return None

    tz_raw = data.get("timezone")
    tz_name = (
        tz_raw.strip()
        if isinstance(tz_raw, str) and tz_raw.strip()
        else settings.DREAM.WINDOW_TIMEZONE
    )
    try:
        ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        logger.warning(
            "Unknown timezone %r in session dream window; using %s",
            tz_name,
            settings.DREAM.WINDOW_TIMEZONE,
        )
        tz_name = settings.DREAM.WINDOW_TIMEZONE
    return DreamWindow(
        start_hour=start, end_hour=end, timezone=tz_name, source="session"
    )


def effective_dream_window(
    session_metadata: dict[str, Any] | None,
) -> DreamWindow | None:
    """Session override if it has one, else the global window."""
    override = parse_session_dream_window(
        (session_metadata or {}).get(settings.DREAM.SESSION_WINDOW_METADATA_KEY)
    )
    return override or global_dream_window()


async def resolve_dream_window(
    db: AsyncSession, collection: models.Collection
) -> DreamWindow | None:
    """The window that applies to this collection, right now.

    A collection is keyed by ``(observer, observed)`` and is **not** tied to one session:
    the same pair can collect observations from several groups. The window still has to
    come from somewhere, so this uses the same heuristic ``execute_dream`` already uses
    to pick the session a dream belongs to — **the session of the newest explicit
    observation**. That keeps the schedule-time and execute-time notions of "which place
    is this dream about" identical, which matters because a mismatch would silently
    apply one group's off-peak hours to another's dream.

    Failures fall back to the global window rather than propagating: a dream that fires
    at a slightly wrong hour is a cost problem, a dream that never fires is a memory
    problem.
    """
    try:
        session_name = await db.scalar(
            select(models.Document.session_name)
            .where(
                models.Document.workspace_name == collection.workspace_name,
                models.Document.observer == collection.observer,
                models.Document.observed == collection.observed,
                models.Document.level == "explicit",
            )
            .order_by(models.Document.created_at.desc(), models.Document.id.desc())
            .limit(1)
        )
        if not session_name:
            return global_dream_window()
        metadata = await db.scalar(
            select(models.Session.h_metadata).where(
                models.Session.workspace_name == collection.workspace_name,
                models.Session.name == session_name,
            )
        )
    except Exception:  # pragma: no cover - defensive; scheduling must not break
        logger.warning(
            "Failed to resolve dream window for %s/%s/%s; using global window",
            collection.workspace_name,
            collection.observer,
            collection.observed,
            exc_info=True,
        )
        return global_dream_window()

    return effective_dream_window(cast(dict[str, Any] | None, metadata or {}) or None)


def clamp_wake_to_window(
    wake: datetime, start_hour: int, end_hour: int, tz: ZoneInfo
) -> datetime:
    """Clamp a wake time into the daily allowed window [start_hour, end_hour).

    The window is interpreted in `tz` local time and may wrap midnight
    (e.g. start=22, end=6). If `wake` already falls inside the window it is
    returned unchanged; otherwise the next window start is returned (UTC).
    """
    local = wake.astimezone(tz)
    today_start = local.replace(hour=start_hour, minute=0, second=0, microsecond=0)
    if start_hour < end_hour:
        in_window = start_hour <= local.hour < end_hour
    else:
        in_window = local.hour >= start_hour or local.hour < end_hour
    if in_window:
        return wake
    if local >= today_start:
        today_start += timedelta(days=1)
    return today_start.astimezone(UTC)


def jitter_within_window(
    wake: datetime,
    *,
    end_hour: int,
    tz: ZoneInfo,
    jitter_minutes: int,
) -> datetime:
    """Spread a deferred wake across the window's remaining span.

    `clamp_wake_to_window` maps every due dream onto the *same* instant — the window
    start — so without this the whole workspace's collections would fire together.
    That is not hypothetical: of 569 dreams in the reference deployment, **133 landed
    in the 02:00 hour** (doc/memory_layer_decision.md §24.2).

    The offset is drawn uniformly from ``[0, jitter_minutes)`` and then trimmed so the
    wake still lands inside the window (at least a minute of headroom is kept).
    ``jitter_minutes=0`` disables it and returns ``wake`` unchanged.
    """
    if jitter_minutes <= 0:
        return wake

    local = wake.astimezone(tz)
    window_end = local.replace(hour=end_hour, minute=0, second=0, microsecond=0)
    if window_end <= local:
        # The window wraps midnight (e.g. 22 -> 6), or we are already past today's end.
        window_end += timedelta(days=1)

    headroom_minutes = (window_end - local).total_seconds() / 60.0
    span = min(float(jitter_minutes), max(0.0, headroom_minutes - 1.0))
    if span <= 0:
        return wake
    return wake + timedelta(minutes=random.uniform(0.0, span))


class DreamScheduler:
    _instance: "DreamScheduler | None" = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if not DreamScheduler._initialized:
            self.pending_dreams: dict[str, asyncio.Task[None]] = {}
            DreamScheduler._initialized = True

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the singleton instance. Only use this in tests."""
        cls._instance = None
        cls._initialized = False

    async def schedule_dream(
        self,
        work_unit_key: str,
        workspace_name: str,
        delay_minutes: float,
        dream_type: DreamType,
        *,
        observer: str,
        observed: str,
        trigger_reason: str | None = None,
        delay_reason: str | None = None,
        documents_since_last_dream_at_schedule: int | None = None,
        document_threshold: int | None = None,
        window: DreamWindow | None = None,
    ) -> None:
        """Schedule a dream for a collection after a delay.

        telemetry kwargs are captured at schedule time and threaded
        through the queue payload so DreamRunEvent can attribute the dream
        back to its scheduling context.

        ``window`` is the **already-resolved** window for this collection's session
        (see ``resolve_dream_window``). Omitting it falls back to the global window, so
        a direct caller keeps the previous behaviour; there is no way to ask for "no
        window at all" other than leaving both global hours unset.
        """
        if not settings.DREAM.ENABLED:
            return

        # Cancel any existing dream for this collection
        await self.cancel_dream(work_unit_key)

        window = window if window is not None else global_dream_window()
        if window is not None:
            tz = window.tz
            wake = datetime.now(UTC) + timedelta(minutes=delay_minutes)
            clamped = clamp_wake_to_window(wake, window.start_hour, window.end_hour, tz)
            if clamped > wake:
                # Without the jitter every deferred dream in the workspace wakes at the
                # same instant (the window start). See `jitter_within_window`.
                jittered = jitter_within_window(
                    clamped,
                    end_hour=window.end_hour,
                    tz=tz,
                    jitter_minutes=settings.DREAM.WINDOW_JITTER_MINUTES,
                )
                logger.info(
                    "Deferring dream for %s into allowed window (%s): %s -> %s",
                    work_unit_key,
                    window.source,
                    wake.isoformat(),
                    jittered.isoformat(),
                )
                delay_minutes = (jittered - datetime.now(UTC)).total_seconds() / 60

        task = asyncio.create_task(
            self._delayed_dream(
                work_unit_key,
                workspace_name,
                delay_minutes,
                dream_type,
                observer=observer,
                observed=observed,
                trigger_reason=trigger_reason,
                delay_reason=delay_reason,
                documents_since_last_dream_at_schedule=documents_since_last_dream_at_schedule,
                document_threshold=document_threshold,
            )
        )
        self.pending_dreams[work_unit_key] = task
        task.add_done_callback(lambda t: self.pending_dreams.pop(work_unit_key, None))

    async def cancel_dream(self, work_unit_key: str) -> bool:
        """Cancel a pending dream. Returns True if a dream was cancelled."""
        if work_unit_key in self.pending_dreams:
            task = self.pending_dreams.pop(work_unit_key)
            task.cancel()
            # Wait for the task to actually finish (including its done callback)
            with contextlib.suppress(asyncio.CancelledError):
                await task
            return True
        return False

    async def cancel_dreams_for_observed(
        self, workspace_name: str, observed: str
    ) -> set[str]:
        """
        Cancel all pending dreams where the observed peer matches.

        This handles both self-observation (observer=observed) and peer-to-peer
        observation (observer!=observed) dreams.

        Args:
            workspace_name: The workspace to match
            observed: The observed peer name to match
        Returns:
            Set of work_unit_keys that were cancelled
        """
        cancelled: set[str] = set()

        # Collect keys to cancel (can't modify dict while iterating)
        keys_to_cancel: list[str] = []
        for work_unit_key in self.pending_dreams:
            parsed = parse_work_unit_key(work_unit_key)
            if parsed.workspace_name == workspace_name and parsed.observed == observed:
                keys_to_cancel.append(work_unit_key)

        # Cancel each matching dream
        for key in keys_to_cancel:
            if await self.cancel_dream(key):
                cancelled.add(key)

        return cancelled

    async def _delayed_dream(
        self,
        work_unit_key: str,
        workspace_name: str,
        delay_minutes: float,
        dream_type: DreamType,
        *,
        observer: str,
        observed: str,
        trigger_reason: str | None = None,
        delay_reason: str | None = None,
        documents_since_last_dream_at_schedule: int | None = None,
        document_threshold: int | None = None,
    ) -> None:
        try:
            await asyncio.sleep(delay_minutes * 60)

            await self.execute_dream(
                workspace_name,
                dream_type,
                observer=observer,
                observed=observed,
                trigger_reason=trigger_reason,
                delay_reason=delay_reason,
                documents_since_last_dream_at_schedule=documents_since_last_dream_at_schedule,
                document_threshold=document_threshold,
            )
            logger.info("Executed dream for %s", work_unit_key)

        except asyncio.CancelledError:
            logger.debug("Dream task cancelled for %s", work_unit_key)
        except Exception as e:
            logger.error("Error in delayed dream for %s: %s", work_unit_key, e)
            if settings.SENTRY.ENABLED:
                sentry_sdk.capture_exception(e)

    async def execute_dream(
        self,
        workspace_name: str,
        dream_type: DreamType,
        *,
        observer: str,
        observed: str,
        trigger_reason: str | None = None,
        delay_reason: str | None = None,
        documents_since_last_dream_at_schedule: int | None = None,
        document_threshold: int | None = None,
    ) -> None:
        """Execute the dream by enqueueing it."""
        from src import crud
        from src.deriver.enqueue import enqueue_dream
        from src.utils.config_helpers import get_configuration

        async with tracked_db("dream_session_lookup") as db:
            stmt = (
                select(models.Document.session_name)
                .where(
                    models.Document.workspace_name == workspace_name,
                    models.Document.observer == observer,
                    models.Document.observed == observed,
                    models.Document.level == "explicit",
                )
                .order_by(models.Document.created_at.desc())
                .limit(1)
            )
            session_name = await db.scalar(stmt)

            if not session_name:
                logger.warning(
                    f"No documents found for {workspace_name}/{observer}/{observed}, skipping dream"
                )
                return

            session = await crud.get_session(
                db, workspace_name=workspace_name, session_name=session_name
            )
            workspace = await crud.get_workspace(db, workspace_name=workspace_name)

            configuration = get_configuration(None, session, workspace)

            if not configuration.dream.enabled:
                logger.debug(
                    f"Dreams disabled for {workspace_name}/{session_name}, skipping dream"
                )
                return

        await enqueue_dream(
            workspace_name,
            observer=observer,
            observed=observed,
            dream_type=dream_type,
            session_name=session_name,
            trigger_reason=trigger_reason,
            delay_reason=delay_reason,
            documents_since_last_dream_at_schedule=documents_since_last_dream_at_schedule,
            document_threshold=document_threshold,
        )

    async def shutdown(self) -> None:
        """Cancel all pending dreams during shutdown."""
        if self.pending_dreams:
            logger.info(f"Cancelling {len(self.pending_dreams)} pending dreams...")
            for task in self.pending_dreams.values():
                task.cancel()
            await asyncio.gather(*self.pending_dreams.values(), return_exceptions=True)
            self.pending_dreams.clear()


async def check_and_schedule_dream(
    db: AsyncSession,
    collection: models.Collection,
) -> bool:
    """
    From the moment a dream is scheduled until it completes or fails, no second
    dream may be enqueued for the same (workspace, observer, observed) — and the
    baseline count advances only when consolidation actually happened.

    Check if a collection has reached the explicit-observation threshold and schedule a timer-based dream.

    This function only schedules a timer-based dream if:
    1. Dreams are enabled
    2. Explicit-observation threshold is reached (dreamer output does not count)
    3. Minimum hours between dreams have passed
    4. No dream is already pending in the queue for this collection (in-flight check)
    5. No dream is already scheduled for this collection

    Args:
        db: Database session
        collection: Collection model to check

    Returns:
        True if a dream timer was scheduled, False otherwise
    """
    if not settings.DREAM.ENABLED:
        return False

    dream_metadata = collection.internal_metadata.get("dream", {})
    last_dream_document_count = dream_metadata.get("last_dream_document_count", 0)
    last_dream_at = dream_metadata.get("last_dream_at")

    # Count explicit-level docs only: dreamer output (deductive/inductive/
    # contradiction) would inflate the threshold and create a feedback loop.
    count_stmt = select(func.count(models.Document.id)).where(
        models.Document.workspace_name == collection.workspace_name,
        models.Document.observer == collection.observer,
        models.Document.observed == collection.observed,
        models.Document.level == "explicit",
    )
    current_explicit_count = int(await db.scalar(count_stmt) or 0)

    documents_since_last_dream = current_explicit_count - last_dream_document_count

    logger.debug(
        "Dream check",
        extra={
            "workspace_name": collection.workspace_name,
            "observer": collection.observer,
            "observed": collection.observed,
            "current_explicit_count": current_explicit_count,
            "last_dream_document_count": last_dream_document_count,
            "documents_since_last_dream": documents_since_last_dream,
            "document_threshold": settings.DREAM.DOCUMENT_THRESHOLD,
        },
    )

    if documents_since_last_dream >= settings.DREAM.DOCUMENT_THRESHOLD:
        # capture *why* this schedule fired (threshold) and
        # *how* it will fire (idle vs immediate). The two gates were
        # intentionally split — collapsing them into a single trigger_reason
        # would lose the scheduling semantics.
        trigger_reason = "document_threshold"
        delay_reason = (
            "idle_timeout" if settings.DREAM.IDLE_TIMEOUT_MINUTES > 0 else "immediate"
        )
        if last_dream_at:
            try:
                last_dream_time = datetime.fromisoformat(last_dream_at)
                hours_since_last_dream = (
                    datetime.now(UTC) - last_dream_time
                ).total_seconds() / 3600

                if hours_since_last_dream < settings.DREAM.MIN_HOURS_BETWEEN_DREAMS:
                    logger.debug(
                        f"Skipping dream for {collection.observer}/{collection.observed}: only {hours_since_last_dream:.1f} hours "
                        + f"since last dream (minimum: {settings.DREAM.MIN_HOURS_BETWEEN_DREAMS})"
                    )
                    # delay_reason = "min_hours_gate" if we DID schedule, but
                    # we don't — return early. Telemetry only records dreams
                    # that actually fire.
                    return False
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Invalid last_dream_at timestamp: {last_dream_at}, error: {e}"
                )

        # Queue is source of truth for in-flight dreams; mirrors
        # uq_queue_dream_pending_work_unit_key.
        enabled_dream_types = settings.DREAM.ENABLED_TYPES
        pending_keys = [
            construct_work_unit_key(
                collection.workspace_name,
                {
                    "task_type": "dream",
                    "observer": collection.observer,
                    "observed": collection.observed,
                    "dream_type": dream_type,
                },
            )
            for dream_type in enabled_dream_types
        ]
        pending_exists = await db.scalar(
            select(
                exists(
                    select(models.QueueItem.id).where(
                        models.QueueItem.task_type == "dream",
                        models.QueueItem.processed == False,  # noqa: E712
                        models.QueueItem.work_unit_key.in_(pending_keys),
                    )
                )
            )
        )
        if pending_exists:
            logger.debug(
                "Skipping dream schedule for %s/%s: pending dream already in queue",
                collection.observer,
                collection.observed,
            )
            return False

        dream_scheduler = get_dream_scheduler()
        if dream_scheduler:
            # §28.4: resolve the window *here*, once, so every dream type of this
            # collection is clamped by the same rule. Doing it lazily (only for the
            # rare schedule that actually fires) keeps the two extra reads off the
            # common path where the threshold has not been reached yet.
            window = await resolve_dream_window(db, collection)
            for dream_type in enabled_dream_types:
                dream_work_unit_key = construct_work_unit_key(
                    collection.workspace_name,
                    {
                        "task_type": "dream",
                        "observer": collection.observer,
                        "observed": collection.observed,
                        "dream_type": dream_type,
                    },
                )
                await dream_scheduler.schedule_dream(
                    dream_work_unit_key,
                    collection.workspace_name,
                    settings.DREAM.IDLE_TIMEOUT_MINUTES,
                    dream_type=DreamType(dream_type),
                    observer=collection.observer,
                    observed=collection.observed,
                    trigger_reason=trigger_reason,
                    delay_reason=delay_reason,
                    documents_since_last_dream_at_schedule=documents_since_last_dream,
                    document_threshold=settings.DREAM.DOCUMENT_THRESHOLD,
                    window=window,
                )
                logger.debug(
                    "Scheduled dream",
                    extra={
                        "workspace_name": collection.workspace_name,
                        "observer": collection.observer,
                        "observed": collection.observed,
                        "documents_since_last_dream": documents_since_last_dream,
                        "document_threshold": settings.DREAM.DOCUMENT_THRESHOLD,
                        "dream_type": dream_type,
                        "window_source": window.source if window else None,
                    },
                )
            return True

    return False

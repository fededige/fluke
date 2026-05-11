"""Custom logger with timestamp tracking."""

import time
from datetime import datetime
from typing import Any, Literal, Union

from torch.nn import Module
from rich import print as rich_print
from rich.panel import Panel
from rich.pretty import Pretty

from fluke.evaluation import PerformanceTracker
from fluke.utils.log import Log


class TimestampPerformanceTracker(PerformanceTracker):
    """Extends PerformanceTracker with a 'time' field that records
    wall-clock timestamps and elapsed seconds for each round.

    Extra keys stored per round in ``self._performance["time"]``:
        - ``"start"``: Unix timestamp when the round started.
        - ``"end"``:   Unix timestamp when the round ended.
        - ``"elapsed"``: Elapsed seconds for that round.
    """

    def __init__(self):
        super().__init__()
        # round -> {"start": float, "end": float, "elapsed": float}
        self._performance["time"] = {}

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def record_start(self, round: int) -> None:
        """Record the wall-clock start time for *round*.

        Args:
            round (int): The federation round number.
        """
        self._performance["time"][round] = {
            "start": time.time(),
            "end": None,
            "elapsed": None,
        }

    def record_end(self, rnd: int) -> None:
        """Record the wall-clock end time for *round* and compute elapsed seconds.

        Args:
            round (int): The federation round number.
        """
        entry = self._performance["time"].get(rnd)
        if entry is None:
            # ``record_start`` was never called – store end only
            self._performance["time"][rnd] = {
                "start": None,
                "end": time.time(),
                "elapsed": None,
            }
            return

        end = time.time()
        entry["end"] = end
        entry["elapsed"] = round(end - entry["start"], 4) if entry["start"] else None

    def get_time(self, round: int) -> dict:
        """Return the timing info for *round*, or an empty dict if not found.

        Args:
            round (int): The federation round number.

        Returns:
            dict: ``{"start": float, "end": float, "elapsed": float}``
        """
        return self._performance["time"].get(round, {})

    def total_elapsed(self) -> float:
        """Sum of all per-round elapsed times (in seconds).

        Returns:
            float: Total elapsed seconds across all rounds, ignoring rounds
            where elapsed is ``None``.
        """
        return sum(
            v["elapsed"]
            for v in self._performance["time"].values()
            if v.get("elapsed") is not None
        )


# ---------------------------------------------------------------------------


class MyLogger(Log):
    """Logger that enriches every round panel with wall-clock timestamps.

    Each round panel shows:
        - ``timestamp_start`` – human-readable start time  (ISO-8601)
        - ``timestamp_end``   – human-readable end time    (ISO-8601)
        - ``elapsed_sec``     – seconds the round took

    The *finished* summary additionally shows the total elapsed time across
    all rounds.

    Args:
        **kwargs: Forwarded to :class:`Log`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Replace the vanilla tracker with the timestamp-aware one
        self.tracker: TimestampPerformanceTracker = TimestampPerformanceTracker()

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def start_round(self, round: int, global_model: Module) -> None:
        self.tracker.record_start(round)
        super().start_round(round, global_model)

    def end_round(self, round: int) -> None:
        self.tracker.record_end(round)

        timing = self.tracker.get_time(round)
        start_dt = (
            datetime.fromtimestamp(timing["start"]).isoformat(timespec="seconds")
            if timing.get("start")
            else "N/A"
        )
        end_dt = (
            datetime.fromtimestamp(timing["end"]).isoformat(timespec="seconds")
            if timing.get("end")
            else "N/A"
        )
        elapsed = timing.get("elapsed")

        # Let the parent build and print its usual stats panel …
        super().end_round(round)

        # … then append a compact timestamp panel right after
        rich_print(
            Panel(
                Pretty(
                    {
                        "timestamp_start": start_dt,
                        "timestamp_end": end_dt,
                        "elapsed_sec": elapsed,
                    },
                    expand_all=True,
                ),
                title=f"Round {round} – Timing",
                width=100,
            )
        )

    def finished(self, round: int) -> None:
        super().finished(round)

        total = self.tracker.total_elapsed()
        rich_print(
            Panel(
                Pretty({"total_elapsed_sec": total}, expand_all=True),
                title="Overall Timing",
                width=100,
            )
        )

    def save(self, path: str) -> None:
        """Extend the JSON file produced by :class:`Log` with timing data.

        Args:
            path (str): Path to the output JSON file.
        """
        import json

        # Re-open what the parent already wrote …
        super().save(path)
        with open(path) as f:
            data = json.load(f)

        # … and inject the timing info
        data["timing"] = {
            str(r): v for r, v in self.tracker["time"].items()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
"""Execution trace — structured log of everything that happened during a workflow run."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

# Cost per 1M tokens (prompt, completion)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gemini/gemini-3.1-pro-preview": {"prompt": 1.25, "completion": 5.00},
    "gemini/gemini-3-flash-preview": {"prompt": 0.50, "completion": 2.00},
    "gemini/gemini-2.0-flash": {"prompt": 0.10, "completion": 0.40},
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4.1": {"prompt": 2.00, "completion": 8.00},
    "gpt-4.1-mini": {"prompt": 0.40, "completion": 1.60},
    "claude-sonnet-4-5-20250514": {"prompt": 3.00, "completion": 15.00},
    "claude-opus-4-6": {"prompt": 15.00, "completion": 75.00},
}

# Fallback pricing if model not in table
DEFAULT_PRICING = {"prompt": 2.50, "completion": 10.00}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
    return (
        prompt_tokens * pricing["prompt"] / 1_000_000
        + completion_tokens * pricing["completion"] / 1_000_000
    )


@dataclass
class TraceEvent:
    """A single event in the execution trace."""

    timestamp: float  # monotonic seconds since run start
    event_type: str  # cell_start, cell_end, tool_call, llm_call, error, heal
    cell_index: int | None = None
    cell_label: str = ""
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "t": round(self.timestamp, 3),
            "type": self.event_type,
        }
        if self.cell_index is not None:
            d["cell"] = self.cell_index
        if self.cell_label:
            d["label"] = self.cell_label
        if self.data:
            d["data"] = self.data
        return d


@dataclass
class ExecutionTrace:
    """Complete trace of a workflow run."""

    run_id: str
    events: list[TraceEvent] = field(default_factory=list)
    start_time: float = 0.0
    on_event: Callable[[TraceEvent], None] | None = None

    def start(self) -> None:
        self.start_time = time.monotonic()

    def _elapsed(self) -> float:
        return time.monotonic() - self.start_time

    def _emit(self, event: TraceEvent) -> None:
        """Append event and fire callback if set."""
        self.events.append(event)
        if self.on_event:
            self.on_event(event)

    def cell_start(self, index: int, label: str) -> None:
        self._emit(
            TraceEvent(
                timestamp=self._elapsed(),
                event_type="cell_start",
                cell_index=index,
                cell_label=label,
            )
        )

    def cell_end(self, index: int, label: str, status: str, duration_ms: float) -> None:
        self._emit(
            TraceEvent(
                timestamp=self._elapsed(),
                event_type="cell_end",
                cell_index=index,
                cell_label=label,
                data={"status": status, "duration_ms": round(duration_ms, 1)},
            )
        )

    def tool_call(
        self,
        action: str,
        duration_ms: float,
        ok: bool,
        cell_index: int | None = None,
        error: str = "",
        input_keys: list[str] | None = None,
    ) -> None:
        data: dict[str, Any] = {
            "action": action,
            "duration_ms": round(duration_ms, 1),
            "ok": ok,
        }
        if error:
            data["error"] = error
        if input_keys:
            data["input_keys"] = input_keys
        self._emit(
            TraceEvent(
                timestamp=self._elapsed(),
                event_type="tool_call",
                cell_index=cell_index,
                data=data,
            )
        )

    def llm_call(
        self,
        model: str,
        duration_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        cell_index: int | None = None,
        has_format: bool = False,
    ) -> None:
        cost = _estimate_cost(model, prompt_tokens, completion_tokens)
        self._emit(
            TraceEvent(
                timestamp=self._elapsed(),
                event_type="llm_call",
                cell_index=cell_index,
                data={
                    "model": model,
                    "duration_ms": round(duration_ms, 1),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost": round(cost, 6),
                    "has_format": has_format,
                },
            )
        )

    def heal(
        self,
        cell_index: int,
        original_error: str,
        healed: bool,
        attempts: int,
    ) -> None:
        self._emit(
            TraceEvent(
                timestamp=self._elapsed(),
                event_type="heal",
                cell_index=cell_index,
                data={
                    "original_error": original_error,
                    "healed": healed,
                    "attempts": attempts,
                },
            )
        )

    def error(self, cell_index: int, error_type: str, message: str) -> None:
        self._emit(
            TraceEvent(
                timestamp=self._elapsed(),
                event_type="error",
                cell_index=cell_index,
                data={"error_type": error_type, "message": message[:500]},
            )
        )

    def summary(self) -> dict:
        """Generate a summary of the run with cost tracking."""
        total_duration = self._elapsed() * 1000

        cell_starts = [e for e in self.events if e.event_type == "cell_start"]
        tool_calls = [e for e in self.events if e.event_type == "tool_call"]
        llm_calls = [e for e in self.events if e.event_type == "llm_call"]
        errors = [e for e in self.events if e.event_type == "error"]
        heals = [e for e in self.events if e.event_type == "heal"]

        total_prompt = sum(e.data.get("prompt_tokens", 0) for e in llm_calls)
        total_completion = sum(e.data.get("completion_tokens", 0) for e in llm_calls)
        total_cost = sum(e.data.get("cost", 0) for e in llm_calls)

        return {
            "run_id": self.run_id,
            "total_duration_ms": round(total_duration, 1),
            "cells": len(cell_starts),
            "tool_calls": len(tool_calls),
            "llm_calls": len(llm_calls),
            "errors": len(errors),
            "heals": len(heals),
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_cost": round(total_cost, 6),
        }

    def to_dict(self) -> dict:
        """Serialize the full trace for storage/API response."""
        return {
            "run_id": self.run_id,
            "events": [e.to_dict() for e in self.events],
            "summary": self.summary(),
        }

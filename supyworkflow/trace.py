"""Execution trace — structured log of everything that happened during a workflow run."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TraceEvent:
    """A single event in the execution trace."""

    timestamp: float  # monotonic seconds since run start
    event_type: str  # cell_start, cell_end, tool_call, llm_call, error
    cell_index: int | None = None
    cell_label: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class ExecutionTrace:
    """Complete trace of a workflow run. Built from structured logs."""

    run_id: str
    events: list[TraceEvent] = field(default_factory=list)
    start_time: float = 0.0

    def start(self) -> None:
        self.start_time = time.monotonic()

    def _elapsed(self) -> float:
        return time.monotonic() - self.start_time

    def cell_start(self, index: int, label: str) -> None:
        self.events.append(TraceEvent(
            timestamp=self._elapsed(),
            event_type="cell_start",
            cell_index=index,
            cell_label=label,
        ))

    def cell_end(self, index: int, label: str, status: str, duration_ms: float) -> None:
        self.events.append(TraceEvent(
            timestamp=self._elapsed(),
            event_type="cell_end",
            cell_index=index,
            cell_label=label,
            data={"status": status, "duration_ms": duration_ms},
        ))

    def tool_call(
        self,
        provider: str,
        action: str,
        duration_ms: float,
        ok: bool,
        error: str = "",
    ) -> None:
        self.events.append(TraceEvent(
            timestamp=self._elapsed(),
            event_type="tool_call",
            data={
                "provider": provider,
                "action": action,
                "duration_ms": duration_ms,
                "ok": ok,
                "error": error,
            },
        ))

    def llm_call(
        self,
        model: str,
        duration_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        self.events.append(TraceEvent(
            timestamp=self._elapsed(),
            event_type="llm_call",
            data={
                "model": model,
                "duration_ms": duration_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        ))

    def error(self, cell_index: int, error_type: str, message: str) -> None:
        self.events.append(TraceEvent(
            timestamp=self._elapsed(),
            event_type="error",
            cell_index=cell_index,
            data={"error_type": error_type, "message": message},
        ))

    def summary(self) -> dict:
        """Generate a summary of the run."""
        total_duration = self._elapsed() * 1000
        cell_count = len([e for e in self.events if e.event_type == "cell_start"])
        tool_calls = [e for e in self.events if e.event_type == "tool_call"]
        llm_calls = [e for e in self.events if e.event_type == "llm_call"]
        errors = [e for e in self.events if e.event_type == "error"]

        total_prompt_tokens = sum(e.data.get("prompt_tokens", 0) for e in llm_calls)
        total_completion_tokens = sum(e.data.get("completion_tokens", 0) for e in llm_calls)

        return {
            "run_id": self.run_id,
            "total_duration_ms": total_duration,
            "cells": cell_count,
            "tool_calls": len(tool_calls),
            "llm_calls": len(llm_calls),
            "errors": len(errors),
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
        }

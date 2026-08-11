"""Decision journal - the append-only record of why the bot did what it did.

Stores the complete trace for every significant decision::

    DATA -> FEATURES -> REGIME -> MODELS -> ENSEMBLE -> RISK -> EXECUTION -> ORDER

The journal is **append-only**: entries are never updated in place. An outcome
is recorded as a *linked* record rather than a mutation of the original, so the
decision as it was made at the time remains exactly reconstructable. That is
what makes replay trustworthy - a journal you can edit is a journal that tells
you what you wish you had thought.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


class DecisionKind(str, Enum):
    ENTRY = "ENTRY"
    NO_TRADE = "NO_TRADE"
    EXIT = "EXIT"
    STOP_MOVE = "STOP_MOVE"
    PARTIAL_CLOSE = "PARTIAL_CLOSE"
    RISK_ACTION = "RISK_ACTION"
    MODE_CHANGE = "MODE_CHANGE"
    OUTCOME = "OUTCOME"


@dataclass(slots=True)
class DecisionTrace:
    """Each stage of the pipeline, captured as it was at decision time."""

    data: dict[str, Any] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)
    regime: dict[str, Any] = field(default_factory=dict)
    models: list[dict[str, Any]] = field(default_factory=list)
    ensemble: dict[str, Any] = field(default_factory=dict)
    scenarios: dict[str, Any] = field(default_factory=dict)
    expected_value: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)
    order: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "data": self.data,
            "features": {k: round(v, 6) for k, v in list(self.features.items())[:120]},
            "regime": self.regime,
            "models": self.models,
            "ensemble": self.ensemble,
            "scenarios": self.scenarios,
            "expected_value": self.expected_value,
            "risk": self.risk,
            "execution": self.execution,
            "order": self.order,
        }

    def stages(self) -> list[tuple[str, bool]]:
        """Which pipeline stages actually produced something."""

        return [
            ("DATA", bool(self.data)),
            ("FEATURES", bool(self.features)),
            ("REGIME", bool(self.regime)),
            ("MODELS", bool(self.models)),
            ("ENSEMBLE", bool(self.ensemble)),
            ("SCENARIOS", bool(self.scenarios)),
            ("EXPECTED VALUE", bool(self.expected_value)),
            ("RISK", bool(self.risk)),
            ("EXECUTION", bool(self.execution)),
            ("ORDER", bool(self.order)),
        ]


@dataclass(slots=True)
class DecisionRecord:
    decision_id: str
    ts: int
    kind: DecisionKind
    symbol: str
    mode: str = "paper"
    timeframe: str = ""
    side: str | None = None
    decision: str = ""
    confidence: float = 0.0
    entry: float = 0.0
    stop: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    risk_amount: float = 0.0
    trade_quality: float = 0.0
    expected_r: float = 0.0
    reasoning: list[str] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)
    trace: DecisionTrace = field(default_factory=DecisionTrace)
    parent_id: str = ""
    outcome: dict[str, Any] = field(default_factory=dict)
    lesson: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "ts": self.ts,
            "kind": self.kind.value,
            "symbol": self.symbol,
            "mode": self.mode,
            "timeframe": self.timeframe,
            "side": self.side,
            "decision": self.decision,
            "confidence": round(self.confidence, 4),
            "entry": self.entry,
            "stop": self.stop,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "tp3": self.tp3,
            "risk_amount": round(self.risk_amount, 4),
            "trade_quality": round(self.trade_quality, 2),
            "expected_r": round(self.expected_r, 4),
            "reasoning": self.reasoning,
            "rejections": self.rejections,
            "trace": self.trace.as_dict(),
            "parent_id": self.parent_id,
            "outcome": self.outcome,
            "lesson": self.lesson,
        }

    def replay(self) -> str:
        """Human-readable reconstruction of the decision."""

        lines = [
            f"DECISION {self.decision_id}  ({self.kind.value})",
            f"  when     {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(self.ts))}",
            f"  symbol   {self.symbol} {self.timeframe}",
            f"  mode     {self.mode}",
            f"  verdict  {self.decision}"
            + (f" {self.side.upper()}" if self.side else ""),
            f"  quality  {self.trade_quality:.0f}/100   confidence {self.confidence:.0%}"
            f"   EV {self.expected_r:+.3f}R",
            "",
            "PIPELINE",
        ]
        for stage, present in self.trace.stages():
            lines.append(f"  {'✓' if present else '·'} {stage}")

        if self.trace.regime:
            lines += ["", "REGIME", f"  {self.trace.regime.get('regime', '?')}"]

        if self.trace.models:
            lines += ["", "MODELS"]
            for model in self.trace.models:
                lines.append(
                    f"  {model.get('name', '?'):18s} {model.get('signal', '?'):10s}"
                    f" conf {model.get('confidence', 0):.0%}"
                )

        if self.trace.ensemble:
            ensemble = self.trace.ensemble
            lines += [
                "",
                "ENSEMBLE",
                f"  {ensemble.get('signal', '?')} at {ensemble.get('confidence', 0):.0%}"
                f"  agreement {ensemble.get('model_agreement', '?')}"
                f"  participation {ensemble.get('participation', 0):.0%}",
            ]

        if self.reasoning:
            lines += ["", "WHY"] + [f"  + {r}" for r in self.reasoning]
        if self.rejections:
            lines += ["", "WHY NOT"] + [f"  - {r}" for r in self.rejections]

        if self.trace.risk:
            lines += ["", "RISK"]
            for key, value in self.trace.risk.items():
                lines.append(f"  {key}: {value}")

        if self.outcome:
            lines += ["", "OUTCOME"]
            for key, value in self.outcome.items():
                lines.append(f"  {key}: {value}")
        if self.lesson:
            lines += ["", f"LESSON: {self.lesson}"]
        return "\n".join(lines)


class DecisionJournal:
    """Append-only journal. Entries are added, never modified."""

    def __init__(self, repositories: Any = None, memory_limit: int = 500) -> None:
        self.repositories = repositories
        self.memory_limit = memory_limit
        self._records: list[DecisionRecord] = []

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:16]

    def record(
        self,
        kind: DecisionKind,
        symbol: str,
        **kwargs: Any,
    ) -> DecisionRecord:
        record = DecisionRecord(
            decision_id=kwargs.pop("decision_id", None) or self.new_id(),
            ts=int(kwargs.pop("ts", None) or time.time()),
            kind=kind,
            symbol=symbol,
            **kwargs,
        )
        self._records.append(record)
        if len(self._records) > self.memory_limit:
            self._records = self._records[-self.memory_limit :]
        self._persist(record)
        return record

    def record_outcome(
        self,
        parent_id: str,
        symbol: str,
        outcome: Mapping[str, Any],
        lesson: str = "",
    ) -> DecisionRecord:
        """Outcomes are *new* records linked to the decision, never edits of it."""

        return self.record(
            kind=DecisionKind.OUTCOME,
            symbol=symbol,
            parent_id=parent_id,
            outcome=dict(outcome),
            lesson=lesson,
        )

    # -- retrieval ---------------------------------------------------------

    def recent(
        self, limit: int = 20, symbol: str | None = None, kind: DecisionKind | None = None
    ) -> list[DecisionRecord]:
        records = self._records
        if symbol:
            records = [r for r in records if r.symbol.upper() == symbol.upper()]
        if kind:
            records = [r for r in records if r.kind is kind]
        return list(reversed(records[-limit:]))

    def get(self, decision_id: str) -> DecisionRecord | None:
        for record in reversed(self._records):
            if record.decision_id == decision_id:
                return record
        if self.repositories is not None:
            try:
                row = self.repositories.journal.get(decision_id)
                if row:
                    return _record_from_row(row)
            except Exception:  # noqa: BLE001
                return None
        return None

    def outcomes_for(self, decision_id: str) -> list[DecisionRecord]:
        return [r for r in self._records if r.parent_id == decision_id]

    def replay(self, decision_id: str) -> str:
        record = self.get(decision_id)
        if record is None:
            return f"no journal entry with id {decision_id}"
        text = record.replay()
        outcomes = self.outcomes_for(decision_id)
        if outcomes:
            text += "\n\nLINKED OUTCOMES"
            for outcome in outcomes:
                text += f"\n  {outcome.outcome}"
                if outcome.lesson:
                    text += f"\n  lesson: {outcome.lesson}"
        return text

    def statistics(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        for record in self._records:
            by_kind[record.kind.value] = by_kind.get(record.kind.value, 0) + 1
        entries = [r for r in self._records if r.kind is DecisionKind.ENTRY]
        no_trades = [r for r in self._records if r.kind is DecisionKind.NO_TRADE]
        total = len(entries) + len(no_trades)
        return {
            "records": len(self._records),
            "by_kind": by_kind,
            "selectivity": (
                round(len(no_trades) / total, 4) if total else None
            ),
            "avg_trade_quality": (
                round(sum(r.trade_quality for r in entries) / len(entries), 2)
                if entries
                else None
            ),
        }

    def _persist(self, record: DecisionRecord) -> None:
        if self.repositories is None:
            return
        try:
            self.repositories.journal.append(record.as_dict())
        except Exception:  # noqa: BLE001 - the journal must never break trading
            pass


def _record_from_row(row: Mapping[str, Any]) -> DecisionRecord:
    trace_data = row.get("trace") or {}
    if isinstance(trace_data, str):
        try:
            trace_data = json.loads(trace_data)
        except ValueError:
            trace_data = {}
    trace = DecisionTrace(
        data=trace_data.get("data", {}),
        features=trace_data.get("features", {}),
        regime=trace_data.get("regime", {}),
        models=trace_data.get("models", []),
        ensemble=trace_data.get("ensemble", {}),
        scenarios=trace_data.get("scenarios", {}),
        expected_value=trace_data.get("expected_value", {}),
        risk=trace_data.get("risk", {}),
        execution=trace_data.get("execution", {}),
        order=trace_data.get("order", {}),
    )
    return DecisionRecord(
        decision_id=str(row.get("decision_id", "")),
        ts=int(row.get("ts", 0)),
        kind=DecisionKind(str(row.get("kind", "NO_TRADE"))),
        symbol=str(row.get("symbol", "")),
        mode=str(row.get("mode", "paper")),
        timeframe=str(row.get("timeframe", "")),
        side=row.get("side"),
        decision=str(row.get("decision", "")),
        confidence=float(row.get("confidence", 0.0) or 0.0),
        entry=float(row.get("entry", 0.0) or 0.0),
        stop=float(row.get("stop", 0.0) or 0.0),
        tp1=float(row.get("tp1", 0.0) or 0.0),
        tp2=float(row.get("tp2", 0.0) or 0.0),
        tp3=float(row.get("tp3", 0.0) or 0.0),
        risk_amount=float(row.get("risk_amount", 0.0) or 0.0),
        trade_quality=float(row.get("trade_quality", 0.0) or 0.0),
        expected_r=float(row.get("expected_r", 0.0) or 0.0),
        reasoning=row.get("reasoning") or [],
        rejections=row.get("rejections") or [],
        trace=trace,
        parent_id=str(row.get("parent_id", "") or ""),
        outcome=row.get("outcome") or {},
        lesson=str(row.get("lesson", "") or ""),
    )


__all__ = ["DecisionJournal", "DecisionRecord", "DecisionTrace", "DecisionKind"]

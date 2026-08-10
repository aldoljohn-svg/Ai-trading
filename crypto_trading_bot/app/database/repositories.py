"""Repositories - the only place that knows table/column names.

Every repository takes a :class:`~app.database.database.Database` so tests can
run against an in-memory SQLite database.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Sequence

from app.database.database import Database, dumps, loads
from app.domain import Candle, Side, Timeframe


def _now() -> int:
    return int(time.time())


class CandleRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def save(self, symbol: str, timeframe: Timeframe | str, candles: Iterable[Candle]) -> int:
        tf = timeframe.value if isinstance(timeframe, Timeframe) else str(timeframe)
        rows = [
            {
                "symbol": symbol,
                "timeframe": tf,
                "ts": c.ts,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
                "quote_volume": c.quote_volume,
            }
            for c in candles
            if c.closed
        ]
        return self.db.upsert_many("candles", rows, conflict=("symbol", "timeframe", "ts"))

    def load(
        self,
        symbol: str,
        timeframe: Timeframe | str,
        limit: int = 500,
        end_ts: int | None = None,
    ) -> list[Candle]:
        tf = timeframe.value if isinstance(timeframe, Timeframe) else str(timeframe)
        sql = "SELECT * FROM candles WHERE symbol=? AND timeframe=?"
        params: list[Any] = [symbol, tf]
        if end_ts is not None:
            sql += " AND ts <= ?"
            params.append(end_ts)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        rows = self.db.query(sql, params)
        rows.reverse()
        return [
            Candle(
                ts=int(r["ts"]),
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                volume=float(r["volume"]),
                quote_volume=float(r["quote_volume"]),
                closed=True,
            )
            for r in rows
        ]

    def last_ts(self, symbol: str, timeframe: Timeframe | str) -> int | None:
        tf = timeframe.value if isinstance(timeframe, Timeframe) else str(timeframe)
        value = self.db.scalar(
            "SELECT MAX(ts) AS m FROM candles WHERE symbol=? AND timeframe=?",
            (symbol, tf),
        )
        return int(value) if value is not None else None

    def count(self) -> int:
        return int(self.db.scalar("SELECT COUNT(*) AS c FROM candles", default=0))


class SignalRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def record(self, payload: dict[str, Any]) -> int:
        row = {
            "ts": payload.get("ts", _now()),
            "symbol": payload["symbol"],
            "side": payload.get("side"),
            "decision": payload.get("decision", "NO_TRADE"),
            "confidence": float(payload.get("confidence", 0.0)),
            "rr": float(payload.get("rr", 0.0)),
            "entry": payload.get("entry"),
            "stop_loss": payload.get("stop_loss"),
            "tp1": payload.get("tp1"),
            "tp2": payload.get("tp2"),
            "tp3": payload.get("tp3"),
            "regime": payload.get("regime"),
            "htf_bias": payload.get("htf_bias"),
            "p_long": payload.get("p_long"),
            "p_short": payload.get("p_short"),
            "p_no_trade": payload.get("p_no_trade"),
            "reasons": dumps(payload.get("reasons", [])),
            "rejections": dumps(payload.get("rejections", [])),
            "scores": dumps(payload.get("scores", {})),
            "features": dumps(payload.get("features", {})),
        }
        return self.db.insert("signals", row)

    def recent(self, limit: int = 50, symbol: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM signals"
        params: list[Any] = []
        if symbol:
            sql += " WHERE symbol=?"
            params.append(symbol)
        sql += " ORDER BY ts DESC, id DESC LIMIT ?"
        params.append(limit)
        rows = self.db.query(sql, params)
        for row in rows:
            for key in ("reasons", "rejections", "scores", "features"):
                row[key] = loads(row.get(key), [] if key in {"reasons", "rejections"} else {})
        return rows


class OpportunityRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def record_batch(self, rows: Sequence[dict[str, Any]]) -> int:
        if not rows:
            return 0
        prepared = []
        for row in rows:
            prepared.append(
                {
                    "ts": row.get("ts", _now()),
                    "symbol": row["symbol"],
                    "opportunity_score": float(row.get("opportunity_score", 0.0)),
                    "technical_score": float(row.get("technical_score", 0.0)),
                    "structure_score": float(row.get("structure_score", 0.0)),
                    "ict_score": float(row.get("ict_score", 0.0)),
                    "rtm_score": float(row.get("rtm_score", 0.0)),
                    "momentum_score": float(row.get("momentum_score", 0.0)),
                    "volume_score": float(row.get("volume_score", 0.0)),
                    "volatility_score": float(row.get("volatility_score", 0.0)),
                    "htf_alignment_score": float(row.get("htf_alignment_score", 0.0)),
                    "fundamental_score": float(row.get("fundamental_score", 0.0)),
                    "ml_probability": float(row.get("ml_probability", 0.0)),
                    "rr_score": float(row.get("rr_score", 0.0)),
                    "trend": row.get("trend"),
                    "regime": row.get("regime"),
                    "status": row.get("status"),
                }
            )
        cols = list(prepared[0])
        placeholders = ", ".join("?" for _ in cols)
        sql = f"INSERT INTO opportunities ({', '.join(cols)}) VALUES ({placeholders})"
        return self.db.execute_many(sql, [[p[c] for c in cols] for p in prepared])

    def latest(self, limit: int = 25) -> list[dict[str, Any]]:
        latest_ts = self.db.scalar("SELECT MAX(ts) AS m FROM opportunities")
        if latest_ts is None:
            return []
        return self.db.query(
            "SELECT * FROM opportunities WHERE ts=? "
            "ORDER BY opportunity_score DESC LIMIT ?",
            (latest_ts, limit),
        )


class OrderRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, order: dict[str, Any]) -> int:
        now = _now()
        row = {
            "client_order_id": order["client_order_id"],
            "exchange_order_id": order.get("exchange_order_id"),
            "trade_id": order.get("trade_id"),
            "symbol": order["symbol"],
            "side": order["side"],
            "intent": order["intent"],
            "order_type": order["order_type"],
            "quantity": float(order["quantity"]),
            "price": order.get("price"),
            "status": order.get("status", "new"),
            "filled_quantity": float(order.get("filled_quantity", 0.0)),
            "average_price": float(order.get("average_price", 0.0)),
            "reduce_only": 1 if order.get("reduce_only") else 0,
            "mode": order.get("mode", "paper"),
            "created_at": now,
            "updated_at": now,
            "error": order.get("error"),
        }
        return self.db.upsert("orders", row, conflict=("client_order_id",))

    def update_status(
        self,
        client_order_id: str,
        status: str,
        filled_quantity: float | None = None,
        average_price: float | None = None,
        exchange_order_id: str | None = None,
        error: str | None = None,
    ) -> None:
        values: dict[str, Any] = {"status": status, "updated_at": _now()}
        if filled_quantity is not None:
            values["filled_quantity"] = float(filled_quantity)
        if average_price is not None:
            values["average_price"] = float(average_price)
        if exchange_order_id is not None:
            values["exchange_order_id"] = exchange_order_id
        if error is not None:
            values["error"] = error
        self.db.update("orders", values, {"client_order_id": client_order_id})

    def get(self, client_order_id: str) -> dict[str, Any] | None:
        return self.db.query_one(
            "SELECT * FROM orders WHERE client_order_id=?", (client_order_id,)
        )

    def open_orders(self, mode: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM orders WHERE status IN ('new','partially_filled')"
        params: list[Any] = []
        if mode:
            sql += " AND mode=?"
            params.append(mode)
        return self.db.query(sql + " ORDER BY created_at DESC", params)

    def recent(self, limit: int = 30) -> list[dict[str, Any]]:
        return self.db.query(
            "SELECT * FROM orders ORDER BY created_at DESC, id DESC LIMIT ?", (limit,)
        )


class FillRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def record(self, fill: dict[str, Any]) -> int:
        row = {
            "order_client_id": fill["order_client_id"],
            "exchange_trade_id": fill.get("exchange_trade_id", ""),
            "symbol": fill["symbol"],
            "side": fill["side"],
            "quantity": float(fill["quantity"]),
            "price": float(fill["price"]),
            "fee": float(fill.get("fee", 0.0)),
            "is_maker": 1 if fill.get("is_maker") else 0,
            "ts": int(fill.get("ts", _now())),
        }
        return self.db.upsert(
            "fills", row, conflict=("order_client_id", "exchange_trade_id")
        )

    def for_order(self, client_order_id: str) -> list[dict[str, Any]]:
        return self.db.query(
            "SELECT * FROM fills WHERE order_client_id=? ORDER BY ts", (client_order_id,)
        )


class PositionRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def save(self, position: dict[str, Any]) -> int:
        now = _now()
        row = {
            "trade_id": position.get("trade_id"),
            "symbol": position["symbol"],
            "side": position["side"],
            "quantity": float(position["quantity"]),
            "entry_price": float(position["entry_price"]),
            "leverage": float(position.get("leverage", 1.0)),
            "stop_loss": position.get("stop_loss"),
            "tp1": position.get("tp1"),
            "tp2": position.get("tp2"),
            "tp3": position.get("tp3"),
            "initial_stop": position.get("initial_stop"),
            "initial_quantity": position.get("initial_quantity"),
            "risk_amount": float(position.get("risk_amount", 0.0)),
            "state": position.get("state", "open"),
            "breakeven_done": 1 if position.get("breakeven_done") else 0,
            "trailing_active": 1 if position.get("trailing_active") else 0,
            "tp1_done": 1 if position.get("tp1_done") else 0,
            "tp2_done": 1 if position.get("tp2_done") else 0,
            "opened_at": int(position.get("opened_at", now)),
            "updated_at": now,
            "mode": position.get("mode", "paper"),
            "meta": dumps(position.get("meta", {})),
        }
        existing_id = position.get("id")
        if existing_id:
            self.db.update("positions", row, {"id": existing_id})
            return int(existing_id)
        return self.db.insert("positions", row)

    def open_positions(self, mode: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM positions WHERE state='open'"
        params: list[Any] = []
        if mode:
            sql += " AND mode=?"
            params.append(mode)
        rows = self.db.query(sql + " ORDER BY opened_at", params)
        for row in rows:
            row["meta"] = loads(row.get("meta"), {})
        return rows

    def close(self, position_id: int) -> None:
        self.db.update(
            "positions", {"state": "closed", "updated_at": _now()}, {"id": position_id}
        )


class TradeRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def open_trade(self, trade: dict[str, Any]) -> int:
        row = {
            "symbol": trade["symbol"],
            "side": trade["side"],
            "mode": trade.get("mode", "paper"),
            "entry_price": float(trade["entry_price"]),
            "exit_price": None,
            "quantity": float(trade["quantity"]),
            "leverage": float(trade.get("leverage", 1.0)),
            "stop_loss": trade.get("stop_loss"),
            "tp1": trade.get("tp1"),
            "tp2": trade.get("tp2"),
            "tp3": trade.get("tp3"),
            "risk_amount": float(trade.get("risk_amount", 0.0)),
            "planned_rr": float(trade.get("planned_rr", 0.0)),
            "realized_pnl": 0.0,
            "fees": float(trade.get("fees", 0.0)),
            "funding": 0.0,
            "r_multiple": 0.0,
            "confidence": float(trade.get("confidence", 0.0)),
            "regime": trade.get("regime"),
            "opened_at": int(trade.get("opened_at", _now())),
            "closed_at": None,
            "status": "open",
            "why_entered": dumps(trade.get("why_entered", [])),
            "why_exited": None,
            "features": dumps(trade.get("features", {})),
        }
        return self.db.insert("trades", row)

    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        realized_pnl: float,
        fees: float,
        funding: float,
        r_multiple: float,
        why_exited: Any,
    ) -> None:
        self.db.update(
            "trades",
            {
                "exit_price": float(exit_price),
                "realized_pnl": float(realized_pnl),
                "fees": float(fees),
                "funding": float(funding),
                "r_multiple": float(r_multiple),
                "closed_at": _now(),
                "status": "closed",
                "why_exited": dumps(why_exited),
            },
            {"id": trade_id},
        )

    def get(self, trade_id: int) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT * FROM trades WHERE id=?", (trade_id,))
        if row:
            row["why_entered"] = loads(row.get("why_entered"), [])
            row["why_exited"] = loads(row.get("why_exited"), [])
            row["features"] = loads(row.get("features"), {})
        return row

    def closed_since(self, since_ts: int, mode: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM trades WHERE status='closed' AND closed_at >= ?"
        params: list[Any] = [since_ts]
        if mode:
            sql += " AND mode=?"
            params.append(mode)
        return self.db.query(sql + " ORDER BY closed_at", params)

    def recent(self, limit: int = 25, mode: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM trades"
        params: list[Any] = []
        if mode:
            sql += " WHERE mode=?"
            params.append(mode)
        sql += " ORDER BY opened_at DESC, id DESC LIMIT ?"
        params.append(limit)
        rows = self.db.query(sql, params)
        for row in rows:
            row["why_entered"] = loads(row.get("why_entered"), [])
            row["why_exited"] = loads(row.get("why_exited"), [])
        return rows

    def realized_pnl_since(self, since_ts: int, mode: str | None = None) -> float:
        sql = "SELECT COALESCE(SUM(realized_pnl), 0) AS s FROM trades WHERE status='closed' AND closed_at >= ?"
        params: list[Any] = [since_ts]
        if mode:
            sql += " AND mode=?"
            params.append(mode)
        return float(self.db.scalar(sql, params, default=0.0) or 0.0)

    def consecutive_losses(self, mode: str | None = None, limit: int = 20) -> int:
        sql = "SELECT realized_pnl FROM trades WHERE status='closed'"
        params: list[Any] = []
        if mode:
            sql += " AND mode=?"
            params.append(mode)
        sql += " ORDER BY closed_at DESC, id DESC LIMIT ?"
        params.append(limit)
        streak = 0
        for row in self.db.query(sql, params):
            if float(row["realized_pnl"] or 0.0) < 0:
                streak += 1
            else:
                break
        return streak


class AccountRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def snapshot(self, data: dict[str, Any]) -> int:
        row = {
            "ts": int(data.get("ts", _now())),
            "mode": data.get("mode", "paper"),
            "equity": float(data.get("equity", 0.0)),
            "balance": float(data.get("balance", 0.0)),
            "available": float(data.get("available", 0.0)),
            "used_margin": float(data.get("used_margin", 0.0)),
            "unrealized_pnl": float(data.get("unrealized_pnl", 0.0)),
            "realized_pnl_day": float(data.get("realized_pnl_day", 0.0)),
            "open_positions": int(data.get("open_positions", 0)),
            "portfolio_risk": float(data.get("portfolio_risk", 0.0)),
            "drawdown": float(data.get("drawdown", 0.0)),
        }
        return self.db.insert("account_snapshots", row)

    def peak_equity(self, mode: str | None = None) -> float:
        sql = "SELECT MAX(equity) AS m FROM account_snapshots"
        params: list[Any] = []
        if mode:
            sql += " WHERE mode=?"
            params.append(mode)
        return float(self.db.scalar(sql, params, default=0.0) or 0.0)

    def equity_at_or_before(self, ts: int, mode: str | None = None) -> float | None:
        sql = "SELECT equity FROM account_snapshots WHERE ts <= ?"
        params: list[Any] = [ts]
        if mode:
            sql += " AND mode=?"
            params.append(mode)
        sql += " ORDER BY ts DESC LIMIT 1"
        row = self.db.query_one(sql, params)
        return float(row["equity"]) if row else None

    def history(self, limit: int = 500, mode: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM account_snapshots"
        params: list[Any] = []
        if mode:
            sql += " WHERE mode=?"
            params.append(mode)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        rows = self.db.query(sql, params)
        rows.reverse()
        return rows


class EventRepository:
    """Risk events + system events - the audit trail."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def risk_event(
        self,
        kind: str,
        message: str,
        severity: str = "WARNING",
        symbol: str | None = None,
        detail: Any = None,
    ) -> int:
        return self.db.insert(
            "risk_events",
            {
                "ts": _now(),
                "kind": kind,
                "severity": severity,
                "symbol": symbol,
                "message": message,
                "detail": dumps(detail or {}),
            },
        )

    def system_event(
        self, component: str, message: str, level: str = "INFO", detail: Any = None
    ) -> int:
        return self.db.insert(
            "system_events",
            {
                "ts": _now(),
                "component": component,
                "level": level,
                "message": message,
                "detail": dumps(detail or {}),
            },
        )

    def recent_risk(self, limit: int = 25) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM risk_events ORDER BY ts DESC, id DESC LIMIT ?", (limit,)
        )
        for row in rows:
            row["detail"] = loads(row.get("detail"), {})
        return rows

    def recent_system(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM system_events ORDER BY ts DESC, id DESC LIMIT ?", (limit,)
        )
        for row in rows:
            row["detail"] = loads(row.get("detail"), {})
        return rows


class FeatureRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def save(
        self,
        symbol: str,
        timeframe: str,
        ts: int,
        payload: dict[str, float],
        label: int | None = None,
        label_horizon: int | None = None,
    ) -> int:
        return self.db.upsert(
            "features",
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "ts": ts,
                "payload": dumps(payload),
                "label": label,
                "label_horizon": label_horizon,
                "created_at": _now(),
            },
            conflict=("symbol", "timeframe", "ts"),
        )

    def labelled(self, limit: int = 100_000) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM features WHERE label IS NOT NULL ORDER BY ts LIMIT ?",
            (limit,),
        )
        for row in rows:
            row["payload"] = loads(row.get("payload"), {})
        return rows

    def set_label(self, symbol: str, timeframe: str, ts: int, label: int, horizon: int) -> None:
        self.db.update(
            "features",
            {"label": label, "label_horizon": horizon},
            {"symbol": symbol, "timeframe": timeframe, "ts": ts},
        )


class ModelRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def register(self, meta: dict[str, Any]) -> int:
        row = {
            "name": meta["name"],
            "version": meta["version"],
            "algorithm": meta.get("algorithm", "unknown"),
            "trained_at": int(meta.get("trained_at", _now())),
            "rows": int(meta.get("rows", 0)),
            "metrics": dumps(meta.get("metrics", {})),
            "feature_names": dumps(meta.get("feature_names", [])),
            "path": meta.get("path"),
            "active": 1 if meta.get("active") else 0,
        }
        return self.db.upsert("model_versions", row, conflict=("name", "version"))

    def activate(self, name: str, version: str) -> None:
        self.db.execute("UPDATE model_versions SET active=0 WHERE name=?", (name,))
        self.db.execute(
            "UPDATE model_versions SET active=1 WHERE name=? AND version=?",
            (name, version),
        )

    def active(self, name: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT * FROM model_versions WHERE name=? AND active=1 "
            "ORDER BY trained_at DESC LIMIT 1",
            (name,),
        )
        if row:
            row["metrics"] = loads(row.get("metrics"), {})
            row["feature_names"] = loads(row.get("feature_names"), [])
        return row

    def all(self, name: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM model_versions"
        params: list[Any] = []
        if name:
            sql += " WHERE name=?"
            params.append(name)
        return self.db.query(sql + " ORDER BY trained_at DESC", params)


class BacktestRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def save(self, result: dict[str, Any]) -> int:
        return self.db.upsert(
            "backtest_results",
            {
                "run_id": result["run_id"],
                "created_at": int(result.get("created_at", _now())),
                "symbols": dumps(result.get("symbols", [])),
                "start_ts": result.get("start_ts"),
                "end_ts": result.get("end_ts"),
                "config": dumps(result.get("config", {})),
                "metrics": dumps(result.get("metrics", {})),
                "equity_curve": dumps(result.get("equity_curve", [])),
                "trades": int(result.get("trades", 0)),
            },
            conflict=("run_id",),
        )

    def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM backtest_results ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        for row in rows:
            row["metrics"] = loads(row.get("metrics"), {})
            row["symbols"] = loads(row.get("symbols"), [])
        return rows


class Repositories:
    """Convenience bundle passed around the application."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.candles = CandleRepository(db)
        self.features = FeatureRepository(db)
        self.signals = SignalRepository(db)
        self.opportunities = OpportunityRepository(db)
        self.orders = OrderRepository(db)
        self.fills = FillRepository(db)
        self.positions = PositionRepository(db)
        self.trades = TradeRepository(db)
        self.account = AccountRepository(db)
        self.events = EventRepository(db)
        self.models = ModelRepository(db)
        self.backtests = BacktestRepository(db)


__all__ = [
    "Repositories",
    "CandleRepository",
    "FeatureRepository",
    "SignalRepository",
    "OpportunityRepository",
    "OrderRepository",
    "FillRepository",
    "PositionRepository",
    "TradeRepository",
    "AccountRepository",
    "EventRepository",
    "ModelRepository",
    "BacktestRepository",
]

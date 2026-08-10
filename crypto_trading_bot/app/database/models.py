"""Schema definition - the single source of truth for all persisted tables.

Tables are declared once as :class:`TableSpec` objects and rendered to dialect
specific DDL.  Portable column types are used everywhere:

==========  ==============  ======================
logical     SQLite          PostgreSQL
==========  ==============  ======================
``PK``      INTEGER PK AI   BIGSERIAL PRIMARY KEY
``INT``     INTEGER         BIGINT
``REAL``    REAL            DOUBLE PRECISION
``TEXT``    TEXT            TEXT
``BOOL``    INTEGER         BOOLEAN
``JSON``    TEXT            TEXT
==========  ==============  ======================
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Column:
    name: str
    type: str
    null: bool = True
    default: str | None = None

    def ddl(self, dialect: str) -> str:
        if self.type == "PK":
            return (
                f"{self.name} INTEGER PRIMARY KEY AUTOINCREMENT"
                if dialect == "sqlite"
                else f"{self.name} BIGSERIAL PRIMARY KEY"
            )
        mapping = {
            "INT": "INTEGER" if dialect == "sqlite" else "BIGINT",
            "REAL": "REAL" if dialect == "sqlite" else "DOUBLE PRECISION",
            "TEXT": "TEXT",
            "BOOL": "INTEGER" if dialect == "sqlite" else "BOOLEAN",
            "JSON": "TEXT",
        }
        sql_type = mapping.get(self.type)
        if sql_type is None:
            raise ValueError(f"unknown column type {self.type!r}")
        parts = [self.name, sql_type]
        if not self.null:
            parts.append("NOT NULL")
        if self.default is not None:
            parts.append(f"DEFAULT {self.default}")
        return " ".join(parts)


@dataclass(frozen=True, slots=True)
class TableSpec:
    name: str
    columns: tuple[Column, ...]
    unique: tuple[tuple[str, ...], ...] = ()
    indexes: tuple[tuple[str, ...], ...] = ()

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    def create_sql(self, dialect: str) -> str:
        body = [c.ddl(dialect) for c in self.columns]
        for cols in self.unique:
            body.append(f"UNIQUE ({', '.join(cols)})")
        joined = ",\n    ".join(body)
        return f"CREATE TABLE IF NOT EXISTS {self.name} (\n    {joined}\n)"

    def index_sql(self) -> list[str]:
        out = []
        for cols in self.indexes:
            index_name = f"idx_{self.name}_{'_'.join(cols)}"
            out.append(
                f"CREATE INDEX IF NOT EXISTS {index_name} "
                f"ON {self.name} ({', '.join(cols)})"
            )
        return out


def _c(name: str, type_: str, null: bool = True, default: str | None = None) -> Column:
    return Column(name=name, type=type_, null=null, default=default)


CANDLES = TableSpec(
    name="candles",
    columns=(
        _c("id", "PK"),
        _c("symbol", "TEXT", null=False),
        _c("timeframe", "TEXT", null=False),
        _c("ts", "INT", null=False),
        _c("open", "REAL", null=False),
        _c("high", "REAL", null=False),
        _c("low", "REAL", null=False),
        _c("close", "REAL", null=False),
        _c("volume", "REAL", null=False, default="0"),
        _c("quote_volume", "REAL", null=False, default="0"),
    ),
    unique=(("symbol", "timeframe", "ts"),),
    indexes=(("symbol", "timeframe", "ts"),),
)

FEATURES = TableSpec(
    name="features",
    columns=(
        _c("id", "PK"),
        _c("symbol", "TEXT", null=False),
        _c("timeframe", "TEXT", null=False),
        _c("ts", "INT", null=False),
        _c("payload", "JSON", null=False),
        _c("label", "INT"),
        _c("label_horizon", "INT"),
        _c("created_at", "INT", null=False, default="0"),
    ),
    unique=(("symbol", "timeframe", "ts"),),
    indexes=(("symbol", "ts"), ("label",)),
)

SIGNALS = TableSpec(
    name="signals",
    columns=(
        _c("id", "PK"),
        _c("ts", "INT", null=False),
        _c("symbol", "TEXT", null=False),
        _c("side", "TEXT"),
        _c("decision", "TEXT", null=False),
        _c("confidence", "REAL", null=False, default="0"),
        _c("rr", "REAL", null=False, default="0"),
        _c("entry", "REAL"),
        _c("stop_loss", "REAL"),
        _c("tp1", "REAL"),
        _c("tp2", "REAL"),
        _c("tp3", "REAL"),
        _c("regime", "TEXT"),
        _c("htf_bias", "TEXT"),
        _c("p_long", "REAL"),
        _c("p_short", "REAL"),
        _c("p_no_trade", "REAL"),
        _c("reasons", "JSON"),
        _c("rejections", "JSON"),
        _c("scores", "JSON"),
        _c("features", "JSON"),
    ),
    indexes=(("symbol", "ts"), ("decision",)),
)

OPPORTUNITIES = TableSpec(
    name="opportunities",
    columns=(
        _c("id", "PK"),
        _c("ts", "INT", null=False),
        _c("symbol", "TEXT", null=False),
        _c("opportunity_score", "REAL", null=False, default="0"),
        _c("technical_score", "REAL", default="0"),
        _c("structure_score", "REAL", default="0"),
        _c("ict_score", "REAL", default="0"),
        _c("rtm_score", "REAL", default="0"),
        _c("momentum_score", "REAL", default="0"),
        _c("volume_score", "REAL", default="0"),
        _c("volatility_score", "REAL", default="0"),
        _c("htf_alignment_score", "REAL", default="0"),
        _c("fundamental_score", "REAL", default="0"),
        _c("ml_probability", "REAL", default="0"),
        _c("rr_score", "REAL", default="0"),
        _c("trend", "TEXT"),
        _c("regime", "TEXT"),
        _c("status", "TEXT"),
    ),
    indexes=(("ts",), ("symbol", "ts")),
)

ORDERS = TableSpec(
    name="orders",
    columns=(
        _c("id", "PK"),
        _c("client_order_id", "TEXT", null=False),
        _c("exchange_order_id", "TEXT"),
        _c("trade_id", "INT"),
        _c("symbol", "TEXT", null=False),
        _c("side", "TEXT", null=False),
        _c("intent", "TEXT", null=False),
        _c("order_type", "TEXT", null=False),
        _c("quantity", "REAL", null=False),
        _c("price", "REAL"),
        _c("status", "TEXT", null=False),
        _c("filled_quantity", "REAL", null=False, default="0"),
        _c("average_price", "REAL", null=False, default="0"),
        _c("reduce_only", "BOOL", null=False, default="0"),
        _c("mode", "TEXT", null=False),
        _c("created_at", "INT", null=False),
        _c("updated_at", "INT", null=False),
        _c("error", "TEXT"),
    ),
    unique=(("client_order_id",),),
    indexes=(("symbol", "status"), ("trade_id",)),
)

FILLS = TableSpec(
    name="fills",
    columns=(
        _c("id", "PK"),
        _c("order_client_id", "TEXT", null=False),
        _c("exchange_trade_id", "TEXT"),
        _c("symbol", "TEXT", null=False),
        _c("side", "TEXT", null=False),
        _c("quantity", "REAL", null=False),
        _c("price", "REAL", null=False),
        _c("fee", "REAL", null=False, default="0"),
        _c("is_maker", "BOOL", null=False, default="0"),
        _c("ts", "INT", null=False),
    ),
    unique=(("order_client_id", "exchange_trade_id"),),
    indexes=(("symbol", "ts"),),
)

POSITIONS = TableSpec(
    name="positions",
    columns=(
        _c("id", "PK"),
        _c("trade_id", "INT"),
        _c("symbol", "TEXT", null=False),
        _c("side", "TEXT", null=False),
        _c("quantity", "REAL", null=False),
        _c("entry_price", "REAL", null=False),
        _c("leverage", "REAL", null=False, default="1"),
        _c("stop_loss", "REAL"),
        _c("tp1", "REAL"),
        _c("tp2", "REAL"),
        _c("tp3", "REAL"),
        _c("initial_stop", "REAL"),
        _c("initial_quantity", "REAL"),
        _c("risk_amount", "REAL", default="0"),
        _c("state", "TEXT", null=False),
        _c("breakeven_done", "BOOL", null=False, default="0"),
        _c("trailing_active", "BOOL", null=False, default="0"),
        _c("tp1_done", "BOOL", null=False, default="0"),
        _c("tp2_done", "BOOL", null=False, default="0"),
        _c("opened_at", "INT", null=False),
        _c("updated_at", "INT", null=False),
        _c("mode", "TEXT", null=False),
        _c("meta", "JSON"),
    ),
    indexes=(("symbol", "state"), ("state",)),
)

TRADES = TableSpec(
    name="trades",
    columns=(
        _c("id", "PK"),
        _c("symbol", "TEXT", null=False),
        _c("side", "TEXT", null=False),
        _c("mode", "TEXT", null=False),
        _c("entry_price", "REAL", null=False),
        _c("exit_price", "REAL"),
        _c("quantity", "REAL", null=False),
        _c("leverage", "REAL", null=False, default="1"),
        _c("stop_loss", "REAL"),
        _c("tp1", "REAL"),
        _c("tp2", "REAL"),
        _c("tp3", "REAL"),
        _c("risk_amount", "REAL", default="0"),
        _c("planned_rr", "REAL", default="0"),
        _c("realized_pnl", "REAL", default="0"),
        _c("fees", "REAL", default="0"),
        _c("funding", "REAL", default="0"),
        _c("r_multiple", "REAL", default="0"),
        _c("confidence", "REAL", default="0"),
        _c("regime", "TEXT"),
        _c("opened_at", "INT", null=False),
        _c("closed_at", "INT"),
        _c("status", "TEXT", null=False),
        _c("why_entered", "JSON"),
        _c("why_exited", "JSON"),
        _c("features", "JSON"),
    ),
    indexes=(("symbol",), ("status",), ("closed_at",)),
)

ACCOUNT_SNAPSHOTS = TableSpec(
    name="account_snapshots",
    columns=(
        _c("id", "PK"),
        _c("ts", "INT", null=False),
        _c("mode", "TEXT", null=False),
        _c("equity", "REAL", null=False),
        _c("balance", "REAL", null=False),
        _c("available", "REAL", null=False, default="0"),
        _c("used_margin", "REAL", null=False, default="0"),
        _c("unrealized_pnl", "REAL", null=False, default="0"),
        _c("realized_pnl_day", "REAL", null=False, default="0"),
        _c("open_positions", "INT", null=False, default="0"),
        _c("portfolio_risk", "REAL", null=False, default="0"),
        _c("drawdown", "REAL", null=False, default="0"),
    ),
    indexes=(("ts",),),
)

RISK_EVENTS = TableSpec(
    name="risk_events",
    columns=(
        _c("id", "PK"),
        _c("ts", "INT", null=False),
        _c("kind", "TEXT", null=False),
        _c("severity", "TEXT", null=False),
        _c("symbol", "TEXT"),
        _c("message", "TEXT", null=False),
        _c("detail", "JSON"),
    ),
    indexes=(("ts",), ("kind",)),
)

SYSTEM_EVENTS = TableSpec(
    name="system_events",
    columns=(
        _c("id", "PK"),
        _c("ts", "INT", null=False),
        _c("component", "TEXT", null=False),
        _c("level", "TEXT", null=False),
        _c("message", "TEXT", null=False),
        _c("detail", "JSON"),
    ),
    indexes=(("ts",), ("component",)),
)

MODEL_VERSIONS = TableSpec(
    name="model_versions",
    columns=(
        _c("id", "PK"),
        _c("name", "TEXT", null=False),
        _c("version", "TEXT", null=False),
        _c("algorithm", "TEXT", null=False),
        _c("trained_at", "INT", null=False),
        _c("rows", "INT", null=False, default="0"),
        _c("metrics", "JSON"),
        _c("feature_names", "JSON"),
        _c("path", "TEXT"),
        _c("active", "BOOL", null=False, default="0"),
    ),
    unique=(("name", "version"),),
    indexes=(("name", "active"),),
)

BACKTEST_RESULTS = TableSpec(
    name="backtest_results",
    columns=(
        _c("id", "PK"),
        _c("run_id", "TEXT", null=False),
        _c("created_at", "INT", null=False),
        _c("symbols", "JSON"),
        _c("start_ts", "INT"),
        _c("end_ts", "INT"),
        _c("config", "JSON"),
        _c("metrics", "JSON"),
        _c("equity_curve", "JSON"),
        _c("trades", "INT", null=False, default="0"),
    ),
    unique=(("run_id",),),
    indexes=(("created_at",),),
)

TABLES: tuple[TableSpec, ...] = (
    CANDLES,
    FEATURES,
    SIGNALS,
    OPPORTUNITIES,
    ORDERS,
    FILLS,
    POSITIONS,
    TRADES,
    ACCOUNT_SNAPSHOTS,
    RISK_EVENTS,
    SYSTEM_EVENTS,
    MODEL_VERSIONS,
    BACKTEST_RESULTS,
)

TABLES_BY_NAME: dict[str, TableSpec] = {t.name: t for t in TABLES}


def schema_sql(dialect: str = "sqlite") -> list[str]:
    statements: list[str] = []
    for table in TABLES:
        statements.append(table.create_sql(dialect))
        statements.extend(table.index_sql())
    return statements

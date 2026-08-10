# Autonomous Crypto Trading Bot — MEXC Futures

An autonomous, multi-timeframe futures trading system for MEXC. It discovers its
own universe, analyses many symbols, ranks opportunities, sizes positions from a
strict risk budget, executes, manages every open trade, and reports everything to
Telegram and a live web dashboard.

> ### Read this first
>
> **This software cannot predict the market and does not guarantee profit.** It is
> built around risk-adjusted performance and capital preservation. `NO TRADE` is a
> valid and *frequent* outcome — in normal conditions the bot rejects most of what
> it looks at, and that is the design working, not a fault.
>
> Leveraged futures can lose your entire deposit, and in fast markets a stop can
> fill far worse than its trigger price. Run it in **PAPER** mode for weeks before
> you even consider live money, and never fund it with money you cannot afford to
> lose.

---

## Table of contents

1. [What it does](#1-what-it-does)
2. [System & VPS requirements](#2-system--vps-requirements)
3. [Install prerequisites](#3-install-prerequisites)
4. [Get the code](#4-get-the-code)
5. [Configure `.env`](#5-configure-env)
6. [MEXC API setup](#6-mexc-api-setup)
7. [Telegram setup](#7-telegram-setup)
8. [Run in paper mode](#8-run-in-paper-mode)
9. [Non-Docker installation](#9-non-docker-installation)
10. [The dashboard](#10-the-dashboard)
11. [Telegram commands](#11-telegram-commands)
12. [Backtesting](#12-backtesting)
13. [Training the model](#13-training-the-model)
14. [Enabling live trading](#14-enabling-live-trading)
15. [Emergency stop & kill switch](#15-emergency-stop--kill-switch)
16. [Day-to-day operations](#16-day-to-day-operations)
17. [How it decides](#17-how-it-decides)
18. [Troubleshooting](#18-troubleshooting)
19. [Security checklist](#19-security-checklist)
20. [Project layout](#20-project-layout)
21. [Known limitations](#21-known-limitations)

---

## 1. What it does

**Every scan cycle (default 60s):**

1. Pulls every MEXC futures contract and one bulk ticker call.
2. Filters out inactive, illiquid, wide-spread and blacklisted symbols.
3. Ranks survivors on volume, volatility, movement and spread; keeps the top N.
4. Deep-analyses those on 5m/15m/30m/1h/4h/1d concurrently.
5. Runs indicators, market structure, ICT, RTM, regime detection and fundamentals.
6. Produces a scored, explained proposal for every symbol — including rejections.
7. Sends approved proposals to the risk engine, which decides size or refuses.
8. Executes only what survives every gate.

**Every few seconds:** each open position is re-evaluated for stops, targets,
break-even, trailing, structural invalidation and portfolio-level de-risking.

**Three modes:** `backtest`, `paper` (default), `live`.

---

## 2. System & VPS requirements

| | Minimum | Recommended |
|---|---|---|
| CPU | 2 vCPU | 4 vCPU |
| RAM | 2 GB | 4 GB |
| Disk | 10 GB SSD | 20 GB SSD |
| Network | stable, low latency to MEXC | same, plus a static IP for the API allowlist |

**Operating system:** Ubuntu 22.04 or 24.04 LTS (Debian 12 also fine). Any Linux
with Docker works.

**Python:** 3.12 or newer if running without Docker. The Docker image pins 3.12.

**Clock accuracy matters.** MEXC rejects signed requests whose timestamp drifts
more than a few seconds. Enable NTP:

```bash
sudo timedatectl set-ntp true
timedatectl status        # expect "System clock synchronized: yes"
```

Keep the server on **UTC** — the daily-loss breaker rolls on the UTC day.

---

## 3. Install prerequisites

### Docker (recommended)

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y ca-certificates curl gnupg git

# Docker's official repository
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io \
                    docker-buildx-plugin docker-compose-plugin

# Run docker without sudo (log out and back in afterwards)
sudo usermod -aG docker "$USER"
newgrp docker

docker --version
docker compose version
```

### Git

```bash
sudo apt install -y git
git --version
```

---

## 4. Get the code

```bash
git clone https://github.com/aldoljohn-svg/Ai-trading.git
cd Ai-trading/crypto_trading_bot
```

---

## 5. Configure `.env`

```bash
cp .env.example .env
nano .env
chmod 600 .env          # readable only by you
```

`.env.example` documents every option inline. The minimum for paper trading is
nothing at all — the defaults are safe and MEXC's public market data needs no
credentials.

**Precedence:** environment variables → `.env` → `config.yaml` → built-in
defaults. Secrets live only in `.env`; `config.yaml` is validated to reject any
key that looks like a credential, so it stays safe to commit.

The most important values:

```ini
TRADING_MODE=paper              # paper | backtest | live
DEFAULT_RISK_PER_TRADE=0.005    # 0.5% of equity per trade
MAX_DAILY_LOSS=0.02             # 2% — halts new entries for the UTC day
MAX_PORTFOLIO_RISK=0.02         # 2% correlation-adjusted open risk
MAX_OPEN_POSITIONS=3
MAX_LEVERAGE=3
MIN_CONFIDENCE=0.70
MIN_RR=2.0
MAX_SYMBOLS_TO_SCAN=100
SCANNER_INTERVAL_SECONDS=60
```

Validate before starting:

```bash
docker compose run --rm bot python -m app.main --check-config
```

The bot **refuses to start** on an invalid or dangerous configuration rather
than silently falling back to a default.

---

## 6. MEXC API setup

Only needed for account access and live trading. Public market data works
without keys.

1. Log in to MEXC → avatar → **API Management**.
2. **Create API** → name it (e.g. `trading-bot-vps`).
3. Permissions: enable **Futures** (read + trade).
   **Do not enable withdrawal.** The bot never needs it, and a leaked key that
   cannot withdraw cannot drain your account.
4. **Bind your VPS IP address.** This is the single most effective protection.
   Find it with `curl -4 ifconfig.me`.
5. Copy the Access Key and Secret Key — the secret is shown **once**.
6. Paste into `.env`:

```ini
MEXC_ACCESS_KEY=your_access_key
MEXC_SECRET_KEY=your_secret_key
```

### Protecting credentials

- `.env` is in `.gitignore`; never commit it.
- `chmod 600 .env`.
- Keys are registered as secrets with the logger at startup: they are scrubbed
  from every log line, every Telegram message and every dashboard response.
  `/api/config` returns masked values only.
- Rotate keys if a machine is ever compromised.
- Use a **separate sub-account** with only the capital you are willing to risk.

> **Operational note about MEXC futures ordering.** MEXC has, for extended
> periods, restricted *futures order placement over the API* to whitelisted
> accounts while leaving market data and read-only endpoints working. If your
> account is affected, this bot surfaces it as a clear
> `ExchangeNotSupported` error naming the cause, rather than pretending an order
> succeeded. Paper and backtest modes are unaffected. Verify your account's API
> trading status with MEXC support before relying on live mode.

---

## 7. Telegram setup

### Create the bot

1. Open Telegram, message **@BotFather**.
2. `/newbot` → choose a name and a username ending in `bot`.
3. Copy the token (looks like `123456789:AAH...`).

### Find your chat ID and user ID

Easiest: message **@userinfobot** — it replies with your user ID.

Or, after sending your new bot a message:

```bash
curl "https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates"
```

Look for `"chat":{"id":...}` (chat ID) and `"from":{"id":...}` (user ID).

### Configure

```ini
TELEGRAM_BOT_TOKEN=123456789:AAH...
TELEGRAM_CHAT_ID=987654321
TELEGRAM_ALLOWED_USER_IDS=987654321      # comma-separated for multiple operators
```

`TELEGRAM_ALLOWED_USER_IDS` is the control allowlist — **only** these user IDs
can issue commands. Anyone else gets a refusal and the attempt is logged. It is
mandatory for live mode.

---

## 8. Run in paper mode

```bash
docker compose build
docker compose up -d
docker compose logs -f bot
```

You should see the startup banner, the pre-flight report, and the first scan
within a minute. Then open the dashboard at `http://<server>:8000` (see
[section 10](#10-the-dashboard) about exposing it safely) and send `/start` to
your Telegram bot.

Try an entirely offline demo first, with no network and no credentials:

```bash
docker compose run --rm bot python scripts/run_simulation.py --cycles 2
```

---

## 9. Non-Docker installation

```bash
sudo apt install -y python3.12 python3.12-venv python3-pip git
git clone https://github.com/aldoljohn-svg/Ai-trading.git
cd Ai-trading/crypto_trading_bot

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
nano .env
chmod 600 .env

python -m app.main --check-config
python -m app.main
```

### Run as a systemd service

```bash
sudo tee /etc/systemd/system/trading-bot.service > /dev/null <<'EOF'
[Unit]
Description=Crypto Trading Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER/Ai-trading/crypto_trading_bot
Environment=TZ=UTC
ExecStart=/home/YOUR_USER/Ai-trading/crypto_trading_bot/.venv/bin/python -m app.main
Restart=on-failure
RestartSec=15
TimeoutStopSec=60
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now trading-bot
sudo systemctl status trading-bot
journalctl -u trading-bot -f
```

---

## 10. The dashboard

**URL:** `http://<your-server>:8000`

Shows equity, balance, available margin, used margin, realised and unrealised
PnL, daily PnL, drawdown, open positions with entry/current/stop/targets/risk/R,
the ranked scanner table, correlation clusters, circuit-breaker state and
component health.

**Endpoints**

| Path | Purpose |
|---|---|
| `/` | dashboard UI |
| `/health` | liveness + component health (503 when degraded) |
| `/api/snapshot` | everything in one call |
| `/api/status`, `/api/account`, `/api/positions` | state |
| `/api/scanner`, `/api/signals`, `/api/trades`, `/api/orders` | activity |
| `/api/risk`, `/api/ai`, `/api/performance`, `/api/preflight` | analysis |
| `/api/config` | configuration, **secrets masked** |
| `/api/equity` | equity curve |
| `/ws` | live push (FastAPI installs only) |
| `POST /api/control/{pause,resume,emergency}` | control |

**Security.** `docker-compose.yml` binds the port to `127.0.0.1` only. Reach it
over an SSH tunnel:

```bash
ssh -L 8000:127.0.0.1:8000 user@your-server
# then browse to http://localhost:8000
```

To expose it publicly, put a reverse proxy with TLS and authentication in front,
and set `DASHBOARD_TOKEN` so the control endpoints require a token.

---

## 11. Telegram commands

| Command | What it does |
|---|---|
| `/start`, `/menu` | control panel with buttons |
| `/status` | live report: state, equity, positions, health |
| `/pause` | stop opening new trades; open positions stay managed |
| `/resume` | resume opening new trades |
| `/stop` | stop the bot (confirmation required); positions are **left open** |
| `/emergency` | close everything and stop (confirmation required) |
| `/positions` | open positions with live PnL and per-position close buttons |
| `/orders` | recent orders |
| `/account` | balance, margin, exposure, drawdown |
| `/performance` | win rate, profit factor, expectancy |
| `/scanner` | ranked scan results |
| `/signals` | recent signals, including why trades were **rejected** |
| `/risk` | limits, current usage, correlation clusters, breakers |
| `/ai` | model status, calibration quality, recent probabilities |
| `/trades` | trade history |
| `/health` | component health |
| `/help` | this list |

The main menu mirrors these as buttons:

```
🤖 TRADING BOT
Status: 🟢 RUNNING

[📊 LIVE REPORT] [📈 MARKET SCANNER]
[💼 POSITIONS]
[▶️ START] [⏸ PAUSE]
[▶️ RESUME] [🛑 STOP]
[🚨 EMERGENCY STOP]
[💰 ACCOUNT] [📊 PERFORMANCE]
[🧠 AI] [⚙️ RISK]
```

Every trade sends a full report on entry (equity, risk %, risk amount, size,
notional, leverage, entry, stop, TP1/2/3, R:R, confidence, regime, HTF
alignment, and the reasons), plus updates on TP hits, stop moves, break-even,
trailing activation, risk reduction and close.

---

## 12. Backtesting

```bash
# Docker
docker compose run --rm bot python scripts/backtest.py \
    --symbols BTCUSDT,ETHUSDT,SOLUSDT --bars 4000

# Walk-forward validation — the one that matters
docker compose run --rm bot python scripts/backtest.py \
    --symbols BTCUSDT,ETHUSDT --walk-forward --folds 4

# Offline, no network
docker compose run --rm bot python scripts/backtest.py --source synthetic
```

**How it works.** Bar-by-bar replay on the execution timeframe using the *same*
signal engine, risk engine, sizing, stop manager, target manager and trailing
logic the live bot uses. Only the fill is simulated.

**No-lookahead guarantees:**

- A signal computed on bar *i*'s close fills at bar *i+1*'s **open**.
- Higher-timeframe series are sliced to bars that had already closed at that
  moment.
- When one bar spans both the stop and a target, the **stop is assumed to have
  been hit first** — from OHLC alone the order is unknowable, and the optimistic
  assumption is exactly what makes backtests stop matching live results.
- Warmup is automatically raised so every context timeframe has real history.
- The analysis window is capped to the same 400 bars the live scanner uses.

**Costs charged:** taker fees, spread crossing, slippage, funding, and optional
latency in bars.

**Metrics:** return, win rate, profit factor, expectancy (currency and R),
Sharpe, Sortino, Calmar, max drawdown and duration, risk of ruin, exposure,
largest win/loss, longest losing streak.

Every report prints a `sample_warning` when the trade count or curve length is
too small for the statistics to mean anything, and flags a suspiciously perfect
result as likely overfit. **Walk-forward efficiency below 0.4 means the
parameters do not generalise — do not trade that configuration.**

---

## 13. Training the model

```bash
docker compose run --rm bot python scripts/train_model.py \
    --symbols BTCUSDT,ETHUSDT,SOLUSDT --limit 6000
```

Labels use the triple-barrier method (upper barrier at +2 ATR, lower at −1 ATR,
vertical at 24 bars) with the pessimistic tie-break. Features are rebuilt bar by
bar from history only. Training/test split is chronological with an embargo gap.

The model is calibrated (Platt or isotonic, whichever measurably improves the
Brier score) and is **refused** if it fails to beat a naive baseline. A rejected
model is not a failure — the bot keeps running on rules alone, which is the safe
outcome.

The bot works fine with no model at all. When one is loaded it contributes at
most `ML_WEIGHT` (capped at 0.5) to confidence; the rules always keep the
majority say, and the model can never create a trade on its own.

---

## 14. Enabling live trading

> Do this only after weeks of paper trading you have actually reviewed.

1. Fund the MEXC **futures** wallet.
2. Confirm API keys have futures permission, no withdrawal, and IP binding.
3. Verify the deployment:

```bash
docker compose run --rm bot python scripts/healthcheck.py
```

4. Edit `.env`:

```ini
TRADING_MODE=live
DATA_SOURCE=mexc
MEXC_ACCESS_KEY=...
MEXC_SECRET_KEY=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
TELEGRAM_ALLOWED_USER_IDS=...
LIVE_CONFIRM_PHRASE=I UNDERSTAND THE RISK
```

5. Restart: `docker compose up -d --force-recreate`

**Live mode is blocked unless all of these pass at startup:**

| # | Check |
|---|---|
| 1 | `TRADING_MODE=live` explicitly set |
| 2 | `httpx` and `websockets` installed |
| 3 | MEXC REST reachable with sane latency |
| 4 | Credentials accepted by a signed request |
| 5 | Account readable with non-zero, sufficient equity |
| 6 | Market data fetched and passing validation |
| 7 | Clock within 5s of the exchange |
| 8 | Contracts discovered and tradable |
| 9 | Database writable |
| 10 | Telegram reachable |
| 11 | Existing positions reconciled cleanly |
| 12 | Risk configuration sane for the account size |

There is no override flag. Any failure leaves the bot flat and tells you exactly
what to fix, in the logs and on Telegram.

**Disabling live:** set `TRADING_MODE=paper` and `docker compose up -d
--force-recreate`. Positions already open on MEXC are **not** closed by this —
close them with `/emergency` first, or manage them manually.

---

## 15. Emergency stop & kill switch

**Emergency stop** — market-closes every position, cancels every order, stops
the bot:

- Telegram: `/emergency` → **🚨 CLOSE ALL + STOP**
- Dashboard: **🚨 Emergency** button
- API: `POST /api/control/emergency`

Both routes require explicit confirmation. Closing at market crystallises losses
immediately — that is the point, but know it before you press it.

**Kill switch** — blocks new entries instantly without restarting or touching
open positions:

```bash
docker compose exec bot touch /app/data/KILL_SWITCH     # engage
docker compose exec bot rm /app/data/KILL_SWITCH        # release
```

**Pause** — `/pause` stops new entries; open positions stay fully managed.

**Stop** — `/stop` stops the bot and **leaves positions open** on the exchange
with their stops in place. Stopping the bot is not an instruction to liquidate.

---

## 16. Day-to-day operations

```bash
# Start / stop / restart
docker compose up -d
docker compose down
docker compose restart bot

# Logs
docker compose logs -f bot            # follow
docker compose logs --tail=200 bot    # recent
tail -f logs/bot.log                  # on-disk, rotated
tail -f logs/errors.log               # warnings and errors only

# Health
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
docker compose ps                     # container health status

# Update
git pull
docker compose build
docker compose up -d
# Schema migrations (new columns) apply automatically at startup.

# Backup
./scripts/backup_db.sh                # -> backups/trading_bot_<ts>.db.gz
./scripts/backup_db.sh /mnt/backups

# Restore
docker compose down
gunzip -c backups/trading_bot_20250101-120000.db.gz > data/trading_bot.db
docker compose up -d
```

Automate backups:

```bash
crontab -e
# every 6 hours
0 */6 * * * cd /home/YOUR_USER/Ai-trading/crypto_trading_bot && ./scripts/backup_db.sh
```

---

## 17. How it decides

### Scanner — picking what to look at

One bulk ticker call ranks every contract on liquidity (40%), healthy volatility
(25%, peaking around 6% daily range and penalised when extreme), 24h movement
(20%) and spread tightness (15%). Only the top `DEEP_ANALYSIS_COUNT` symbols get
expensive multi-timeframe analysis. That pre-screen decides *who gets analysed*,
never who gets traded.

### Scoring

Eleven components, each 0–100 for a specific direction: technical, structure,
ICT, RTM, momentum, volume, volatility, HTF alignment, fundamental, R:R, plus
the ML probability. Weighted into `OPPORTUNITY_SCORE`. Execution and context
timeframes are blended so the bot neither chases a 5-minute pattern with no
context nor enters a great trend at a terrible location.

**A high score is not permission to trade.** Ranking decides attention; the gates
decide action.

### Gates — all must pass

confidence ≥ `MIN_CONFIDENCE` · R:R ≥ `MIN_RR` (scaled by regime) · liquidity ok
· spread ok · order-book depth ok · volatility inside band · market data healthy
· no price anomaly · regime permits entries · no fundamental danger · no circuit
breaker · risk budget available · portfolio and correlation limits respected ·
margin available · liquidation safely beyond the stop.

### Stop loss

Placed where the **thesis is wrong**, not at a fixed distance. Candidates, in
order of preference: beyond the swing defining the leg → beyond the order block
or RTM zone being traded from → beyond the liquidity that would be swept on
invalidation → an ATR floor. The furthest candidate that still fits inside
`MAX_STOP_PCT` wins, because a stop just inside obvious liquidity is a stop that
gets taken for no reason. Clamped to `[MIN_STOP_PCT, MAX_STOP_PCT]`.

### Take profit

Targets come from **where price is likely to travel** — unswept liquidity pools,
support/resistance clusters, opposing fair value gaps, range boundaries — sorted
by distance, de-duplicated within half an ATR, and capped at 8R. R-multiples only
fill gaps when no structural target exists. This is what makes the R:R gate
meaningful: if the only thing above us is 1.2R away, the trade is correctly
rejected.

### Position size

```
risk_amount   = equity × risk_per_trade × regime_multiplier
base_quantity = risk_amount / (stop_distance + round-trip fees)
contracts     = round_DOWN(base_quantity / contract_size)
```

Rounding is always **down**, then the realised risk is recomputed from the
rounded size and re-checked. Size is then trimmed for available margin and the
exposure cap.

### Leverage

Leverage follows the stop distance, never the other way round. Size is already
fixed by the risk budget; leverage only decides how much margin it consumes — so
a *wider* stop needs *less* leverage. The engine caps it at whatever keeps the
estimated liquidation at least 1.5× the stop distance away, then at
`MAX_LEVERAGE`, the contract's own max, and a volatility penalty.

### Risk & correlation

BTC, ETH and SOL longs are not three independent bets. Correlation is *measured*
from recent returns (defaulting to 0.6 for unknown pairs, because crypto pairs
usually are correlated). Portfolio risk is reported as `sqrt(wᵀCw)` over signed
open risks; same-direction correlated positions are additionally grouped into
clusters, and each cluster's **gross** risk is capped by
`MAX_CORRELATED_EXPOSURE`.

Once a stop reaches break-even, that position's open risk becomes zero and it
stops consuming portfolio budget.

**Never used:** martingale, loss chasing, averaging down, increasing risk after
losses, unlimited leverage. Risk is *cut* after consecutive losses and in
drawdown, never raised.

### Position management

Priority order every cycle: stop hit → emergency de-risk → invalidation exits
(structure flip, fundamental danger, data anomaly, volatility spike while
underwater) → targets → partial de-risk → stop maintenance.

TP1 banks 40% and moves the stop to break-even; TP2 banks 35% more and tightens
the trail; TP3 closes the runner. Trailing activates at 1.8R (chandelier, ATR
based) and tightens as more is banked. **A stop can only ever move in the
direction that reduces risk** — widening is structurally impossible, not merely
discouraged.

### Capital protection

Circuit breakers halt on: daily loss (until the next UTC day, and manual reset
cannot clear it), max drawdown, consecutive losses, exchange error rate, stale or
corrupt market data, reconciliation mismatch, kill-switch file, manual trip and
emergency. Degraded infrastructure is treated as a trading risk, not just an ops
problem.

### Fundamentals

Funding and open interest come from the exchange. Macro providers and news feeds
are optional and configured, not hard-coded. **Missing data is `UNKNOWN` and
`UNKNOWN` is never converted into bullish or bearish** — a fully unknown snapshot
scores exactly 50. Fundamentals can veto or shrink a trade; they can never create
one. Populate `data/economic_calendar.json` to get blackout windows around CPI,
FOMC, NFP and similar:

```json
{"events": [{"name": "US CPI", "ts": 1735689600, "impact": "high"}]}
```

---

## 18. Troubleshooting

**Bot will not start**
```bash
docker compose run --rm bot python -m app.main --check-config
docker compose logs --tail=100 bot
```
Configuration errors print the exact field and why it was rejected.

**`403 Forbidden` / `401` from MEXC** — IP not on the API allowlist (check
`curl -4 ifconfig.me`), wrong keys, or missing futures permission.

**`Signature verification failed` / timestamp errors** — clock drift.
`sudo timedatectl set-ntp true`.

**`ExchangeNotSupported` on order placement** — MEXC has restricted futures API
trading for that account. See the note in [section 6](#6-mexc-api-setup).

**No trades for hours or days** — usually correct behaviour. Check `/signals` and
`/scanner` for the actual rejection reasons. Common ones: HTF context in
`CONFLICT`, R:R below the regime-scaled minimum, confidence under 70%. Lower
`MIN_CONFIDENCE`/`MIN_RR` only if you have backtested that change.

**`risk budget only affords 0 contracts`** — the account is too small for that
symbol's minimum contract at 0.5% risk. Trade cheaper symbols or increase equity.

**Telegram silent** — verify the token, send the bot a message first, and check
that your user ID is in `TELEGRAM_ALLOWED_USER_IDS`.
```bash
curl "https://api.telegram.org/bot<TOKEN>/getMe"
```

**Dashboard unreachable** — it binds to `127.0.0.1` by design. Use the SSH tunnel
in [section 10](#10-the-dashboard).

**Reconciliation mismatch** — the bot pauses new entries and alerts. Compare
`/positions` against the MEXC app, resolve manually, then `/resume`.

**Container restarting** — `docker compose logs --tail=200 bot`; check disk with
`df -h` and memory with `free -m`.

**Verify everything at once**
```bash
docker compose run --rm bot python scripts/healthcheck.py
```

---

## 19. Security checklist

- [ ] `.env` has `chmod 600` and is never committed (`.gitignore` covers it)
- [ ] MEXC key has **futures only**, **no withdrawal**
- [ ] MEXC key is **bound to the VPS IP**
- [ ] Using a sub-account with limited capital
- [ ] `TELEGRAM_ALLOWED_USER_IDS` set (mandatory for live)
- [ ] Dashboard not publicly exposed, or behind TLS + auth with `DASHBOARD_TOKEN`
- [ ] SSH key auth only, password login disabled, firewall (`ufw`) enabled
- [ ] Automated database backups running
- [ ] NTP enabled, server on UTC
- [ ] Reviewed paper results before enabling live
- [ ] Know how to trigger `/emergency` from your phone

Secrets never reach logs, Telegram or the dashboard: keys are registered with the
log filter at startup, and secret-shaped strings (`apiKey=`, JSON `"secret"`,
bot-token URLs, `Signature:`) are scrubbed by pattern even if they were never
registered.

---

## 20. Project layout

```
crypto_trading_bot/
├── app/
│   ├── main.py                  entry point: engine + Telegram + dashboard
│   ├── engine.py                orchestrator, loops, crash recovery, views
│   ├── config.py                loading + strict validation + redaction
│   ├── logger.py                logging with mandatory secret scrubbing
│   ├── domain.py                shared types (Candle, Side, Regime, …)
│   ├── compat.py                optional-dependency matrix
│   ├── exchange/                base interface, MEXC adapter, synthetic feed
│   ├── data/                    market data facade, cache, validators, websocket
│   ├── indicators/              RSI, ATR, MA40/80/160 + slopes, MACD, ADX, …
│   ├── market_structure/        swings, HH/HL/LH/LL, BOS/CHOCH/MSS, levels
│   ├── ict/                     FVG, order blocks, liquidity, sweeps, premium
│   ├── rtm/                     bases, legs, RBR/DBD/RBD/DBR, compression
│   ├── regime/                  regime detection + per-regime playbooks
│   ├── fundamental/             macro, news/calendar, aggregation
│   ├── scanner/                 universe, deep analysis, ranking
│   ├── signals/                 scoring, signal engine, trade proposal
│   ├── ml/                      features, dataset, train, calibrate, predict
│   ├── risk/                    sizing, portfolio risk, breakers, risk engine
│   ├── portfolio/               positions and account state
│   ├── execution/               order manager, execution engine, reconciliation
│   ├── position_manager/        stops, targets, trailing, live management
│   ├── backtest/                engine, metrics, walk-forward
│   ├── paper/                   paper broker
│   ├── telegram/                bot, commands, keyboards, notifications
│   ├── dashboard/               API, websocket, stdlib fallback, frontend
│   ├── database/                schema, DB-API layer, repositories
│   └── health/                  health monitor, live pre-flight
├── tests/                       344 tests
├── scripts/                     backtest, train, simulate, healthcheck, backup
├── data/  logs/  models/
├── .env.example   config.yaml   requirements.txt
├── Dockerfile     docker-compose.yml
└── README.md
```

Run the tests:

```bash
pip install -r requirements-dev.txt
pytest -q
```

---

## 21. Known limitations

Stated plainly, because you are trusting this with money:

- **No edge is proven.** The repository ships a strategy and the machinery to
  evaluate it. Whether it is profitable on real MEXC data over your chosen
  symbols is an open question you must answer with walk-forward testing.
- **Backtest results included in development used synthetic data** from the
  built-in simulator. Those numbers describe the simulator, not the market.
- **Slippage and funding are modelled, not measured.** Real fills in fast markets
  can be materially worse than the configured assumptions.
- **MEXC may block futures API ordering** on your account (see section 6).
- **Keyword news classification is crude.** It can block or shrink a trade; it is
  deliberately never allowed to create one.
- **Correlation is estimated from recent returns** and shifts in a crisis —
  usually toward 1.0, exactly when diversification is most needed.
- **The liquidation estimate is approximate.** MEXC's tiered maintenance-margin
  schedule is authoritative; the built-in estimate is deliberately conservative.
- **`sqrt(wᵀCw)` measures portfolio risk as volatility.** If every stop is hit
  simultaneously the realised loss is the gross sum, which is why the cluster cap
  uses gross risk.
- **Single-process, single-account.** No multi-account or multi-exchange support.

---

## Licence & disclaimer

Provided as-is, with no warranty of any kind. Nothing here is financial advice.
Trading leveraged cryptocurrency derivatives carries a high risk of losing your
entire deposit. You are solely responsible for any use of this software and for
complying with the laws and exchange terms that apply to you.

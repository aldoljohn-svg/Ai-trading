# Ai-trading

Autonomous cryptocurrency trading system for MEXC Futures.

The project lives in **[`crypto_trading_bot/`](crypto_trading_bot/)** — see its
[README](crypto_trading_bot/README.md) for full installation, configuration and
operating instructions.

```bash
cd crypto_trading_bot
cp .env.example .env      # defaults are safe: PAPER mode, no credentials needed
docker compose up -d
docker compose logs -f bot
```

Defaults to **paper trading**. Live trading requires `TRADING_MODE=live` plus
passing twelve pre-flight checks at startup.

> This software cannot predict the market and does not guarantee profit. Trading
> leveraged derivatives can lose your entire deposit.

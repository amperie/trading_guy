from pathlib import Path
from textwrap import dedent

import nbformat as nbf


NOTEBOOK = Path("notebooks/alpaca_market_data_archive.ipynb")


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


nb = nbf.v4.new_notebook()
nb["cells"] = [
    md(
        """
        ## Goal

        Download a recoverable local archive of Alpaca market data before shutting the account down.

        This notebook writes one CSV per object and granularity under the configured output directory.
        It writes each page atomically, records per-file state in sidecar JSON files, and resumes from
        the newest timestamp already written when rerun. If `END` advances in the future, completed files
        are updated from their latest local timestamp.
        """
    ),
    md(
        """
        ## Setup

        Put Alpaca credentials in one of these places:

        - `accounts.yaml` under `AlpacaAPIAccount.api_key` and `AlpacaAPIAccount.secret_key`
        - environment variables: `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`
        - `.env` with the same environment variable names

        Current Alpaca docs checked while creating this notebook:

        - Market Data API page: equities/options historical API calls are listed as `200 / min` on Basic and `10,000 / min` on Algo Trader Plus, with stock/ETF history shown as since 2016.
        - Rate-limit responses include `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset`; HTTP 429 should be retried with backoff.
        - Bars endpoints accept `[1-59]Min`, so `5Min` is valid for stocks, crypto, and options bars.
        - Forex rates support `5Sec`, `1Min`, and `1Day`, not `5Min`; this notebook uses `1Min` for intraday FX so you can resample to 5-minute locally.
        - Options historical data starts around February 2024.
        - News historical data dates back to 2015.
        """
    ),
    md(
        """
        ## Download Universe

        The first code cell below is the control panel for the archive. The default universe is split into practical research groups:

        - `leveraged_etfs`: daily-reset leveraged equity funds across broad indexes and major sectors. These are useful for studying volatility decay, crash path dependency, and high-beta trend or mean-reversion behavior.
        - `funds`: non-leveraged ETFs covering broad U.S. indexes, international equity, bonds, credit, commodities, REITs, and sector exposures. These give cleaner baseline regimes and cross-asset context.
        - `stocks`: liquid large-cap single names across technology, financials, energy, healthcare, consumer, industrials, utilities, and real estate. These help test whether strategies generalize beyond index products.
        - `equity_symbols`: the deduped union of leveraged ETFs, funds, and stocks. The daily stock/ETF tier uses this list.
        - `core_intraday_equities`: a smaller subset for `5Min` bars. Five-minute bars get large quickly, so this keeps intraday history focused on the instruments most likely to matter for backtests.
        - `crypto_symbols`: major crypto pairs. These add 24/7 trading, weekend behavior, exchange-specific liquidity, and crash regimes that do not line up cleanly with equities.
        - `forex_pairs`: major and selected risk-sensitive currency pairs. Alpaca's forex endpoint supports `1Min` and `1Day`, so intraday FX is downloaded as `1Min` and can be resampled to `5Min`.
        - `option_underlyings`: liquid names used for optional contract discovery and option-bar downloads. This tier is disabled by default because contracts can explode into thousands of files and many API calls.
        - `news_symbols`: equity symbols plus crypto symbols converted to Alpaca's news format, used for historical news pulls.

        Change `OUTPUT_DIR`, `BACKUP_OUTPUT_DIR`, the symbol lists, or the `DOWNLOAD_PLAN` entries before running the download cells. Setting a tier's `enabled` value to `False` skips that tier.
        """
    ),
    md(
        """
        ## Symbol Reference

        ### Leveraged ETFs

        | Symbol | What it is |
        |---|---|
        | `TQQQ` | ProShares UltraPro QQQ; 3x daily Nasdaq-100 exposure. |
        | `QLD` | ProShares Ultra QQQ; 2x daily Nasdaq-100 exposure. |
        | `UPRO` | ProShares UltraPro S&P500; 3x daily S&P 500 exposure. |
        | `SSO` | ProShares Ultra S&P500; 2x daily S&P 500 exposure. |
        | `SPXL` | Direxion Daily S&P 500 Bull 3X Shares; 3x daily S&P 500 exposure. |
        | `UDOW` | ProShares UltraPro Dow30; 3x daily Dow Jones Industrial Average exposure. |
        | `DDM` | ProShares Ultra Dow30; 2x daily Dow Jones Industrial Average exposure. |
        | `TNA` | Direxion Daily Small Cap Bull 3X Shares; 3x daily Russell 2000-style small-cap exposure. |
        | `UWM` | ProShares Ultra Russell2000; 2x daily Russell 2000 exposure. |
        | `SOXL` | Direxion Daily Semiconductor Bull 3X Shares; 3x daily semiconductor-sector exposure. |
        | `TECL` | Direxion Daily Technology Bull 3X Shares; 3x daily technology-sector exposure. |
        | `FAS` | Direxion Daily Financial Bull 3X Shares; 3x daily financial-sector exposure. |
        | `ERX` | Direxion Daily Energy Bull ETF; leveraged daily energy-sector exposure. |
        | `DRN` | Direxion Daily Real Estate Bull 3X Shares; 3x daily real-estate-sector exposure. |
        | `CURE` | Direxion Daily Healthcare Bull 3X Shares; 3x daily healthcare-sector exposure. |
        | `LABU` | Direxion Daily S&P Biotech Bull 3X Shares; 3x daily biotech exposure. |
        | `YINN` | Direxion Daily FTSE China Bull 3X Shares; 3x daily China large-cap equity exposure. |
        | `URTY` | ProShares UltraPro Russell2000; 3x daily Russell 2000 exposure. |

        ### Funds And ETFs

        | Symbol | What it is |
        |---|---|
        | `SPY` | SPDR S&P 500 ETF Trust; large-cap U.S. equity benchmark. |
        | `VOO` | Vanguard S&P 500 ETF; low-cost S&P 500 exposure. |
        | `IVV` | iShares Core S&P 500 ETF; low-cost S&P 500 exposure. |
        | `VTI` | Vanguard Total Stock Market ETF; broad U.S. equity market exposure. |
        | `QQQ` | Invesco QQQ Trust; Nasdaq-100 / large-cap growth-heavy exposure. |
        | `IWM` | iShares Russell 2000 ETF; U.S. small-cap exposure. |
        | `DIA` | SPDR Dow Jones Industrial Average ETF Trust; Dow 30 exposure. |
        | `EFA` | iShares MSCI EAFE ETF; developed international equity exposure. |
        | `EEM` | iShares MSCI Emerging Markets ETF; emerging-market equity exposure. |
        | `VEA` | Vanguard FTSE Developed Markets ETF; developed ex-U.S. equity exposure. |
        | `TLT` | iShares 20+ Year Treasury Bond ETF; long-duration U.S. Treasury exposure. |
        | `IEF` | iShares 7-10 Year Treasury Bond ETF; intermediate Treasury exposure. |
        | `SHY` | iShares 1-3 Year Treasury Bond ETF; short-duration Treasury exposure. |
        | `BND` | Vanguard Total Bond Market ETF; broad U.S. bond exposure. |
        | `HYG` | iShares iBoxx High Yield Corporate Bond ETF; high-yield credit exposure. |
        | `LQD` | iShares iBoxx Investment Grade Corporate Bond ETF; investment-grade credit exposure. |
        | `GLD` | SPDR Gold Shares; gold bullion exposure. |
        | `SLV` | iShares Silver Trust; silver bullion exposure. |
        | `VNQ` | Vanguard Real Estate ETF; U.S. REIT exposure. |
        | `XLF` | Financial Select Sector SPDR Fund; financial-sector exposure. |
        | `XLK` | Technology Select Sector SPDR Fund; technology-sector exposure. |
        | `XLE` | Energy Select Sector SPDR Fund; energy-sector exposure. |
        | `XLV` | Health Care Select Sector SPDR Fund; healthcare-sector exposure. |
        | `XLI` | Industrial Select Sector SPDR Fund; industrial-sector exposure. |
        | `XLP` | Consumer Staples Select Sector SPDR Fund; defensive consumer-staples exposure. |
        | `XLY` | Consumer Discretionary Select Sector SPDR Fund; cyclical consumer exposure. |
        | `XLU` | Utilities Select Sector SPDR Fund; utilities-sector exposure. |
        | `XLB` | Materials Select Sector SPDR Fund; materials-sector exposure. |
        | `XLRE` | Real Estate Select Sector SPDR Fund; real-estate-sector exposure. |

        ### Stocks

        | Symbol | What it is |
        |---|---|
        | `AAPL` | Apple; mega-cap consumer hardware, services, and platform ecosystem. |
        | `MSFT` | Microsoft; software, cloud infrastructure, enterprise technology. |
        | `NVDA` | NVIDIA; GPUs, AI accelerators, semiconductor cycle exposure. |
        | `AMZN` | Amazon; e-commerce, cloud infrastructure, logistics, digital advertising. |
        | `GOOGL` | Alphabet Class A; search, ads, YouTube, cloud, AI. |
        | `META` | Meta Platforms; social media, digital ads, AI infrastructure. |
        | `TSLA` | Tesla; electric vehicles, batteries, high-beta retail/speculative flow. |
        | `AVGO` | Broadcom; semiconductors, infrastructure software, networking chips. |
        | `AMD` | Advanced Micro Devices; CPUs, GPUs, data-center semiconductor exposure. |
        | `JPM` | JPMorgan Chase; large U.S. bank and credit-cycle proxy. |
        | `BAC` | Bank of America; large U.S. bank, rates and credit sensitivity. |
        | `GS` | Goldman Sachs; investment banking, trading, capital markets. |
        | `XOM` | Exxon Mobil; integrated oil and gas major. |
        | `CVX` | Chevron; integrated oil and gas major. |
        | `UNH` | UnitedHealth Group; managed care and healthcare services. |
        | `JNJ` | Johnson & Johnson; diversified healthcare and pharmaceuticals. |
        | `LLY` | Eli Lilly; large-cap pharmaceuticals, obesity/diabetes drug exposure. |
        | `PFE` | Pfizer; large-cap pharmaceuticals. |
        | `PG` | Procter & Gamble; defensive consumer staples. |
        | `KO` | Coca-Cola; defensive global beverage staples. |
        | `COST` | Costco; defensive retail and consumer spending proxy. |
        | `WMT` | Walmart; defensive retail, grocery, and low-end consumer proxy. |
        | `HD` | Home Depot; housing, home improvement, consumer cyclicals. |
        | `CAT` | Caterpillar; industrials, construction, mining, global cyclicals. |
        | `DE` | Deere; agriculture machinery and industrial cyclicals. |
        | `BA` | Boeing; aerospace, defense-adjacent industrial, event-risk exposure. |
        | `GE` | GE Aerospace; industrial aerospace and engine exposure. |
        | `NEE` | NextEra Energy; regulated utility and renewables exposure. |
        | `DUK` | Duke Energy; regulated utility exposure. |
        | `PLD` | Prologis; logistics real estate and industrial REIT exposure. |

        ### Crypto

        | Symbol | What it is |
        |---|---|
        | `BTC/USD` | Bitcoin versus U.S. dollar; dominant crypto store-of-value/liquidity proxy. |
        | `ETH/USD` | Ethereum versus U.S. dollar; smart-contract platform and crypto beta proxy. |
        | `SOL/USD` | Solana versus U.S. dollar; high-throughput layer-1 crypto exposure. |
        | `AVAX/USD` | Avalanche versus U.S. dollar; layer-1 smart-contract exposure. |
        | `LINK/USD` | Chainlink versus U.S. dollar; oracle-network token exposure. |
        | `DOGE/USD` | Dogecoin versus U.S. dollar; meme/retail-flow crypto proxy. |
        | `LTC/USD` | Litecoin versus U.S. dollar; older payment-focused crypto exposure. |
        | `BCH/USD` | Bitcoin Cash versus U.S. dollar; Bitcoin fork/payment-chain exposure. |
        | `UNI/USD` | Uniswap versus U.S. dollar; decentralized-exchange token exposure. |
        | `AAVE/USD` | Aave versus U.S. dollar; decentralized lending token exposure. |
        | `DOT/USD` | Polkadot versus U.S. dollar; interoperability/layer-0 crypto exposure. |
        | `XRP/USD` | XRP versus U.S. dollar; payments-focused crypto exposure. |

        ### Forex

        | Symbol | What it is |
        |---|---|
        | `EURUSD` | Euro versus U.S. dollar; most liquid developed-market FX pair. |
        | `GBPUSD` | British pound versus U.S. dollar; developed-market FX and U.K. macro exposure. |
        | `USDJPY` | U.S. dollar versus Japanese yen; rates, carry, and risk-off sensitivity. |
        | `USDCHF` | U.S. dollar versus Swiss franc; developed-market safe-haven FX pair. |
        | `USDCAD` | U.S. dollar versus Canadian dollar; North America and oil-sensitive FX pair. |
        | `AUDUSD` | Australian dollar versus U.S. dollar; commodity and China-growth sensitivity. |
        | `NZDUSD` | New Zealand dollar versus U.S. dollar; commodity/risk-sensitive FX pair. |
        | `USDMXN` | U.S. dollar versus Mexican peso; liquid emerging-market carry/risk pair. |
        | `USDZAR` | U.S. dollar versus South African rand; high-volatility emerging-market FX pair. |
        | `USDNOK` | U.S. dollar versus Norwegian krone; oil-sensitive developed-market FX pair. |
        """
    ),
    code(
        """
        from __future__ import annotations

        import json
        import os
        import random
        import re
        import shutil
        import time
        from datetime import datetime, timedelta, timezone
        from pathlib import Path
        from typing import Any

        import pandas as pd
        import requests
        import yaml

        DATA_BASE_URL = "https://data.alpaca.markets"
        TRADING_BASE_URL = "https://paper-api.alpaca.markets"

        # ===== User controls =====
        PROJECT_ROOT = next(
            (path for path in [Path.cwd(), *Path.cwd().parents] if (path / "pyproject.toml").exists()),
            Path.cwd(),
        )
        ALPACA_ACCOUNT_NAME = "AlpacaAPIAccount"
        OUTPUT_DIR = PROJECT_ROOT / "data/alpaca_archive"
        BACKUP_OUTPUT_DIR = PROJECT_ROOT / "data/alpaca_archive_backup"  # Set to None to disable backup copies.

        STARTS = {
            "stocks": "2016-01-01",        # Alpaca docs currently list stock/ETF history as since 2016.
            "crypto": "2016-01-01",        # The API will naturally return only available history.
            "forex": "2016-01-01",
            "options": "2024-02-01",       # Alpaca options history starts around Feb 2024.
            "news": "2015-01-01",          # Alpaca news docs say Benzinga history dates back to 2015.
            "corporate_actions": "2016-01-01",
        }
        END = datetime.now(timezone.utc).date().isoformat()

        leveraged_etfs = [
            "TQQQ", "QLD", "UPRO", "SSO", "SPXL", "UDOW", "DDM", "TNA", "UWM",
            "SOXL", "TECL", "FAS", "ERX", "DRN", "CURE", "LABU", "YINN", "URTY",
        ]

        funds = [
            "SPY", "VOO", "IVV", "VTI", "QQQ", "IWM", "DIA", "EFA", "EEM", "VEA",
            "TLT", "IEF", "SHY", "BND", "HYG", "LQD", "GLD", "SLV", "VNQ",
            "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE",
        ]

        stocks = [
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD",
            "JPM", "BAC", "GS", "XOM", "CVX", "UNH", "JNJ", "LLY", "PFE", "PG",
            "KO", "COST", "WMT", "HD", "CAT", "DE", "BA", "GE", "NEE", "DUK", "PLD",
        ]

        equity_symbols = sorted(set(leveraged_etfs + funds + stocks))

        core_intraday_equities = sorted(set([
            "SPY", "QQQ", "IWM", "TLT", "GLD", "TQQQ", "QLD", "UPRO", "SSO", "SPXL",
            "SOXL", "TECL", "TNA", "UWM", "FAS", "ERX", "AAPL", "MSFT", "NVDA",
            "TSLA", "AMZN", "GOOGL", "META", "JPM", "XOM", "UNH", "CAT",
        ]))

        crypto_symbols = [
            "BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD", "DOGE/USD",
            "LTC/USD", "BCH/USD", "UNI/USD", "AAVE/USD", "DOT/USD", "XRP/USD",
        ]

        forex_pairs = [
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
            "USDMXN", "USDZAR", "USDNOK",
        ]

        option_underlyings = ["SPY", "QQQ", "IWM", "TLT", "GLD", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]
        news_symbols = sorted(set(equity_symbols + [s.replace("/", "") for s in crypto_symbols]))

        DOWNLOAD_PLAN = {
            "stocks_daily": {"enabled": True, "symbols": equity_symbols, "timeframe": "1Day"},
            "stocks_5min": {"enabled": True, "symbols": core_intraday_equities, "timeframe": "5Min"},
            "crypto_daily": {"enabled": True, "symbols": crypto_symbols, "timeframe": "1Day"},
            "crypto_5min": {"enabled": True, "symbols": crypto_symbols, "timeframe": "5Min"},
            "forex_daily": {"enabled": True, "symbols": forex_pairs, "timeframe": "1Day"},
            "forex_1min": {"enabled": True, "symbols": forex_pairs, "timeframe": "1Min"},
            "corporate_actions": {"enabled": True, "symbols": equity_symbols},
            "news": {"enabled": True, "symbols": news_symbols},
            "options_5min": {"enabled": False, "symbols": option_underlyings, "timeframe": "5Min"},
        }

        # Set to True only if you really want option contract discovery + bars. This can explode into
        # a very large number of files/calls even with the filters below.
        DOWNLOAD_OPTIONS = bool(DOWNLOAD_PLAN["options_5min"]["enabled"])

        # If options are enabled, this limits contract discovery to a practical research subset.
        OPTION_EXPIRATION_DAYS_AHEAD = 120
        OPTION_STRIKE_BAND_PCT = 0.30

        REQUEST_LIMIT_PER_PAGE = 10_000
        NEWS_LIMIT_PER_PAGE = 50
        MIN_SECONDS_BETWEEN_CALLS = 0.35  # Conservative for 200/min plans.
        MAX_RETRIES = 8
        """
    ),
    md("## Helpers"),
    code(
        r"""
        def mask_secret(value: str | None) -> str:
            if not value:
                return "<missing>"
            return f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "****"


        def load_dotenv(path: Path | None = None) -> None:
            path = path or PROJECT_ROOT / ".env"
            if not path.exists():
                return
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


        def alpaca_headers() -> dict[str, str]:
            load_dotenv()
            account_path = PROJECT_ROOT / "accounts.yaml"
            if account_path.exists():
                accounts = yaml.safe_load(account_path.read_text(encoding="utf-8")) or {}
                account = accounts.get(ALPACA_ACCOUNT_NAME) or {}
                api_key = account.get("api_key")
                secret_key = account.get("secret_key")
                if api_key and secret_key:
                    print(f"Using Alpaca credentials from accounts.yaml:{ALPACA_ACCOUNT_NAME} ({mask_secret(api_key)})")
                    return {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}

            api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
            secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
            if not api_key or not secret_key:
                raise RuntimeError(
                    "Missing Alpaca credentials. Checked "
                    f"{account_path}, {PROJECT_ROOT / '.env'}, and environment variables "
                    "ALPACA_API_KEY/ALPACA_SECRET_KEY or APCA_API_KEY_ID/APCA_API_SECRET_KEY."
                )
            print(f"Using Alpaca credentials from environment/.env ({mask_secret(api_key)})")
            return {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}


        HEADERS = alpaca_headers()


        def safe_name(value: str) -> str:
            return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


        def paths(dataset: str, object_name: str, granularity: str) -> tuple[Path, Path]:
            base = OUTPUT_DIR / dataset / safe_name(granularity)
            base.mkdir(parents=True, exist_ok=True)
            stem = safe_name(object_name)
            return base / f"{stem}.csv", base / f"{stem}.state.json"


        def backup_path(path: Path) -> Path | None:
            if BACKUP_OUTPUT_DIR is None:
                return None
            return BACKUP_OUTPUT_DIR / path.relative_to(OUTPUT_DIR)


        def mirror_to_backup(path: Path) -> None:
            backup = backup_path(path)
            if backup is None or not path.exists():
                return
            backup.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = backup.with_suffix(backup.suffix + ".tmp")
            shutil.copy2(path, tmp_path)
            os.replace(tmp_path, backup)


        def restore_missing_from_backup(path: Path) -> None:
            backup = backup_path(path)
            if path.exists() or backup is None or not backup.exists():
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, path)


        def read_state(state_path: Path) -> dict[str, Any]:
            restore_missing_from_backup(state_path)
            if not state_path.exists():
                return {}
            try:
                return json.loads(state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {"status": "bad_state_file"}


        def write_state(state_path: Path, state: dict[str, Any]) -> None:
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
            mirror_to_backup(state_path)


        def latest_timestamp(csv_path: Path) -> str | None:
            restore_missing_from_backup(csv_path)
            if not csv_path.exists() or csv_path.stat().st_size == 0:
                return None
            try:
                ts = pd.read_csv(csv_path, usecols=["timestamp"])["timestamp"]
            except Exception:
                return None
            if ts.empty:
                return None
            return str(pd.to_datetime(ts, utc=True).max().isoformat().replace("+00:00", "Z"))


        def normalize_resume_timestamp(value: str | None) -> str | None:
            if not value:
                return None
            return str(pd.to_datetime(value, utc=True).isoformat().replace("+00:00", "Z"))


        def request_already_complete(state: dict[str, Any], params: dict[str, Any]) -> bool:
            if state.get("status") != "complete":
                return False
            requested_end = state.get("requested_end")
            requested_start = state.get("requested_start")
            return bool(
                requested_end
                and requested_start
                and str(requested_end) >= str(params.get("end", ""))
                and str(requested_start) <= str(params.get("start", ""))
            )


        def row_sort_columns(df: pd.DataFrame) -> list[str]:
            return [col for col in ["timestamp", "symbol", "object", "id", "process_date", "updated_at"] if col in df.columns]


        def dedupe_columns(df: pd.DataFrame) -> list[str]:
            for cols in (
                ["id"],
                ["timestamp", "symbol"],
                ["timestamp", "object"],
                ["process_date", "symbol", "type"],
                ["updated_at", "headline", "source"],
            ):
                if all(col in df.columns for col in cols):
                    return cols
            return list(df.columns)


        def append_rows(csv_path: Path, rows: list[dict[str, Any]]) -> int:
            if not rows:
                return 0
            existing = pd.DataFrame()
            if csv_path.exists() and csv_path.stat().st_size > 0:
                existing = pd.read_csv(csv_path, on_bad_lines="skip")

            before = len(existing)
            combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
            combined = combined.drop_duplicates(subset=dedupe_columns(combined), keep="last")
            sort_cols = row_sort_columns(combined)
            if sort_cols:
                combined = combined.sort_values(sort_cols, kind="stable")

            tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
            combined.to_csv(tmp_path, index=False)
            os.replace(tmp_path, csv_path)
            mirror_to_backup(csv_path)
            return max(0, len(combined) - before)


        def normalize_records(payload: Any, data_key: str, object_name: str | None = None) -> list[dict[str, Any]]:
            data = payload.get(data_key, []) if isinstance(payload, dict) else payload
            if isinstance(data, dict):
                records = []
                for symbol, values in data.items():
                    if isinstance(values, dict):
                        values = [values]
                    for item in values or []:
                        row = dict(item)
                        row.setdefault("symbol", symbol)
                        records.append(row)
            elif isinstance(data, list):
                records = [dict(item) for item in data]
            else:
                records = []

            for row in records:
                if "t" in row and "timestamp" not in row:
                    row["timestamp"] = row.pop("t")
                if "timestamp" not in row:
                    for candidate in ("updated_at", "created_at", "process_date", "date"):
                        if candidate in row:
                            row["timestamp"] = row[candidate]
                            break
                if object_name:
                    row.setdefault("object", object_name)
            return records


        def request_json(url: str, params: dict[str, Any]) -> requests.Response:
            delay = 1.0
            for attempt in range(1, MAX_RETRIES + 1):
                response = requests.get(url, headers=HEADERS, params=params, timeout=60)
                remaining = response.headers.get("X-RateLimit-Remaining")
                reset = response.headers.get("X-RateLimit-Reset")
                if response.status_code == 429:
                    wait = delay + random.uniform(0, 0.75)
                    if reset and reset.isdigit():
                        wait = max(wait, int(reset) - int(time.time()) + 1)
                    print(f"rate limited; sleeping {wait:.1f}s")
                    time.sleep(wait)
                    delay = min(delay * 2, 90)
                    continue
                if response.status_code in {500, 502, 503, 504}:
                    wait = delay + random.uniform(0, 0.75)
                    print(f"server error {response.status_code}; sleeping {wait:.1f}s")
                    time.sleep(wait)
                    delay = min(delay * 2, 90)
                    continue
                if remaining is not None and remaining.isdigit() and int(remaining) <= 2:
                    wait = 2.0
                    if reset and reset.isdigit():
                        wait = max(wait, int(reset) - int(time.time()) + 1)
                    print(f"near rate limit; sleeping {wait:.1f}s")
                    time.sleep(wait)
                response.raise_for_status()
                time.sleep(MIN_SECONDS_BETWEEN_CALLS)
                return response
            response.raise_for_status()
            return response


        def fetch_paged(
            *,
            dataset: str,
            object_name: str,
            granularity: str,
            url: str,
            params: dict[str, Any],
            data_key: str,
            complete_when_empty: bool = True,
        ) -> None:
            csv_path, state_path = paths(dataset, object_name, granularity)
            state = read_state(state_path)
            target_start = params.get("start")
            target_end = params.get("end")
            if request_already_complete(state, params):
                print(f"skip complete: {csv_path}")
                return

            resume_start = latest_timestamp(csv_path)
            if resume_start and "start" in params:
                params = dict(params)
                params["start"] = resume_start

            total_added = 0
            page_token = None
            try:
                while True:
                    page_params = dict(params)
                    if page_token:
                        page_params["page_token"] = page_token
                    response = request_json(url, page_params)
                    payload = response.json()
                    rows = normalize_records(payload, data_key, object_name)
                    rows = [
                        row for row in rows
                        if normalize_resume_timestamp(row.get("timestamp")) != resume_start
                    ]
                    added = append_rows(csv_path, rows)
                    total_added += added
                    page_token = payload.get("next_page_token")
                    write_state(state_path, {
                        "status": "running",
                        "dataset": dataset,
                        "object": object_name,
                        "granularity": granularity,
                        "rows_added_this_run": total_added,
                        "last_http_status": response.status_code,
                        "next_page_token": page_token,
                        "requested_start": target_start,
                        "requested_end": target_end,
                    })
                    print(f"{dataset} {object_name} {granularity}: +{added} rows")
                    if not page_token:
                        if rows or complete_when_empty:
                            write_state(state_path, {
                                "status": "complete",
                                "dataset": dataset,
                                "object": object_name,
                                "granularity": granularity,
                                "rows_added_this_run": total_added,
                                "requested_start": target_start,
                                "requested_end": target_end,
                            })
                        break
            except Exception as exc:
                write_state(state_path, {
                    "status": "error",
                    "dataset": dataset,
                    "object": object_name,
                    "granularity": granularity,
                    "rows_added_this_run": total_added,
                    "error": repr(exc),
                })
                print(f"ERROR {dataset} {object_name} {granularity}: {exc}")
        """
    ),
    md("## Download Stock And ETF Bars"),
    code(
        """
        def download_stock_bars(symbols: list[str], timeframe: str, start: str, end: str, feed: str = "sip") -> None:
            for symbol in symbols:
                fetch_paged(
                    dataset="stocks",
                    object_name=symbol,
                    granularity=timeframe,
                    url=f"{DATA_BASE_URL}/v2/stocks/{symbol}/bars",
                    params={
                        "timeframe": timeframe,
                        "start": start,
                        "end": end,
                        "limit": REQUEST_LIMIT_PER_PAGE,
                        "adjustment": "all",
                        "feed": feed,
                        "sort": "asc",
                    },
                    data_key="bars",
                )


        plan = DOWNLOAD_PLAN["stocks_daily"]
        if plan["enabled"]:
            download_stock_bars(plan["symbols"], plan["timeframe"], STARTS["stocks"], END)

        plan = DOWNLOAD_PLAN["stocks_5min"]
        if plan["enabled"]:
            download_stock_bars(plan["symbols"], plan["timeframe"], STARTS["stocks"], END)
        """
    ),
    md("## Download Crypto Bars"),
    code(
        """
        def download_crypto_bars(symbols: list[str], timeframe: str, start: str, end: str, loc: str = "us") -> None:
            for symbol in symbols:
                fetch_paged(
                    dataset="crypto",
                    object_name=symbol,
                    granularity=timeframe,
                    url=f"{DATA_BASE_URL}/v1beta3/crypto/{loc}/bars",
                    params={
                        "symbols": symbol,
                        "timeframe": timeframe,
                        "start": start,
                        "end": end,
                        "limit": REQUEST_LIMIT_PER_PAGE,
                        "sort": "asc",
                    },
                    data_key="bars",
                )


        plan = DOWNLOAD_PLAN["crypto_daily"]
        if plan["enabled"]:
            download_crypto_bars(plan["symbols"], plan["timeframe"], STARTS["crypto"], END)

        plan = DOWNLOAD_PLAN["crypto_5min"]
        if plan["enabled"]:
            download_crypto_bars(plan["symbols"], plan["timeframe"], STARTS["crypto"], END)
        """
    ),
    md("## Download Forex Rates"),
    code(
        """
        def download_forex_rates(pairs: list[str], timeframe: str, start: str, end: str) -> None:
            for pair in pairs:
                fetch_paged(
                    dataset="forex",
                    object_name=pair,
                    granularity=timeframe,
                    url=f"{DATA_BASE_URL}/v1beta1/forex/rates",
                    params={
                        "currency_pairs": pair,
                        "timeframe": timeframe,
                        "start": start,
                        "end": end,
                        "limit": REQUEST_LIMIT_PER_PAGE,
                    },
                    data_key="rates",
                )


        plan = DOWNLOAD_PLAN["forex_daily"]
        if plan["enabled"]:
            download_forex_rates(plan["symbols"], plan["timeframe"], STARTS["forex"], END)

        plan = DOWNLOAD_PLAN["forex_1min"]
        if plan["enabled"]:
            download_forex_rates(plan["symbols"], plan["timeframe"], STARTS["forex"], END)
        """
    ),
    md("## Download Corporate Actions"),
    code(
        """
        plan = DOWNLOAD_PLAN["corporate_actions"]
        if plan["enabled"]:
            for symbol in plan["symbols"]:
                fetch_paged(
                    dataset="corporate_actions",
                    object_name=symbol,
                    granularity="events",
                    url=f"{DATA_BASE_URL}/v1/corporate-actions",
                    params={
                        "symbols": symbol,
                        "start": STARTS["corporate_actions"],
                        "end": END,
                        "data_quality": "complete",
                    },
                    data_key="corporate_actions",
                )
        """
    ),
    md("## Download News"),
    code(
        """
        def download_news(symbols: list[str], start: str, end: str) -> None:
            # Keep news chunked by symbol so reruns can skip completed names.
            for symbol in symbols:
                fetch_paged(
                    dataset="news",
                    object_name=symbol,
                    granularity="articles",
                    url=f"{DATA_BASE_URL}/v1beta1/news",
                    params={
                        "symbols": symbol,
                        "start": start,
                        "end": end,
                        "limit": NEWS_LIMIT_PER_PAGE,
                        "sort": "asc",
                        "include_content": "true",
                    },
                    data_key="news",
                )


        plan = DOWNLOAD_PLAN["news"]
        if plan["enabled"]:
            download_news(plan["symbols"], STARTS["news"], END)
        """
    ),
    md("## Optional Options Bars"),
    code(
        """
        def current_price(symbol: str) -> float | None:
            csv_path, _ = paths("stocks", symbol, "1Day")
            if not csv_path.exists():
                return None
            df = pd.read_csv(csv_path, usecols=["timestamp", "c", "close"] if False else None)
            close_col = "c" if "c" in df.columns else "close"
            if close_col not in df.columns or df.empty:
                return None
            return float(pd.to_numeric(df[close_col], errors="coerce").dropna().iloc[-1])


        def option_contracts_for(underlying: str) -> list[str]:
            price = current_price(underlying)
            contracts = []
            expiration_lte = (datetime.now(timezone.utc).date() + timedelta(days=OPTION_EXPIRATION_DAYS_AHEAD)).isoformat()
            for status in ("active", "inactive"):
                params = {
                    "underlying_symbols": underlying,
                    "status": status,
                    "expiration_date_gte": STARTS["options"],
                    "expiration_date_lte": expiration_lte,
                    "limit": REQUEST_LIMIT_PER_PAGE,
                }
                if price:
                    params["strike_price_gte"] = round(price * (1 - OPTION_STRIKE_BAND_PCT), 2)
                    params["strike_price_lte"] = round(price * (1 + OPTION_STRIKE_BAND_PCT), 2)

                page_token = None
                while True:
                    page_params = dict(params)
                    if page_token:
                        page_params["page_token"] = page_token
                    payload = request_json(f"{TRADING_BASE_URL}/v2/options/contracts", page_params).json()
                    rows = payload.get("option_contracts", [])
                    contracts.extend(row["symbol"] for row in rows if row.get("symbol"))
                    page_token = payload.get("next_page_token")
                    if not page_token:
                        break
            return sorted(set(contracts))


        def download_option_bars() -> None:
            if not DOWNLOAD_OPTIONS:
                print("DOWNLOAD_OPTIONS is False; skipping options bars.")
                return
            plan = DOWNLOAD_PLAN["options_5min"]
            for underlying in plan["symbols"]:
                contracts = option_contracts_for(underlying)
                print(f"{underlying}: {len(contracts)} contracts")
                for contract in contracts:
                    fetch_paged(
                        dataset="options",
                        object_name=contract,
                        granularity=plan["timeframe"],
                        url=f"{DATA_BASE_URL}/v1beta1/options/bars",
                        params={
                            "symbols": contract,
                            "timeframe": plan["timeframe"],
                            "start": STARTS["options"],
                            "end": END,
                            "limit": REQUEST_LIMIT_PER_PAGE,
                            "sort": "asc",
                            "feed": "indicative",
                        },
                        data_key="bars",
                    )


        download_option_bars()
        """
    ),
    md("## Checks"),
    code(
        """
        summary = []
        for csv_path in sorted(OUTPUT_DIR.rglob("*.csv")):
            state_path = csv_path.with_suffix(".state.json")
            state = read_state(state_path)
            try:
                rows = sum(1 for _ in csv_path.open("r", encoding="utf-8")) - 1
            except Exception:
                rows = None
            summary.append({
                "dataset": csv_path.relative_to(OUTPUT_DIR).parts[0],
                "granularity": csv_path.relative_to(OUTPUT_DIR).parts[1],
                "file": str(csv_path),
                "rows": rows,
                "status": state.get("status", "missing_state"),
                "error": state.get("error"),
            })

        summary_df = pd.DataFrame(summary)
        if not summary_df.empty:
            display(summary_df.groupby(["dataset", "granularity", "status"]).agg(files=("file", "count"), rows=("rows", "sum")).reset_index())
            display(summary_df[summary_df["status"].eq("error")].head(50))
        else:
            print("No files downloaded yet.")
        """
    ),
    md(
        """
        ## Next Steps

        Rerun the notebook any time it stops. Completed files are skipped; incomplete files resume from the latest timestamp already written.

        For Great Recession coverage, Alpaca's current equity history limit appears to be too recent. Use a second source such as Stooq, Nasdaq Data Link,
        Polygon, Databento, or paid survivorship-bias-aware data if you need 2006-2009 bars.
        """
    ),
]

nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NOTEBOOK)
print(NOTEBOOK)

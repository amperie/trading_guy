"""Interactive debug REPL for pausing a running engine and inspecting state.

Press Ctrl-C during any backtest or live session to enter the REPL.
Type 'help' at the prompt for the full command list.
"""

import json
import os
import signal
import sys
import threading
from collections import deque
from datetime import datetime


class DebugREPL:
    """
    Pause the engine between ticks and inspect or dump component state.

    Attach engine components via attach(), then call enter() to block
    until the user types 'c' (resume) or 'q' (stop the engine).

    Tab-completion is available for all commands and, after 'eval',
    for Python attribute access on al/pf/om.
    """

    COMMANDS = [
        "al", "algorithm",
        "pf", "portfolio",
        "om", "ordermanager",
        "dump", "eval", "help",
        "continue", "c",
        "quit", "q",
    ]
    DUMP_TARGETS = [
        "al", "algorithm",
        "pf", "portfolio",
        "om", "ordermanager",
        "pf.signals", "pf.ticks", "pf.values",
        "om.orders", "om.pending",
        "al.history", "al.indicators",
    ]

    def __init__(self):
        self.al = None
        self.pf = None
        self.om = None
        self._quit = False
        self._readline_ready = False

    def attach(self, al=None, pf=None, om=None):
        """Attach engine components for inspection."""
        self.al = al
        self.pf = pf
        self.om = om

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def enter(self):
        """Block until the user types 'c' (resume) or 'q' (quit engine)."""
        self._quit = False
        self._setup_readline()
        print("\n\n=== DEBUG MODE === type 'help' for commands, 'c' to resume ===\n")

        # While the REPL is active, temporarily restore the default SIGINT
        # handler so Ctrl-C can be used to exit the prompt (main thread only).
        in_main = threading.current_thread() is threading.main_thread()
        saved_handler = None
        if in_main:
            try:
                saved_handler = signal.getsignal(signal.SIGINT)
                signal.signal(signal.SIGINT, signal.default_int_handler)
            except Exception:
                saved_handler = None

        try:
            self._loop()
        finally:
            if in_main and saved_handler is not None:
                try:
                    signal.signal(signal.SIGINT, saved_handler)
                except Exception:
                    pass

    def _loop(self):
        while True:
            try:
                line = input("debug> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nResuming...")
                return

            if not line:
                continue

            parts = line.split(None, 1)
            cmd = parts[0].lower()
            args = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("c", "continue"):
                print("Resuming...")
                return
            elif cmd in ("q", "quit"):
                self._quit = True
                return
            elif cmd == "help":
                self._cmd_help()
            elif cmd in ("al", "algorithm"):
                self._cmd_print(self.al, "Algorithm")
            elif cmd in ("pf", "portfolio"):
                self._cmd_print(self.pf, "Portfolio")
            elif cmd in ("om", "ordermanager"):
                self._cmd_print(self.om, "OrderManager")
            elif cmd == "dump":
                self._cmd_dump(args)
            elif cmd == "eval":
                self._cmd_eval(args)
            else:
                print(f"Unknown command '{cmd}'. Type 'help'.")

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def _cmd_help(self):
        print("""
  al / algorithm            Print algorithm state (JSON)
  pf / portfolio            Print portfolio state (JSON)
  om / ordermanager         Print order manager state (JSON)

  dump al [file]            JSON: full algorithm state          → debug_al.json
  dump pf [file]            JSON: full portfolio state          → debug_pf.json
  dump om [file]            JSON: full order manager state      → debug_om.json
  dump pf.signals [file]    CSV: signals history                → debug_pf_signals.csv
  dump pf.ticks [file]      CSV: tick price history             → debug_pf_ticks.csv
  dump pf.values [file]     CSV: equity curve (value + cash)   → debug_pf_values.csv
  dump om.orders [file]     CSV: all orders                     → debug_om_orders.csv
  dump om.pending [file]    CSV: pending orders only            → debug_om_pending.csv
  dump al.history [file]    CSV: price_data_history deques      → debug_al_history.csv
  dump al.indicators [file] CSV: algo indicators (if impl.)     → debug_al_indicators.csv

  eval <expr>               Evaluate Python (al/pf/om in scope, tab-complete)
  c / continue              Resume execution
  q / quit                  Stop the engine
""")

    def _cmd_print(self, obj, label):
        if obj is None:
            print(f"{label}: not attached")
            return
        print(f"\n=== {label} ({type(obj).__name__}) ===")
        print(json.dumps(self._serialize(vars(obj)), indent=2, default=str))
        print()

    def _cmd_dump(self, args):
        parts = args.split(None, 1)
        if not parts:
            print("Usage: dump <target> [filename]")
            return
        target = parts[0].lower()
        filename = parts[1].strip() if len(parts) > 1 else None

        dispatch = {
            "al":            (self._dump_json,          self.al,   "debug_al.json"),
            "algorithm":     (self._dump_json,          self.al,   "debug_al.json"),
            "pf":            (self._dump_json,          self.pf,   "debug_pf.json"),
            "portfolio":     (self._dump_json,          self.pf,   "debug_pf.json"),
            "om":            (self._dump_json,          self.om,   "debug_om.json"),
            "ordermanager":  (self._dump_json,          self.om,   "debug_om.json"),
            "pf.signals":    (self._dump_pf_signals,    None,      "debug_pf_signals.csv"),
            "pf.ticks":      (self._dump_pf_ticks,      None,      "debug_pf_ticks.csv"),
            "pf.values":     (self._dump_pf_values,     None,      "debug_pf_values.csv"),
            "om.orders":     (self._dump_om_orders,     False,     "debug_om_orders.csv"),
            "om.pending":    (self._dump_om_orders,     True,      "debug_om_pending.csv"),
            "al.history":    (self._dump_al_history,    None,      "debug_al_history.csv"),
            "al.indicators": (self._dump_al_indicators, None,      "debug_al_indicators.csv"),
        }

        if target not in dispatch:
            print(f"Unknown target '{target}'. Options: {', '.join(dispatch)}")
            return

        fn, arg, default_file = dispatch[target]
        fn(arg, filename or default_file)

    def _cmd_eval(self, expr):
        if not expr:
            print("Usage: eval <expression>")
            return
        try:
            result = eval(expr, {"al": self.al, "pf": self.pf, "om": self.om})
            print(repr(result))
        except Exception as e:
            print(f"Error: {e}")

    # ------------------------------------------------------------------
    # Dump handlers
    # ------------------------------------------------------------------

    def _dump_json(self, obj, path):
        if obj is None:
            print("Object not attached.")
            return
        with open(path, "w") as f:
            json.dump(self._serialize(vars(obj)), f, indent=2, default=str)
        print(f"Saved → {os.path.abspath(path)}")

    def _dump_pf_signals(self, _, path):
        import pandas as pd
        if self.pf is None:
            print("Portfolio not attached.")
            return
        history = getattr(self.pf, "signals_history", {})
        if not history:
            print("No signals_history (keep_history may be disabled).")
            return
        rows = []
        for ts, signals in history.items():
            for sig in signals:
                row = {
                    "timestamp": ts,
                    "symbol": sig.symbol,
                    "type": sig.type.name,
                    "strength": sig.strength,
                }
                for k, v in (sig.metadata or {}).items():
                    row[f"metadata.{k}"] = v
                rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Saved {len(rows)} rows → {os.path.abspath(path)}")

    def _dump_pf_ticks(self, _, path):
        import pandas as pd
        if self.pf is None:
            print("Portfolio not attached.")
            return
        history = getattr(self.pf, "tick_history", {})
        if not history:
            print("No tick_history (keep_history may be disabled).")
            return
        rows = []
        for ts, tick in history.items():
            for bar in tick:
                rows.append({
                    "timestamp": ts,
                    "symbol": bar.symbol,
                    "open": bar.open, "high": bar.high,
                    "low": bar.low, "close": bar.close,
                    "volume": bar.volume, "vwap": bar.vwap,
                })
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Saved {len(rows)} rows → {os.path.abspath(path)}")

    def _dump_pf_values(self, _, path):
        import pandas as pd
        if self.pf is None:
            print("Portfolio not attached.")
            return
        value_history = getattr(self.pf, "value_history", {})
        cash_history = getattr(self.pf, "cash_history", {})
        if not value_history:
            print("No value_history (keep_history may be disabled).")
            return
        rows = [
            {"timestamp": ts, "total_value": v, "cash": cash_history.get(ts)}
            for ts, v in value_history.items()
        ]
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Saved {len(rows)} rows → {os.path.abspath(path)}")

    def _dump_om_orders(self, pending_only, path):
        import pandas as pd
        if self.om is None:
            print("OrderManager not attached.")
            return
        source = self.om.pending_orders_by_id if pending_only else self.om.all_orders
        rows = []
        for o in source.values():
            rows.append({
                "order_id": o.order_id,
                "type": o.type.name,
                "action": o.action.name,
                "symbol": o.symbol,
                "price": o.price,
                "quantity": o.quantity,
                "status": o.status.name,
                "placed_datetime": o.placed_datetime,
                "executed_datetime": o.executed_datetime,
                "cash": o.cash,
                "tx_cost": o.tx_cost,
                "parent_id": o.parent_id,
                "platform_id": getattr(o, "platform_id", None),
            })
        if not rows:
            print("No orders.")
            return
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Saved {len(rows)} rows → {os.path.abspath(path)}")

    def _dump_al_history(self, _, path):
        import pandas as pd
        if self.al is None:
            print("Algorithm not attached.")
            return
        history = getattr(self.al, "price_data_history", {})
        if not history:
            print("No price_data_history (history_length may be 0).")
            return
        rows = []
        for symbol, pdq in history.items():
            for bar in pdq:
                rows.append({
                    "symbol": symbol,
                    "timestamp": bar.timestamp,
                    "open": bar.open, "high": bar.high,
                    "low": bar.low, "close": bar.close,
                    "volume": bar.volume, "vwap": bar.vwap,
                })
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Saved {len(rows)} rows → {os.path.abspath(path)}")

    def _dump_al_indicators(self, _, path):
        import pandas as pd
        if self.al is None:
            print("Algorithm not attached.")
            return
        if not hasattr(self.al, "get_indicator_dataframe"):
            print("Algorithm does not implement get_indicator_dataframe() — no indicators to dump.")
            return
        df = self.al.get_indicator_dataframe()
        if df is None or (hasattr(df, "empty") and df.empty):
            print("Algorithm returned no indicator data.")
            return
        df.to_csv(path, index=False)
        print(f"Saved {len(df)} rows → {os.path.abspath(path)}")

    # ------------------------------------------------------------------
    # readline setup
    # ------------------------------------------------------------------

    def _setup_readline(self):
        if self._readline_ready:
            return
        try:
            import readline
            import rlcompleter

            repl = self

            class _Completer:
                def __init__(self):
                    # Attribute completer for eval expressions (al.x, pf.y, etc.)
                    self._attr = rlcompleter.Completer(
                        {"al": repl.al, "pf": repl.pf, "om": repl.om}
                    )
                    self._matches = []

                def complete(self, text, state):
                    if state == 0:
                        buf = readline.get_line_buffer()
                        parts = buf.split()
                        first = parts[0].lower() if parts else ""

                        if first == "dump" and (len(parts) > 1 or buf.endswith(" ")):
                            # Complete dump target names
                            prefix = parts[1] if len(parts) > 1 else ""
                            self._matches = [t for t in DebugREPL.DUMP_TARGETS
                                             if t.startswith(prefix)]
                        elif first == "eval":
                            # Python attribute completion for al/pf/om
                            self._matches = []
                            for i in range(200):
                                m = self._attr.complete(text, i)
                                if m is None:
                                    break
                                self._matches.append(m)
                        elif not parts or (len(parts) == 1 and not buf.endswith(" ")):
                            # First word: complete command names
                            self._matches = [c for c in DebugREPL.COMMANDS
                                             if c.startswith(text)]
                        else:
                            self._matches = []

                    return self._matches[state] if state < len(self._matches) else None

            readline.set_completer(_Completer().complete)
            # Use only whitespace as delimiters so "al.price_h<tab>" works as one token
            readline.set_completer_delims(" \t\n")
            # macOS ships libedit; GNU readline uses a different binding call
            doc = getattr(readline, "__doc__", "") or ""
            if "libedit" in doc or sys.platform == "darwin":
                readline.parse_and_bind("bind ^I rl_complete")
            else:
                readline.parse_and_bind("tab: complete")
            self._readline_ready = True
        except Exception:
            pass  # readline unavailable — silently skip completion

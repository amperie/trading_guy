# Orders and OrderManager

Base: `core.order_manager.OrderManager`. Override:
- `_submit_order_to_backend(order, tick, positions, pf_cash)`
- `_update_order_status_from_backend(order, tick, positions, pf_cash)`
- `_update_orders_statuses_from_backend(orders, tick, positions, pf_cash)`

Public: `submit_order()`, `update_order_status()`, `update_pending_orders()`.
Tracks: `_all_orders`, `_pending_orders_by_id`, `_filled_orders_by_id`.
Impl: `BacktestingOM` (`core/om/`) — instant market fills; bracket stop/profit logic.

**OrderType:** `MARKET`, `BRACKET`, `STOP_LOSS`, `PROFIT_TAKER`
**OrderStatus:** `PENDING` → `PENDING_SALE` (bracket filled, child pending) → `FILLED` / `CANCELED`
**OrderAction:** `BUY`, `SELL`

**BracketOrder.create_bracket_order(...)** → one `BracketOrder`. Children: `get_child_order("STOP"|"PROFIT"|"MANUAL_ORDER")`.
Props: `MANUAL_SALE`, `SOLD_ORDER`. Children have `parent_id`. Triggering one child cancels the other.
Order IDs: `f"local-{uuid.uuid4()}"`.

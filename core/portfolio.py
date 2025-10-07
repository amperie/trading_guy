"""
Portfolio class
- Stores configuration about the portfolio strategy and priorities for each security
- Tracks what the portfolio is comprised of
- Takes signals as input and adjusts the portfolio
Interfaces:
- Initialize
- create_orders: takes signal inputs and creates a list of orders
    - Keeps a history of orders and signals
    - Has an OrderManager member to handle the orders
"""

class Portfolio:
    pass
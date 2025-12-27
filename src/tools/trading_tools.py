"""Trading tools for stock and options trading.

This module provides LangChain tools for executing trading operations:
- Stock trading (buy/sell)
- Options trading (buy/sell calls and puts)
- Position management (clear positions)
- Price alerts (stock price alerts)

These tools are used by the Task Agent to execute user trading requests.
Note: These are placeholder implementations for demonstration purposes.
"""

from langchain_core.tools import tool
from typing import Optional


@tool
def buy_stock(symbol: str, quantity: int, order_type: str = "market") -> str:
    """
    Buy stocks.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
        quantity: Number of shares to buy
        order_type: Type of order - 'market' or 'limit' (default: 'market')
    
    Returns:
        Confirmation message with order details
    """
    # Simple implementation - in production, this would connect to a trading API
    return f"Order placed: Buy {quantity} shares of {symbol} ({order_type} order)"


@tool
def sell_stock(symbol: str, quantity: int, order_type: str = "market") -> str:
    """
    Sell stocks.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
        quantity: Number of shares to sell
        order_type: Type of order - 'market' or 'limit' (default: 'market')
    
    Returns:
        Confirmation message with order details
    """
    # Simple implementation - in production, this would connect to a trading API
    return f"Order placed: Sell {quantity} shares of {symbol} ({order_type} order)"


@tool
def buy_options(symbol: str, option_type: str, quantity: int, order_type: str = "market") -> str:
    """
    Buy options contracts.
    
    Args:
        symbol: Underlying stock symbol (e.g., 'AAPL', 'GOOGL')
        option_type: 'call' or 'put'
        quantity: Number of option contracts to buy
        order_type: Type of order - 'market' or 'limit' (default: 'market')
    
    Returns:
        Confirmation message with order details
    """
    # Simple implementation - in production, this would connect to a trading API
    return f"Order placed: Buy {quantity} {option_type} option contracts of {symbol} ({order_type} order)"


@tool
def sell_options(symbol: str, option_type: str, quantity: int, order_type: str = "market") -> str:
    """
    Sell options contracts.
    
    Args:
        symbol: Underlying stock symbol (e.g., 'AAPL', 'GOOGL')
        option_type: 'call' or 'put'
        quantity: Number of option contracts to sell
        order_type: Type of order - 'market' or 'limit' (default: 'market')
    
    Returns:
        Confirmation message with order details
    """
    # Simple implementation - in production, this would connect to a trading API
    return f"Order placed: Sell {quantity} {option_type} option contracts of {symbol} ({order_type} order)"


@tool
def clear_positions(symbol: Optional[str] = None) -> str:
    """
    Clear all open positions or positions for a specific symbol.
    
    Args:
        symbol: Optional stock symbol to clear positions for (e.g., 'AAPL', 'GOOGL').
                If None, clears all open positions.
    
    Returns:
        Confirmation message with cleared positions details
    """
    # Simple implementation - in production, this would connect to a trading API
    if symbol:
        return f"All open positions for {symbol} have been cleared."
    else:
        return "All open positions have been cleared."


@tool
def stock_price_alert(symbol: str, target_price: float, direction: str = "above") -> str:
    """
    Set up a price alert for a stock.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
        target_price: Target price to monitor
        direction: Alert direction - 'above' or 'below' the target price (default: 'above')
    
    Returns:
        Confirmation message with alert details
    """
    # Simple implementation - in production, this would connect to a trading API
    return f"Price alert set for {symbol}: Alert when price goes {direction} ${target_price:.2f}"




"""Trading tools for stock and options trading."""

from langchain_core.tools import tool
from guardrails.tool_guardrails import tool_guardrails


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
    # Guardrail validation
    args = {'symbol': symbol, 'quantity': quantity, 'order_type': order_type}
    validation = tool_guardrails.validate_trading_tool('buy_stock', args)
    if not validation.passed:
        return f"Error: {validation.reason}"
    
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
    # Guardrail validation
    args = {'symbol': symbol, 'quantity': quantity, 'order_type': order_type}
    validation = tool_guardrails.validate_trading_tool('sell_stock', args)
    if not validation.passed:
        return f"Error: {validation.reason}"
    
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
    # Guardrail validation
    args = {'symbol': symbol, 'option_type': option_type, 'quantity': quantity, 'order_type': order_type}
    validation = tool_guardrails.validate_trading_tool('buy_options', args)
    if not validation.passed:
        return f"Error: {validation.reason}"
    
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
    # Guardrail validation
    args = {'symbol': symbol, 'option_type': option_type, 'quantity': quantity, 'order_type': order_type}
    validation = tool_guardrails.validate_trading_tool('sell_options', args)
    if not validation.passed:
        return f"Error: {validation.reason}"
    
    # Simple implementation - in production, this would connect to a trading API
    return f"Order placed: Sell {quantity} {option_type} option contracts of {symbol} ({order_type} order)"




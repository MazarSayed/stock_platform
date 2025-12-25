"""Basic tool guardrails for trading operations."""

import re
from typing import Dict
from models.models import GuardrailResult
from utils.utils import logger


class ToolGuardrails:
    """Guardrails for trading tool invocations."""

    # Trading tool names
    TRADING_TOOLS = ['buy_stock', 'sell_stock', 'buy_options', 'sell_options']
    
    # Limits
    MAX_QUANTITY = 10000  # Maximum shares/contracts per order
    MIN_QUANTITY = 1  # Minimum shares/contracts per order
    MAX_ORDERS_PER_SESSION = 10  # Maximum trading orders per session
    
    # Valid order types
    VALID_ORDER_TYPES = ['market', 'limit']
    
    def __init__(self):
        """Initialize tool guardrails."""
        self.order_counts: Dict[str, int] = {}  # Track orders per session
    
    def validate_symbol(self, symbol: str) -> GuardrailResult:
        """Validate stock symbol format."""
        if not symbol:
            return GuardrailResult(
                passed=False,
                reason="Stock symbol cannot be empty"
            )
        
        # Normalize to uppercase
        symbol = symbol.upper().strip()
        
        # Check format: should be 1-5 uppercase letters/numbers
        if not re.match(r'^[A-Z0-9]{1,5}$', symbol):
            return GuardrailResult(
                passed=False,
                reason=f"Invalid symbol format: {symbol}. Symbols should be 1-5 uppercase alphanumeric characters"
            )
        
        return GuardrailResult(passed=True)
    
    def validate_quantity(self, quantity: int) -> GuardrailResult:
        """Validate order quantity."""
        if not isinstance(quantity, int):
            return GuardrailResult(
                passed=False,
                reason="Quantity must be an integer"
            )
        
        if quantity < self.MIN_QUANTITY:
            return GuardrailResult(
                passed=False,
                reason=f"Quantity must be at least {self.MIN_QUANTITY}"
            )
        
        if quantity > self.MAX_QUANTITY:
            return GuardrailResult(
                passed=False,
                reason=f"Quantity {quantity} exceeds maximum allowed ({self.MAX_QUANTITY} shares/contracts)"
            )
        
        return GuardrailResult(passed=True)
    
    def validate_order_type(self, order_type: str) -> GuardrailResult:
        """Validate order type."""
        if not order_type:
            return GuardrailResult(passed=True)  # Default to market
        
        order_type_lower = order_type.lower().strip()
        if order_type_lower not in self.VALID_ORDER_TYPES:
            return GuardrailResult(
                passed=False,
                reason=f"Invalid order type: {order_type}. Must be one of: {', '.join(self.VALID_ORDER_TYPES)}"
            )
        
        return GuardrailResult(passed=True)
    
    def check_order_limit(self, session_id: str) -> GuardrailResult:
        """Check if session has exceeded order limit."""
        order_count = self.order_counts.get(session_id, 0)
        
        if order_count >= self.MAX_ORDERS_PER_SESSION:
            return GuardrailResult(
                passed=False,
                reason=f"Maximum order limit ({self.MAX_ORDERS_PER_SESSION}) reached for this session. Please start a new session."
            )
        
        return GuardrailResult(passed=True)
    
    def increment_order_count(self, session_id: str):
        """Increment order count for a session."""
        self.order_counts[session_id] = self.order_counts.get(session_id, 0) + 1
        logger.info(f"Order count for session {session_id}: {self.order_counts[session_id]}")
    
    def validate_trading_tool(
        self,
        tool_name: str,
        args: Dict,
        session_id: str = None
    ) -> GuardrailResult:
        """Validate trading tool invocation with all checks."""
        # Check if tool is a trading tool
        if tool_name not in self.TRADING_TOOLS:
            return GuardrailResult(passed=True)  
        
        # Check order limit per session
        if session_id:
            limit_check = self.check_order_limit(session_id)
            if not limit_check.passed:
                return limit_check
        
        # Validate symbol
        symbol = args.get('symbol', '')
        symbol_check = self.validate_symbol(symbol)
        if not symbol_check.passed:
            return symbol_check
        
        # Validate quantity
        quantity = args.get('quantity', 0)
        quantity_check = self.validate_quantity(quantity)
        if not quantity_check.passed:
            return quantity_check
        
        # Validate order type
        order_type = args.get('order_type', 'market')
        order_type_check = self.validate_order_type(order_type)
        if not order_type_check.passed:
            return order_type_check
        
        if session_id:
            self.increment_order_count(session_id)
        
        return GuardrailResult(passed=True)
    
    def reset_session(self, session_id: str):
        """Reset order count for a session."""
        if session_id in self.order_counts:
            del self.order_counts[session_id]
            logger.info(f"Reset order count for session {session_id}")


# Global instance for easy import
tool_guardrails = ToolGuardrails()


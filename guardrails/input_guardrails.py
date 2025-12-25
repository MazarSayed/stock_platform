"""Basic input guardrails for LLM application security."""

import re
from models.models import GuardrailResult
from utils.utils import logger


class InputGuardrails:
    """Basic input safety guardrails to detect malicious or off-topic inputs."""
    
    # Common prompt injection patterns
    INJECTION_PATTERNS = [
        r'ignore\s+(previous|above|all)\s+(instructions|prompts|rules)',
        r'forget\s+(everything|all|previous)',
        r'you\s+are\s+now\s+(a|an)\s+',
        r'act\s+as\s+(if\s+you\s+are\s+)?(a|an)\s+',
        r'system\s*:\s*',
        r'<\|(system|assistant|user)\|>',
        r'\[INST\]|\[/INST\]',
        r'###\s*(system|instruction|prompt)',
        r'disregard\s+(all|previous)',
    ]
    
    # Off-topic keywords (non-stock trading related)
    OFF_TOPIC_KEYWORDS = [
        'hack', 'exploit', 'vulnerability', 'password', 'credit card',
        'ssn', 'social security', 'malware', 'virus', 'phishing',
        'cryptocurrency mining', 'bitcoin wallet', 'drug', 'illegal',
    ]
    
    def check_prompt_injection(self, input_text: str) -> GuardrailResult:
        """Check for prompt injection attempts."""
        input_lower = input_text.lower()
        
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, input_lower):
                logger.warning(f"Prompt injection detected: {pattern}")
                return GuardrailResult(
                    passed=False,
                    reason="Potential prompt injection detected"
                )
        
        return GuardrailResult(passed=True)
    
    def check_off_topic(self, input_text: str) -> GuardrailResult:
        """Check if input is off-topic (not related to stock trading)."""
        input_lower = input_text.lower()
        
        # Check for off-topic keywords
        for keyword in self.OFF_TOPIC_KEYWORDS:
            if keyword in input_lower:
                logger.warning(f"Off-topic keyword detected: {keyword}")
                return GuardrailResult(
                    passed=False,
                    reason="Query is not related to stock trading platform"
                )
        
        return GuardrailResult(passed=True)
    
    def sanitize_input(self, input_text: str) -> str:
        """Sanitize input by removing suspicious patterns."""
        sanitized = input_text
        
        # Remove common injection delimiters
        sanitized = re.sub(r'<\|.*?\|>', '', sanitized)
        sanitized = re.sub(r'\[INST\].*?\[/INST\]', '', sanitized, flags=re.DOTALL)
        sanitized = re.sub(r'###\s*(system|instruction|prompt).*', '', sanitized, flags=re.IGNORECASE)
        
        return sanitized.strip()
    
    def validate_input(self, input_text: str) -> GuardrailResult:
        """Run all input validation checks."""
        # Check prompt injection
        injection_check = self.check_prompt_injection(input_text)
        if not injection_check.passed:
            return injection_check
        
        # Check topic relevance
        topic_check = self.check_off_topic(input_text)
        if not topic_check.passed:
            return topic_check
        
        # Sanitize input
        sanitized = self.sanitize_input(input_text)
        
        return GuardrailResult(
            passed=True,
            sanitized_output=sanitized if sanitized != input_text else None
        )


# Global instance for easy import
input_guardrails = InputGuardrails()


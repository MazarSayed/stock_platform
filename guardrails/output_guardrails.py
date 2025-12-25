"""Basic output guardrails for LLM application security."""

import re
from utils.utils import logger
from models.models import GuardrailResult


class OutputGuardrails:
    """Basic output guardrails to filter sensitive data and validate responses."""
    
    # Sensitive data patterns
    SENSITIVE_PATTERNS = [
        (r'\b[A-Z0-9]{20,}\b', 'Potential API key or token'),
        (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', 'Potential credit card number'),
        (r'\b\d{3}-\d{2}-\d{4}\b', 'Potential SSN'),
        (r'password\s*[:=]\s*\S+', 'Password exposure'),
        (r'api[_-]?key\s*[:=]\s*\S+', 'API key exposure'),
        (r'secret\s*[:=]\s*\S+', 'Secret exposure'),
        (r'token\s*[:=]\s*\S+', 'Token exposure'),
    ]
    
    # Off-domain keywords (things that shouldn't be in stock trading responses)
    OFF_DOMAIN_KEYWORDS = [
        'hack', 'exploit', 'vulnerability', 'malware', 'virus',
        'illegal activity', 'drug', 'weapon', 'violence',
        'generate code for', 'write a script to', 'execute command',
        'bypass security', 'crack password', 'unauthorized access',
    ]
    
    # Potentially harmful code patterns
    CODE_PATTERNS = [
        r'```(?:python|bash|sh|shell|javascript|js|sql)',
        r'<script[^>]*>',
        r'eval\s*\(',
        r'exec\s*\(',
        r'subprocess\.',
        r'os\.system\s*\(',
    ]
    
    def check_sensitive_data(self, output_text: str) -> GuardrailResult:
        """Check for sensitive data in output."""
        for pattern, description in self.SENSITIVE_PATTERNS:
            if re.search(pattern, output_text, re.IGNORECASE):
                logger.warning(f"Sensitive data detected in output: {description}")
                return GuardrailResult(
                    passed=False,
                    reason=f"Response contains potentially sensitive data: {description}",
                    sanitized_output=self.sanitize_sensitive_data(output_text)
                )
        
        return GuardrailResult(passed=True)
    
    def check_domain_boundary(self, output_text: str) -> GuardrailResult:
        """Check if output stays within domain boundaries."""
        output_lower = output_text.lower()
        
        # Check for off-domain keywords
        for keyword in self.OFF_DOMAIN_KEYWORDS:
            if keyword in output_lower:
                logger.warning(f"Off-domain content detected: {keyword}")
                return GuardrailResult(
                    passed=False,
                    reason="Response contains content outside the stock trading platform domain",
                    blocked=True
                )
        
        # Check for code injection patterns
        for pattern in self.CODE_PATTERNS:
            if re.search(pattern, output_text, re.IGNORECASE):
                logger.warning(f"Code injection pattern detected: {pattern}")
                return GuardrailResult(
                    passed=False,
                    reason="Response contains potentially harmful code patterns",
                    blocked=True
                )
        
        return GuardrailResult(passed=True)
    
    def sanitize_sensitive_data(self, output_text: str) -> str:
        """Sanitize output by redacting sensitive patterns."""
        sanitized = output_text
        
        # Redact potential API keys/tokens
        sanitized = re.sub(r'\b([A-Z0-9]{20,})\b', r'[REDACTED-TOKEN]', sanitized)
        
        # Redact potential credit cards
        sanitized = re.sub(
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            '[REDACTED-CARD]',
            sanitized
        )
        
        # Redact potential SSN
        sanitized = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED-SSN]', sanitized)
        
        # Redact password patterns
        sanitized = re.sub(
            r'password\s*[:=]\s*\S+',
            'password: [REDACTED]',
            sanitized,
            flags=re.IGNORECASE
        )
        
        # Redact API key patterns
        sanitized = re.sub(
            r'api[_-]?key\s*[:=]\s*\S+',
            'api_key: [REDACTED]',
            sanitized,
            flags=re.IGNORECASE
        )
        
        # Redact secret patterns
        sanitized = re.sub(
            r'secret\s*[:=]\s*\S+',
            'secret: [REDACTED]',
            sanitized,
            flags=re.IGNORECASE
        )
        
        return sanitized
    
    def validate_output(self, output_text: str) -> GuardrailResult:
        """Run all output validation checks."""
        if not output_text:
            return GuardrailResult(
                passed=False,
                reason="Empty output",
                blocked=False
            )
        
        # 1. Check for sensitive data
        sensitive_check = self.check_sensitive_data(output_text)
        if not sensitive_check.passed and sensitive_check.sanitized_output:
            output_text = sensitive_check.sanitized_output
            logger.info("Output sanitized due to sensitive data detection")
        
        # 2. Check domain boundaries (this can block)
        domain_check = self.check_domain_boundary(output_text)
        if not domain_check.passed:
            if domain_check.blocked:
                return GuardrailResult(
                    passed=False,
                    reason=domain_check.reason or "Response violates domain boundaries",
                    blocked=True
                )
        
        # Return sanitized output if it was modified
        was_sanitized = not sensitive_check.passed and sensitive_check.sanitized_output
        return GuardrailResult(
            passed=True,
            sanitized_output=output_text if was_sanitized else None
        )


# Global instance for easy import
output_guardrails = OutputGuardrails()


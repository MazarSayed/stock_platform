"""Utility functions for the stock trading platform.

This module provides shared utility functions used across the platform:
- YAML prompt loading and formatting
- Logging configuration
- Common helper functions
"""

import yaml
import logging
from pathlib import Path

# Configure and create logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("stock_platform")


def load_prompt_yaml(prompt_key: str, prompts_file: str = "prompts/prompts.yaml") -> str:
    """Load a prompt from the prompts YAML file."""
    prompts_path = Path(__file__).parent.parent.parent / prompts_file
    
    with open(prompts_path, 'r', encoding='utf-8') as f:
        prompts_data = yaml.safe_load(f)
    
    # Navigate through nested keys (e.g., "agents.faq_agent")
    value = prompts_data
    for key in prompt_key.split('.'):
        value = value[key]
    
    return value.strip()


def format_prompt(template: str, **kwargs) -> str:
    """Format a prompt template with keyword arguments."""
    return template.format(**kwargs)


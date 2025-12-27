"""Script to run auto-evaluation on ground truth test cases.

This script executes the automated evaluation pipeline, running all test cases
from the ground truth file against the LangGraph workflow and generating
comprehensive evaluation reports with metrics and scores.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.evaluation.auto_evaluator import AutoEvaluator
from langfuse.langchain import CallbackHandler


async def main():
    """Run evaluation on all ground truth test cases."""
    evaluator = AutoEvaluator()
    langfuse_handler = CallbackHandler()    
    # Run evaluation
    await evaluator.run_evaluation(langfuse_handler=langfuse_handler)


if __name__ == "__main__":
    asyncio.run(main())


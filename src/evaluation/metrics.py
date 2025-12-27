"""Basic metrics tracking for evaluation.

This module provides the MetricsTracker class for collecting, aggregating,
and reporting evaluation metrics. It tracks individual test results, calculates
aggregate statistics, and saves results to JSON files.
"""

import json
from datetime import datetime
from typing import List, Optional, Dict
from pathlib import Path
from collections import defaultdict
import sys

# Add project root to path for models
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from models.models import EvaluationMetrics, AggregateMetrics
from ..utils.utils import logger


class MetricsTracker:
    """Track and aggregate basic evaluation metrics."""
    
    def __init__(self, output_dir: str = "data/evaluation/results"):
        """Initialize metrics tracker.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: List[EvaluationMetrics] = []
        self.tool_usage: Dict[str, int] = defaultdict(int)
        
    def add_metric(self, metric: EvaluationMetrics):
        """Add a single evaluation metric.
        
        Args:
            metric: EvaluationMetrics object to add
        """
        self.metrics.append(metric)
        # Track tool usage
        for tool in metric.actual_tools:
            self.tool_usage[tool] += 1
        
    def calculate_aggregate_metrics(self) -> AggregateMetrics:
        """Calculate aggregate metrics from all collected metrics.
        
        Returns:
            AggregateMetrics object with summary statistics
        """
        if not self.metrics:
            return AggregateMetrics(
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                average_relevance=0.0,
                average_accuracy=0.0,
                average_latency_ms=0.0,
                agent_match_rate=0.0,
                tools_match_rate=0.0,
                tool_usage_stats={},
                timestamp=datetime.now().isoformat()
            )
        
        total = len(self.metrics)
        passed = sum(1 for m in self.metrics if m.relevance_score >= 0.7 and m.accuracy_score >= 0.7 and m.agent_match and m.tools_match)
        failed = total - passed
        
        avg_relevance = sum(m.relevance_score for m in self.metrics) / total
        avg_accuracy = sum(m.accuracy_score for m in self.metrics) / total
        avg_latency = sum(m.latency_ms for m in self.metrics) / total
        agent_match_rate = sum(1 for m in self.metrics if m.agent_match) / total if total > 0 else 0.0
        tools_match_rate = sum(1 for m in self.metrics if m.tools_match) / total if total > 0 else 0.0
        
        return AggregateMetrics(
            total_tests=total,
            passed_tests=passed,
            failed_tests=failed,
            average_relevance=avg_relevance,
            average_accuracy=avg_accuracy,
            average_latency_ms=avg_latency,
            agent_match_rate=agent_match_rate,
            tools_match_rate=tools_match_rate,
            tool_usage_stats=dict(self.tool_usage),
            timestamp=datetime.now().isoformat()
        )
    
    def save_results(self, filename: Optional[str] = None):
        """Save evaluation results to JSON file.
        
        Args:
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = self.output_dir / filename
        
        results = {
            "individual_metrics": [m.model_dump() for m in self.metrics],
            "aggregate_metrics": self.calculate_aggregate_metrics().model_dump(),
            "summary": {
                "total_evaluations": len(self.metrics),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved evaluation results to {output_path}")
        return output_path
    
    def print_summary(self):
        """Print summary of evaluation metrics."""
        aggregate = self.calculate_aggregate_metrics()
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Tests: {aggregate.total_tests}")
        print(f"Passed: {aggregate.passed_tests} ({aggregate.passed_tests/aggregate.total_tests*100:.1f}%)")
        print(f"Failed: {aggregate.failed_tests} ({aggregate.failed_tests/aggregate.total_tests*100:.1f}%)")
        print(f"\nAverage Scores:")
        print(f"  Relevance: {aggregate.average_relevance:.3f}")
        print(f"  Accuracy: {aggregate.average_accuracy:.3f}")
        print(f"\nValidation:")
        print(f"  Agent Match Rate: {aggregate.agent_match_rate:.3f} ({aggregate.agent_match_rate*100:.1f}%)")
        print(f"  Tools Match Rate: {aggregate.tools_match_rate:.3f} ({aggregate.tools_match_rate*100:.1f}%)")
        print(f"\nTool Usage Stats:")
        if aggregate.tool_usage_stats:
            for tool, count in sorted(aggregate.tool_usage_stats.items(), key=lambda x: x[1], reverse=True):
                print(f"  {tool}: {count}")
        else:
            print("  No tools used")
        print(f"\nPerformance:")
        print(f"  Average Latency: {aggregate.average_latency_ms:.2f} ms")
        print("="*60 + "\n")

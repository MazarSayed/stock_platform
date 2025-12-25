"""Basic auto-evaluation script that compares responses against ground-truth."""

import asyncio
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langfuse import get_client
from graph.chatgrapgh import research_graph
from graph.state import State
from evaluation.metrics import MetricsTracker
from models.models import EvaluationMetrics, EvaluationScores
from utils.utils import load_prompt_yaml, format_prompt, logger


class AutoEvaluator:
    """Basic auto-evaluator that tests chatbot against ground-truth Q&A pairs."""
    
    def __init__(
        self,
        ground_truth_path: str = "data/evaluation/ground_truth.yaml",
        judge_model: str = "gpt-4o"
    ):
        self.ground_truth_path = Path(ground_truth_path)
        self.judge_llm = ChatOpenAI(model=judge_model, temperature=0)
        self.langfuse = get_client()
        self.metrics_tracker = MetricsTracker()
        self.ground_truth = self.load_ground_truth()
        self.session_id = None
        
    def load_ground_truth(self) -> List[Dict[str, Any]]:
        """Load ground-truth Q&A pairs from YAML."""
        if not self.ground_truth_path.exists():
            logger.warning(f"Ground truth file not found: {self.ground_truth_path}")
            return []
        
        with open(self.ground_truth_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data.get('test_cases', [])
    
    def evaluate_answer(
        self,
        expected_answer: str,
        actual_answer: str
    ) -> Dict[str, float]:
        """Evaluate answer using LLM-as-a-Judge with structured output."""
        prompt_template = load_prompt_yaml("evaluation.llm_judge")
        prompt = format_prompt(prompt_template, expected_answer=expected_answer, actual_answer=actual_answer)


        evaluation = self.judge_llm.with_structured_output(EvaluationScores).invoke(prompt)
        return {
            "relevance": evaluation.relevance,
            "accuracy": evaluation.accuracy
        }
    
    def extract_agent_from_messages(self, messages: List[Any]) -> str:
        """Extract the agent name from messages."""
        # Look for the last message with a name attribute
        for msg in reversed(messages):
            if hasattr(msg, "name") and msg.name and msg.name.endswith("_agent"):
                return msg.name
        return "supervisor"
    
    def extract_tools_from_trace(self, trace_id: str) -> List[str]:
        """Extract tool calls from Langfuse trace."""
        try:
            trace = self.langfuse.fetch_trace(trace_id)
            if not trace:
                return []
            
            tool_calls = []
            # Get observations from trace
            observations = trace.get('observations', [])
            
            for obs in observations:
                obs_type = obs.get('type', '')
                obs_name = obs.get('name', '').lower()
                
                # Check if it's a tool call
                if obs_type == 'SPAN' and ('tool' in obs_name or any(
                    tool_name in obs_name for tool_name in [
                        'buy_stock', 'sell_stock', 'buy_options', 'sell_options',
                        'faq_rag', 'market_analysis', 'web_search', 'tavily'
                    ]
                )):
                    tool_name = obs.get('name', '')
                    # Normalize tool names
                    if 'faq_rag' in tool_name.lower():
                        tool_name = 'faq_rag_tool'
                    elif 'market_analysis' in tool_name.lower():
                        tool_name = 'market_analysis_rag_tool'
                    elif 'web_search' in tool_name.lower() or 'tavily' in tool_name.lower():
                        tool_name = 'web_search_tool'
                    elif 'buy_stock' in tool_name.lower():
                        tool_name = 'buy_stock'
                    elif 'sell_stock' in tool_name.lower():
                        tool_name = 'sell_stock'
                    elif 'buy_options' in tool_name.lower():
                        tool_name = 'buy_options'
                    elif 'sell_options' in tool_name.lower():
                        tool_name = 'sell_options'
                    
                    if tool_name and tool_name not in tool_calls:
                        tool_calls.append(tool_name)
            
            return tool_calls
        except Exception as e:
            logger.warning(f"Could not extract tools from trace: {e}")
            return []
    
    async def evaluate_single_test(
        self,
        test_case: Dict[str, Any],
        langfuse_handler=None
    ) -> EvaluationMetrics:
        """Evaluate a single test case."""
        test_id = test_case.get('id', 'unknown')
        question = test_case.get('question', '')
        expected_answer = test_case.get('expected_answer', '')
        expected_agent = test_case.get('expected_agent', '')
        expected_tools = test_case.get('expected_tools', [])
        
        logger.info(f"Evaluating test {test_id}: {question[:50]}...")
        logger.info(f"Expected agent: {expected_agent}, Expected tools: {expected_tools}")
        
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "next": "supervisor",
            "response": ""
        }
        
        config = {}
        if langfuse_handler:
            config["callbacks"] = [langfuse_handler]
        
        start_time = time.time()
        trace_id = None
        
        try:
            # Create trace (associate with session if available)
            trace_params = {
                "name": f"evaluation_{test_id}",
                "input": {"question": question, "test_id": test_id},
                "metadata": {
                    "expected_agent": expected_agent,
                    "expected_tools": expected_tools,
                    "evaluation": True
                }
            }
            if self.session_id:
                trace_params["session_id"] = self.session_id
            
            trace = self.langfuse.trace(**trace_params)
            trace_id = trace.id
            
            # Invoke graph
            result = await research_graph.ainvoke(initial_state, config)
            
            # Extract response
            actual_answer = result.get("response", "")
            if not actual_answer and result.get("messages"):
                last_msg = result["messages"][-1]
                if hasattr(last_msg, "content"):
                    actual_answer = last_msg.content
            
            # Extract agent
            actual_agent = self.extract_agent_from_messages(result.get("messages", []))
            
            # Extract tools from trace
            actual_tools = self.extract_tools_from_trace(trace_id)
            
            # Validate agent and tools
            agent_match = (actual_agent == expected_agent) if expected_agent else True
            expected_tools_set = set(expected_tools) if expected_tools else set()
            actual_tools_set = set(actual_tools)
            tools_match = expected_tools_set.issubset(actual_tools_set) if expected_tools else True
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Evaluate answer
            scores = self.evaluate_answer(expected_answer, actual_answer)
            
            logger.info(f"Test {test_id} - Agent: {actual_agent} (expected: {expected_agent}, match: {agent_match})")
            logger.info(f"Test {test_id} - Tools: {actual_tools} (expected: {expected_tools}, match: {tools_match})")
            
            # Create metric
            metric = EvaluationMetrics(
                test_id=test_id,
                question=question,
                expected_answer=expected_answer,
                actual_answer=actual_answer,
                expected_agent=expected_agent,
                actual_agent=actual_agent,
                expected_tools=expected_tools,
                actual_tools=actual_tools,
                agent_match=agent_match,
                tools_match=tools_match,
                relevance_score=scores["relevance"],
                accuracy_score=scores["accuracy"],
                latency_ms=latency_ms,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                trace_id=trace_id
            )
            
            # Add to tracker
            self.metrics_tracker.add_metric(metric)
            
            # Add scores to Langfuse
            try:
                self.langfuse.score(trace_id, "relevance", scores["relevance"])
                self.langfuse.score(trace_id, "accuracy", scores["accuracy"])
                self.langfuse.score(trace_id, "agent_match", 1.0 if agent_match else 0.0)
                self.langfuse.score(trace_id, "tools_match", 1.0 if tools_match else 0.0)
            except Exception as e:
                logger.warning(f"Failed to add scores: {e}")
            
            return metric
            
        except Exception as e:
            logger.error(f"Error evaluating test {test_id}: {e}")
            error_metric = EvaluationMetrics(
                test_id=test_id,
                question=question,
                expected_answer=expected_answer,
                actual_answer="",
                expected_agent=expected_agent,
                actual_agent="error",
                expected_tools=expected_tools,
                actual_tools=[],
                agent_match=False,
                tools_match=False,
                relevance_score=0.0,
                accuracy_score=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                trace_id=trace_id,
                error=str(e)
            )
            self.metrics_tracker.add_metric(error_metric)
            return error_metric
    
    async def run_evaluation(
        self,
        langfuse_handler=None,
        max_tests: Optional[int] = None) -> MetricsTracker:
        
        """Run evaluation on all ground-truth test cases."""
        test_cases = self.ground_truth[:max_tests] if max_tests else self.ground_truth
        
        if not test_cases:
            logger.error("No test cases found!")
            return self.metrics_tracker
        
        # Create session ID for this evaluation run (sessions are created automatically when session_id is used)
        from datetime import datetime
        self.session_id = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting evaluation of {len(test_cases)} test cases...")
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"Running test {i}/{len(test_cases)}")
            await self.evaluate_single_test(test_case, langfuse_handler)
            await asyncio.sleep(0.5)
        
        # Save results and print summary
        self.metrics_tracker.save_results()
        self.metrics_tracker.print_summary()
        
        return self.metrics_tracker

"""Pydantic models for the stock trading platform.

This module defines all Pydantic data models used throughout the application:
- API request/response models (ChatMessage, ChatRequest, ChatResponse)
- Evaluation models (EvaluationMetrics, AggregateMetrics, EvaluationScores)
- RAG extraction models (QAPair, FAQExtraction)
- Guardrail models (GuardrailResult)
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


# Evaluation Models
class EvaluationMetrics(BaseModel):
    """Basic metrics for a single evaluation run."""
    test_id: str
    question: str
    expected_answer: str
    actual_answer: str
    expected_agent: str = ""
    actual_agent: str = ""
    expected_tools: List[str] = Field(default_factory=list)
    actual_tools: List[str] = Field(default_factory=list)
    agent_match: bool = False
    tools_match: bool = False
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    accuracy_score: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    timestamp: str = ""
    trace_id: Optional[str] = None
    error: Optional[str] = None


class AggregateMetrics(BaseModel):
    """Basic aggregate metrics."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    average_relevance: float = Field(ge=0.0, le=1.0)
    average_accuracy: float = Field(ge=0.0, le=1.0)
    average_latency_ms: float = Field(ge=0.0)
    agent_match_rate: float = Field(ge=0.0, le=1.0)
    tools_match_rate: float = Field(ge=0.0, le=1.0)
    tool_usage_stats: Dict[str, int] = Field(default_factory=dict)
    timestamp: str


class EvaluationScores(BaseModel):
    """Structured output for LLM evaluation."""
    relevance: float = Field(description="Relevance score (0-1)")
    accuracy: float = Field(description="Accuracy score (0-1)")


# API Models
class ChatMessage(BaseModel):
    """Chat message model."""
    role: str
    content: str
    agent: Optional[str] = None


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
    session_id: str
    agent: Optional[str] = None
    messages: List[ChatMessage] = []


# RAG Extractor Models
class QAPair(BaseModel):
    """Q&A pair structure."""
    question: str
    answer: str


class FAQExtraction(BaseModel):
    """FAQ extraction structure."""
    qa_pairs: List[QAPair]


# Guardrail Models
class GuardrailResult(BaseModel):
    """Result of a guardrail validation check."""
    passed: bool
    reason: Optional[str] = None
    sanitized_output: Optional[str] = None
    blocked: bool = False


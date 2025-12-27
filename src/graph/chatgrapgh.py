"""LangGraph workflow definition for multi-agent stock trading platform.

This module defines the main LangGraph workflow that orchestrates multiple
specialized agents (FAQ, Task, Market Insights) through a supervisor pattern.
The graph routes requests to appropriate agents and manages conversation state.
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from ..nodes.agents import faq_node, task_node, market_insights_node, research_supervisor_node
from .state import State


def route_supervisor(state: State) -> str:
    """Route to next node based on supervisor's decision.
    
    Args:
        state: Current graph state
        
    Returns:
        Next node name or END
    """
    next_node = state.get("next", "FINISH")
    
    # Map supervisor decisions to nodes
    if next_node == "faq_agent":
        return "faq_agent"
    elif next_node == "task_agent":
        return "task_agent"
    elif next_node == "market_insights_agent":
        return "market_insights_agent"
    elif next_node == "FINISH":
        return END
    else:
        # Default to END to prevent infinite loops
        return END


research_builder = StateGraph(State)
research_builder.add_node("supervisor", research_supervisor_node)
research_builder.add_node("faq_agent", faq_node)
research_builder.add_node("task_agent", task_node)
research_builder.add_node("market_insights_agent", market_insights_node)

# Set entry point
research_builder.add_edge(START, "supervisor")

# Add conditional edge from supervisor to agents or END
research_builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "faq_agent": "faq_agent",
        "task_agent": "task_agent",
        "market_insights_agent": "market_insights_agent",
        END: END,
    }
)

# All agents return to supervisor
research_builder.add_edge("faq_agent", "supervisor")
research_builder.add_edge("task_agent", "supervisor")
research_builder.add_edge("market_insights_agent", "supervisor")


# Compile with checkpointer for state management
checkpointer = MemorySaver()
research_graph = research_builder.compile(checkpointer=checkpointer)
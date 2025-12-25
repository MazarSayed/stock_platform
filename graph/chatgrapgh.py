from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from nodes.agents import faq_node, task_node, market_insights_node, research_supervisor_node
from langgraph.graph import START, END
from graph.state import State


def route_supervisor(state: State) -> str:
    """Route based on supervisor's decision in state."""
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
"""Agent node implementations for FAQ, task, and market insights agents.

This module creates and configures three specialized agents:
- FAQ Agent: Answers platform questions using RAG retrieval
- Task Agent: Executes trading operations (buy/sell stocks and options, clear positions, set price alerts)
- Market Insights Agent: Provides market analysis using RAG and web search

Each agent is implemented as a LangGraph node function that processes
state and returns updated state with agent responses.
"""

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode
from .supervisor import make_supervisor_node
from ..graph.state import State
from ..tools.trading_tools import buy_stock, sell_stock, buy_options, sell_options, clear_positions, stock_price_alert
from ..utils.utils import load_prompt_yaml, logger

llm = ChatOpenAI(model="gpt-4o")    

# Load prompts from YAML
FAQ_AGENT_PROMPT = load_prompt_yaml("agents.faq_agent")
TASK_AGENT_PROMPT = load_prompt_yaml("agents.task_agent")
MARKET_INSIGHTS_AGENT_PROMPT = load_prompt_yaml("agents.market_insights_agent")


# Create agents
from ..tools import faq_rag_tool, market_analysis_rag_tool, web_search_tool

faq_agent = create_agent(
    model=llm,
    tools=[faq_rag_tool],
    system_prompt=FAQ_AGENT_PROMPT
)

def faq_node(state: State, config: RunnableConfig = None) -> dict:
    """FAQ agent node that answers questions using RAG.
    
    Args:
        state: Current graph state
        config: Optional runtime configuration
        
    Returns:
        Updated state with agent response
    """
    invoke_config = config if config else {}
    result = faq_agent.invoke(state, config=invoke_config)
    last_message_content = result["messages"][-1].content if result.get("messages") else "FAQ response completed."
    
    # Get existing messages and append the agent's response
    existing_messages = state.get("messages", [])
    agent_message = HumanMessage(content=last_message_content, name="faq_agent")
    
    # Return update - explicit edge will route back to supervisor
    return {
        "messages": existing_messages + [agent_message],
        "response": last_message_content
    }


task_agent = create_agent(
    model=llm,
    tools=[sell_stock, buy_stock, sell_options, buy_options, clear_positions, stock_price_alert],
    system_prompt=TASK_AGENT_PROMPT
)

def task_node(state: State, config: RunnableConfig = None) -> dict:
    """Task agent node that executes trading operations.
    
    Args:
        state: Current graph state
        config: Optional runtime configuration
        
    Returns:
        Updated state with agent response
    """
    invoke_config = config if config else {}
    result = task_agent.invoke(state, config=invoke_config)
    last_message_content = result["messages"][-1].content if result.get("messages") else "Task completed."
    
    # Get existing messages and append the agent's response
    existing_messages = state.get("messages", [])
    agent_message = HumanMessage(content=last_message_content, name="task_agent")
    
    # Return update - explicit edge will route back to supervisor
    return {
        "messages": existing_messages + [agent_message],
        "response": last_message_content
    }


market_insights_agent = create_agent(
    model=llm,
    tools=[market_analysis_rag_tool, web_search_tool],
    system_prompt=MARKET_INSIGHTS_AGENT_PROMPT
)

def market_insights_node(state: State, config: RunnableConfig = None) -> dict:
    """Market insights agent node that provides market analysis.
    
    Args:
        state: Current graph state
        config: Optional runtime configuration
        
    Returns:
        Updated state with agent response
    """
    invoke_config = config if config else {}
    result = market_insights_agent.invoke(state, config=invoke_config)
    last_message_content = result["messages"][-1].content if result.get("messages") else "Market insights analysis completed."
    
    # Get existing messages and append the agent's response
    existing_messages = state.get("messages", [])
    agent_message = HumanMessage(content=last_message_content, name="market_insights_agent")
    
    # Return update - explicit edge will route back to supervisor
    return {
        "messages": existing_messages + [agent_message],
        "response": last_message_content
    }


research_supervisor_node = make_supervisor_node(llm, ["faq_agent", "task_agent", "market_insights_agent"])


from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from nodes.supervisor import make_supervisor_node
from graph.state import State
from tools.trading_tools import buy_stock, sell_stock, buy_options, sell_options
from utils.utils import load_prompt_yaml

llm = ChatOpenAI(model="gpt-4o")    

# Load prompts from YAML
FAQ_AGENT_PROMPT = load_prompt_yaml("agents.faq_agent")
TASK_AGENT_PROMPT = load_prompt_yaml("agents.task_agent")
MARKET_INSIGHTS_AGENT_PROMPT = load_prompt_yaml("agents.market_insights_agent")


# Create agents
from tools import faq_rag_tool, market_analysis_rag_tool, web_search_tool

faq_agent = create_agent(
    model=llm,
    tools=[faq_rag_tool],
    system_prompt=FAQ_AGENT_PROMPT
)

def faq_node(state: State) -> dict:
    result = faq_agent.invoke(state)
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
    tools=[sell_stock, buy_stock, sell_options, buy_options],
    system_prompt=TASK_AGENT_PROMPT
)

def task_node(state: State) -> dict:
    result = task_agent.invoke(state)
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

def market_insights_node(state: State) -> dict:
    result = market_insights_agent.invoke(state)
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


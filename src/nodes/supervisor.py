"""Supervisor node implementation for routing requests to appropriate agents.

This module provides the supervisor node factory function that creates a
routing node. The supervisor analyzes incoming messages and decides which
specialized agent should handle the request, or if the conversation should
finish. It uses structured LLM output to make routing decisions.
"""

from typing import Callable
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from ..graph.state import State
from pydantic import BaseModel, Field
from ..utils.utils import load_prompt_yaml, logger

def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> Callable:
    """Create a supervisor node that routes to the appropriate worker.
    
    Args:
        llm: Language model for supervisor decisions
        members: List of agent names to route to
        
    Returns:
        Supervisor node function
    """
    options = ["FINISH"] + members
    
    # Load supervisor prompt from YAML
    system_prompt = load_prompt_yaml("supervisor")

    class Router(BaseModel):
        """Worker to route to next. If no workers needed, route to FINISH."""
        next: str = Field(description=f"Must be one of: {', '.join(options)}")

    def supervisor_node(state: State) -> dict:
        """Supervisor node that routes requests to appropriate agents.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with next node and response
        """
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        
        

        last_message = state.get("messages", [])[-1] if state.get("messages") else None
        agent_responded = (
            last_message 
            and hasattr(last_message, "name") 
            and last_message.name 
            and last_message.name.endswith("_agent")
        )
        
        if agent_responded:
            goto = "FINISH"
        else:
            response = llm.with_structured_output(Router).invoke(messages)
            goto = response.next
            
            if goto not in options:
                goto = "FINISH"
        
        update_dict = {"next": goto}
        
        if goto == "FINISH":
            final_response = state.get("response")
            
            if not final_response:
                messages = state.get("messages", [])
                for msg in reversed(messages):
                    if hasattr(msg, "content") and msg.content:
                        final_response = msg.content
                        break
            
            if final_response:
                update_dict["response"] = final_response
        
        return update_dict

    return supervisor_node


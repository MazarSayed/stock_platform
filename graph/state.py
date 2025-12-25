
from langgraph.graph import MessagesState
from pydantic import Field
from langchain_core.messages import BaseMessage


class State(MessagesState):
    """State for the supervisor node. MessagesState already provides the messages field."""
    next: str
    messages: list[BaseMessage] = Field(default_factory=list, description="Message to the user and assistant in a list of messages")
    response: str = Field(description="Response to the user")

   
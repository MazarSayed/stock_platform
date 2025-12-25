"""FastAPI application for LangGraph-based stock trading platform agent."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pathlib import Path
import sys
import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import logger and models (after path setup)
from utils.utils import logger
from models.models import ChatMessage, ChatRequest, ChatResponse
from guardrails.input_guardrails import input_guardrails
from guardrails.output_guardrails import output_guardrails

load_dotenv()

from langfuse.langchain import CallbackHandler
from graph.chatgrapgh import research_graph


app = FastAPI(
    title="Stock Trading Platform Assistant API",
    description="API for interacting with LangGraph-based stock trading platform agent",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage (use Redis/DB in production)
sessions = {}


@app.get("/")
async def root():
    return {"message": "Stock Trading Platform Assistant API", "version": "1.0.0"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a chat message to the agent.
    
    Args:
        request: Chat request with message and optional session_id
    
    Returns:
        Chat response with agent's reply
    """
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        # INPUT GUARDRAILS: Validate user input
        input_check = input_guardrails.validate_input(request.message)
        if not input_check.passed:
            logger.warning(f"Input guardrail failed: {input_check.reason}")
            return ChatResponse(
                response=f"I'm sorry, but I can only help with questions related to the stock trading platform. {input_check.reason}",
                session_id=session_id,
                agent="guardrail",
                messages=[]
            )
        
        # Use sanitized input if available
        user_message = input_check.sanitized_output or request.message
        
        if session_id not in sessions:
            sessions[session_id] = {
                "messages": [],
                "thread_id": str(uuid.uuid4())
            }
        
        session = sessions[session_id]
        config = {"configurable": {"thread_id": session["thread_id"]}}
        
        # Create Langfuse handler with session_id for this request
        try:
            handler = CallbackHandler(session_id=session_id)
            config["callbacks"] = [handler]
        except Exception:
            pass  # Langfuse is optional
        
        # Get existing messages or start fresh
        current_messages = [HumanMessage(content=user_message)]
        try:
            existing_state = await research_graph.aget_state(config)
            if existing_state and existing_state.values:
                existing_messages = existing_state.values.get("messages", [])
                current_messages = existing_messages + [HumanMessage(content=user_message)]
        except Exception:
            pass  # Start fresh if state retrieval fails
        
        # Invoke graph
        initial_state = {
            "messages": current_messages,
            "next": "supervisor",
            "response": ""
        }
        result = await research_graph.ainvoke(initial_state, config)
        
        # Extract response
        response_text = result.get("response", "")
        agent_used = None
        
        if not response_text and result.get("messages"):
            last_msg = result["messages"][-1]
            if hasattr(last_msg, "content"):
                response_text = last_msg.content
            if hasattr(last_msg, "name") and last_msg.name:
                agent_used = last_msg.name
        
        response_text = response_text or "I apologize, but I couldn't generate a response. Please try rephrasing your question."
        
        # OUTPUT GUARDRAILS: Validate and sanitize response
        output_check = output_guardrails.validate_output(response_text)
        
        if output_check.blocked:
            logger.warning(f"Response blocked: {output_check.reason}")
            response_text = "I apologize, but I cannot provide that information. Please ask about stock trading platform features, account management, or trading operations."
        elif output_check.sanitized_output:
            response_text = output_check.sanitized_output
            logger.info("Response sanitized before returning to user")
        
        # Update session messages
        session["messages"].append(ChatMessage(role="user", content=request.message))
        session["messages"].append(ChatMessage(role="assistant", content=response_text, agent=agent_used or "supervisor"))
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            agent=agent_used or "supervisor",
            messages=session["messages"]
        )
        
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


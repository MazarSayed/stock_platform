"""FastAPI application for LangGraph-based stock trading platform agent.

This module provides the REST API interface for the stock trading platform.
It handles chat requests, manages sessions, integrates with Langfuse for
observability, and invokes the LangGraph workflow to process user queries.

Endpoints:
    GET /: Health check
    POST /chat: Send messages to the agent system
    GET /chat/{session_id}: Retrieve chat history for a session
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pathlib import Path
import sys
import uuid
from dotenv import load_dotenv

# Load environment variables FIRST before any imports that need them
load_dotenv()

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from langchain_core.messages import HumanMessage
from src.utils.utils import logger
from models.models import ChatMessage, ChatRequest, ChatResponse
from langfuse.langchain import CallbackHandler
from src.graph.chatgrapgh import research_graph
from src.state import StateDB
from guardrails import input_guardrails, output_guardrails


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

# SQLite database for state persistence
state_db = StateDB(db_path="data/db/state.db")


@app.get("/")
async def root():
    """Health check endpoint.
    
    Returns:
        API information
    """
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
        
        # Validate input with guardrails
        input_validation = input_guardrails.validate_input(request.message)
        if not input_validation.passed:
            logger.warning(f"Input validation failed for session {session_id}: {input_validation.reason}")
            raise HTTPException(
                status_code=400,
                detail=input_validation.reason or "Input validation failed"
            )
        
        # Use sanitized input if available
        user_message = input_validation.sanitized_output or request.message
        
        # Get or create thread_id for this session
        thread_id = state_db.get_thread_id(session_id)
        if not thread_id:
            thread_id = str(uuid.uuid4())
            state_db.create_session(session_id, thread_id)
            logger.info(f"Created new session {session_id} with thread {thread_id}")
        else:
            state_db.update_session_timestamp(session_id)
        
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            handler = CallbackHandler()
            config["callbacks"] = [handler]
            config["metadata"] = {"langfuse_session_id": session_id}
            logger.info(f"Langfuse handler initialized for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse: {e}", exc_info=True)
        
        # Get existing messages or start fresh
        current_messages = [HumanMessage(content=user_message)]
        try:
            existing_state = await research_graph.aget_state(config)
            if existing_state and existing_state.values:
                existing_messages = existing_state.values.get("messages", [])
                current_messages = existing_messages + [HumanMessage(content=user_message)]
        except (KeyError, AttributeError, ValueError) as e:
            logger.debug(f"No existing state found or invalid state: {e}")
        
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
            # Only use agent messages, not user messages
            for msg in reversed(result["messages"]):
                if hasattr(msg, "name") and msg.name and msg.name.endswith("_agent"):
                    if hasattr(msg, "content"):
                        response_text = msg.content
                        agent_used = msg.name
                        break
        
        if not response_text:
            response_text = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
        
        # Validate output with guardrails
        output_validation = output_guardrails.validate_output(response_text)
        if not output_validation.passed:
            if output_validation.blocked:
                logger.warning(f"Output blocked for session {session_id}: {output_validation.reason}")
                response_text = "I apologize, but I cannot provide that response due to security policies."
            elif output_validation.sanitized_output:
                logger.info(f"Output sanitized for session {session_id}")
                response_text = output_validation.sanitized_output
        
        # Save messages to database
        state_db.add_message(session_id, "user", user_message)
        state_db.add_message(session_id, "assistant", response_text, agent_used or "supervisor")
        
        # Get all messages for response
        db_messages = state_db.get_messages(session_id)
        chat_messages = [
            ChatMessage(role=msg["role"], content=msg["content"], agent=msg.get("agent"))
            for msg in db_messages
        ]
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            agent=agent_used or "supervisor",
            messages=chat_messages
        )
        
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


@app.get("/chat/{session_id}", response_model=List[ChatMessage])
async def get_chat_history(session_id: str):
    """
    Get chat history for a session.
    
    Args:
        session_id: Session identifier
    
    Returns:
        List of chat messages
    """
    thread_id = state_db.get_thread_id(session_id)
    if not thread_id:
        raise HTTPException(status_code=404, detail="Session not found")
    
    db_messages = state_db.get_messages(session_id)
    return [
        ChatMessage(role=msg["role"], content=msg["content"], agent=msg.get("agent"))
        for msg in db_messages
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
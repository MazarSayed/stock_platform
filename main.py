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
            pass
        
        # Get existing messages or start fresh
        current_messages = [HumanMessage(content=request.message)]
        try:
            existing_state = await research_graph.aget_state(config)
            if existing_state and existing_state.values:
                existing_messages = existing_state.values.get("messages", [])
                current_messages = existing_messages + [HumanMessage(content=request.message)]
        except Exception:
            pass
        
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
        
        if not response_text:
            response_text = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
        
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


@app.get("/chat/{session_id}", response_model=List[ChatMessage])
async def get_chat_history(session_id: str):
    """
    Get chat history for a session.
    
    Args:
        session_id: Session identifier
    
    Returns:
        List of chat messages
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return sessions[session_id]["messages"]




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


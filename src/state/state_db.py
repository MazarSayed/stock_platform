"""SQLite database implementation for state persistence.

This module provides the StateDB class for managing session state and messages.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any


class StateDB:
    """SQLite database for storing session state and messages."""
    
    def __init__(self, db_path: str = "data/db/state.db"):
        """Initialize the state database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize database tables."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL UNIQUE,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get_connection(self):
        """Get a database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def create_session(self, session_id: str, thread_id: str) -> bool:
        """Create a new session.
        
        Args:
            session_id: Unique session identifier
            thread_id: Unique thread identifier for LangGraph
            
        Returns:
            True if session was created, False if it already exists
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (session_id, thread_id, updated_at)
                VALUES (?, ?, ?)
            """, (session_id, thread_id, datetime.now()))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            conn.close()
            return False
    
    def get_thread_id(self, session_id: str) -> Optional[str]:
        """Get thread_id for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Thread ID if session exists, None otherwise
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT thread_id FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        conn.close()
        return row["thread_id"] if row else None
    
    def update_session_timestamp(self, session_id: str):
        """Update the updated_at timestamp for a session.
        
        Args:
            session_id: Session identifier
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", 
                      (datetime.now(), session_id))
        conn.commit()
        conn.close()
    
    def add_message(self, session_id: str, role: str, content: str, agent: Optional[str] = None):
        """Add a message to a session.
        
        Args:
            session_id: Session identifier
            role: Message role ('user' or 'assistant')
            content: Message content
            agent: Optional agent name that handled the message
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO messages (session_id, role, content, agent, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, role, content, agent, datetime.now()))
        conn.commit()
        conn.close()
    
    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of message dictionaries
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, content, agent, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY created_at ASC
        """, (session_id,))
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "agent": row["agent"],
                "created_at": row["created_at"]
            }
            for row in rows
        ]

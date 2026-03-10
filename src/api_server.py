"""
FastAPI wrapper for RAG System
ใช้สำหรับ integrate กับ Langflow หรือระบบอื่นๆ

Install:
    pip install fastapi uvicorn

Run:
    uvicorn src.api_server:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import yaml
import uuid
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from src.core.chat_engine import ChatEngine


app = FastAPI(
    title="RAG System API",
    description="Internal RAG system for organizational knowledge",
    version="1.0.0"
)

# CORS for Langflow
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify Langflow URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files for dashboard
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

static_dir = os.path.join(os.path.dirname(__file__), "api", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Load config and initialize engine
with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

engine = ChatEngine(config)
print("[API] ChatEngine initialized")

# Initialize Escalation Handler
from src.utils.escalation_handler import EscalationHandler
escalation_handler = EscalationHandler(config)
print("[API] EscalationHandler initialized")

# Initialize Feedback Handler
from src.utils.feedback_handler import FeedbackHandler
feedback_handler = FeedbackHandler(log_dir="logs")
print("[API] FeedbackHandler initialized")


# Request/Response models
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    bypass_cache: Optional[bool] = False


class QueryResponse(BaseModel):
    answer: str
    route: str
    latency_ms: float
    sources: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    teams_count: int
    positions_count: int


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "RAG System API",
        "docs": "/docs",
        "health": "/health",
        "dashboard": "/dashboard"
    }


@app.get("/dashboard")
async def dashboard():
    """Serve monitoring dashboard"""
    static_dir = os.path.join(os.path.dirname(__file__), "api", "static")
    dashboard_path = os.path.join(static_dir, "dashboard.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path)
    return {"error": "Dashboard not found"}


@app.get("/feedback-demo")
async def feedback_demo():
    """Serve feedback demo page"""
    static_dir = os.path.join(os.path.dirname(__file__), "api", "static")
    feedback_path = os.path.join(static_dir, "feedback_demo.html")
    if os.path.exists(feedback_path):
        return FileResponse(feedback_path)
    return {"error": "Feedback demo not found"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        teams_count=len(engine.directory_handler.team_index),
        positions_count=len(engine.directory_handler.position_index)
    )


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a RAG query
    
    Example:
        POST /query
        {
            "query": "รายชื่อทีมทั้งหมด",
            "session_id": "user123",
            "bypass_cache": false
        }
    """
    try:
        # If no session_id provided (e.g. from Langflow without session config),
        # generate a unique per-request ID to prevent context bleeding between conversations.
        # Trade-off: each request is fully isolated (no cross-request memory from this caller).
        # To enable multi-turn context, the caller must explicitly send session_id.
        effective_session_id = request.session_id or f"anon_{uuid.uuid4().hex[:12]}"
        
        # Process query
        result = engine.process(
            request.query,
            session_id=effective_session_id
        )
        
        # Extract latency
        latency_ms = 0.0
        if "latencies" in result:
            total_latency = sum(result["latencies"].values())
            latency_ms = total_latency
        
        # Extract sources (if any)
        sources = []
        if "hits" in result and result["hits"]:
            sources = [
                hit.get("metadata", {}).get("url") or hit.get("metadata", {}).get("source")
                for hit in result["hits"]
                if hit.get("metadata")
            ]
            sources = [s for s in sources if s]  # Remove None
        
        return QueryResponse(
            answer=result.get("answer", "ไม่สามารถตอบได้"),
            route=result.get("route", "unknown"),
            latency_ms=latency_ms,
            sources=sources if sources else None,
            metadata={
                "intent": result.get("intent"),
                "latencies": result.get("latencies")
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/teams", response_model=Dict[str, Any])
async def list_teams():
    """List all teams in the system"""
    teams = list(engine.directory_handler.team_index.keys())
    return {
        "teams": sorted(teams),
        "count": len(teams)
    }


@app.get("/stats", response_model=Dict[str, Any])
async def get_stats():
    """Get system statistics for monitoring dashboard"""
    import json
    import os
    
    # Load dashboard stats
    stats_file = os.path.join("logs", "dashboard_stats.json")
    stats = {}
    if os.path.exists(stats_file):
        try:
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)
        except:
            pass
    
    # Load recent queries
    query_log = os.path.join("logs", "query_metrics.jsonl")
    recent_queries = []
    if os.path.exists(query_log):
        try:
            with open(query_log, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # Get last 10 queries
                for line in lines[-10:]:
                    try:
                        entry = json.loads(line)
                        recent_queries.append(entry)
                    except:
                        pass
        except:
            pass
    
    return {
        "teams": len(engine.directory_handler.team_index),
        "positions": len(engine.directory_handler.position_index),
        "vector_store": {
            "initialized": engine.vs is not None
        },
        "query_stats": stats,
        "recent_queries": recent_queries
    }


@app.post("/escalate")
async def escalate(session_id: str = "default"):
    """Manual escalation endpoint - user requests human help"""
    response = escalation_handler.get_escalation_response(reason="manual")
    return response


@app.post("/feedback")
async def submit_feedback(request: Dict[str, Any]):
    """
    Submit user feedback for a query.
    
    Body:
        query_id: str - Unique query identifier
        session_id: str - User session ID
        query: str - Original query text
        answer: str - Answer preview
        rating: "like" | "dislike"
        comment: str (optional) - User comment
        route: str (optional) - Route taken
    """
    try:
        query_id = request.get("query_id")
        session_id = request.get("session_id", "default")
        query = request.get("query", "")
        answer = request.get("answer", "")
        rating = request.get("rating")
        comment = request.get("comment", "")
        route = request.get("route", "unknown")
        
        # Validate required fields
        if not query_id or not rating:
            return {"success": False, "message": "Missing required fields: query_id, rating"}
        
        # Save feedback
        success = feedback_handler.save_feedback(
            session_id=session_id,
            query_id=query_id,
            query=query,
            answer_preview=answer,
            rating=rating,
            comment=comment,
            route=route
        )
        
        if success:
            return {"success": True, "message": "Feedback saved successfully"}
        else:
            return {"success": False, "message": "Failed to save feedback"}
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}


@app.get("/feedback/stats")
async def get_feedback_stats():
    """Get aggregate feedback statistics"""
    stats = feedback_handler.get_feedback_stats()
    return stats


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

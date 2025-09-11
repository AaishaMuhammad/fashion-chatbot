from typing import Optional
from uuid import uuid4
from contextlib import asynccontextmanager

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, FastAPI
from pydantic import BaseModel

from src.recommender_graph.graph import create_recommender_graph

router = APIRouter(prefix="/recommend", tags=["Recommender"])

graph_app = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph_app
    graph_app = create_recommender_graph()
    yield

class QuestionRequest(BaseModel):

    question: str

def get_or_create_thread_id(request: Request, thread_id: Optional[str] = Cookie(default=None)) -> str:

    if thread_id is None:
        new_id = str(uuid4())
        request.state.new_thread_id = new_id
        return new_id

    return thread_id

@router.post("/", response_model=dict)
def get_chat_response(
    request: Request,
    response: Response,
    body: QuestionRequest,
    thread_id: str = Depends(get_or_create_thread_id)
): 
    
    try: 
        if hasattr(request, "state") and hasattr(request.state, "new_thread_id"):
            response.set_cookie("thread_id", request.state.new_thread_id)

        config = {"configurable": {"thread_id":thread_id}}
        result = graph_app.invoke({"query": body.question}, config=config)

        recommendation = result.get("recommendation", "No recommendation found.")
        
        return {
            "question": body.question,
            "thread_id": thread_id,
            "answer": recommendation
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    



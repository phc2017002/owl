# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uuid
import asyncio
from datetime import datetime
import os
from dotenv import load_dotenv

# Import CAMEL components
from camel.models import ModelFactory
from camel.toolkits import (
    CodeExecutionToolkit,
    ExcelToolkit,
    ImageAnalysisToolkit,
    SearchToolkit,
    VideoAnalysisToolkit,
    BrowserToolkit,
)
from camel.types import ModelPlatformType, ModelType
from camel.logger import set_log_level

# Import custom modules
from utils import OwlRolePlaying, run_society, DocumentProcessingToolkit

# Load environment variables
load_dotenv()

# Set up logging
set_log_level(level="INFO")

# Initialize FastAPI app
app = FastAPI(title="CAMEL-AI Agent API", 
              description="API for interacting with CAMEL-AI agent society",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify this in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for tasks
tasks_store = {}

# Models for request and response
class QuestionRequest(BaseModel):
    question: str
    timeout: Optional[int] = 120  # Default timeout of 2 minutes
    parameters: Optional[Dict[str, Any]] = None

class TaskResponse(BaseModel):
    task_id: str
    status: str
    created_at: str

class TaskStatus(BaseModel):
    task_id: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    answer: Optional[str] = None
    chat_history: Optional[List[Dict[str, Any]]] = None
    token_count: Optional[int] = None
    error: Optional[str] = None

# Initialize model instances
def initialize_models():
    """Initialize and return models for different components."""
    return {
        "user": ModelFactory.create(
            model_platform=ModelPlatformType.QWEN,
            model_type=ModelType.QWEN_MAX,
            model_config_dict={"temperature": 0},
        ),
        "assistant": ModelFactory.create(
            model_platform=ModelPlatformType.QWEN,
            model_type=ModelType.QWEN_MAX,
            model_config_dict={"temperature": 0},
        ),
        "web": ModelFactory.create(
            model_platform=ModelPlatformType.QWEN,
            model_type=ModelType.QWEN_MAX,
            model_config_dict={"temperature": 0},
        ),
        "planning": ModelFactory.create(
            model_platform=ModelPlatformType.QWEN,
            model_type=ModelType.QWEN_MAX,
            model_config_dict={"temperature": 0},
        ),
        "video": ModelFactory.create(
            model_platform=ModelPlatformType.QWEN,
            model_type=ModelType.QWEN_MAX,
            model_config_dict={"temperature": 0},
        ),
        "image": ModelFactory.create(
            model_platform=ModelPlatformType.QWEN,
            model_type=ModelType.QWEN_MAX,
            model_config_dict={"temperature": 0},
        ),
        "document": ModelFactory.create(
            model_platform=ModelPlatformType.QWEN,
            model_type=ModelType.QWEN_MAX,
            model_config_dict={"temperature": 0},
        ),
    }

# Initialize toolkits
def initialize_toolkits(models):
    """Initialize and return toolkits with provided models."""
    return [
        *BrowserToolkit(
            headless=True,  # Use headless mode for server deployment
            web_agent_model=models["web"],
            planning_agent_model=models["planning"],
            output_language="English",
        ).get_tools(),
        *VideoAnalysisToolkit(model=models["video"]).get_tools(),
        *CodeExecutionToolkit(sandbox="subprocess", verbose=True).get_tools(),
        *ImageAnalysisToolkit(model=models["image"]).get_tools(),
        SearchToolkit().search_duckduckgo,
        SearchToolkit().search_google,  # Comment this out if you don't have google search
        SearchToolkit().search_wiki,
        *ExcelToolkit().get_tools(),
        *DocumentProcessingToolkit(model=models["document"]).get_tools(),
    ]

# Construct a society of agents
def construct_society(question: str, models=None, tools=None):
    """
    Construct a society of agents based on the given question.

    Args:
        question (str): The task or question to be addressed by the society.
        models (Dict): Pre-initialized models (optional).
        tools (List): Pre-initialized tools (optional).

    Returns:
        OwlRolePlaying: A configured society of agents ready to address the question.
    """
    if models is None:
        models = initialize_models()
    
    if tools is None:
        tools = initialize_toolkits(models)

    # Configure agent roles and parameters
    user_agent_kwargs = {"model": models["user"]}
    assistant_agent_kwargs = {"model": models["assistant"], "tools": tools}

    # Configure task parameters
    task_kwargs = {
        "task_prompt": question,
        "with_task_specify": False,
    }

    # Create and return the society
    society = OwlRolePlaying(
        **task_kwargs,
        user_role_name="user",
        user_agent_kwargs=user_agent_kwargs,
        assistant_role_name="assistant",
        assistant_agent_kwargs=assistant_agent_kwargs,
        output_language="English",
    )

    return society

# Background task to process a question
async def process_question(task_id: str, question: str, timeout: int):
    """
    Process a question in the background.
    
    Args:
        task_id (str): The ID of the task.
        question (str): The question to be processed.
        timeout (int): The timeout in seconds.
    """
    try:
        # Update task status to processing
        tasks_store[task_id]["status"] = "processing"
        
        # Initialize models and tools (can be optimized to be shared across requests)
        models = initialize_models()
        tools = initialize_toolkits(models)
        
        # Construct the society
        society = construct_society(question, models, tools)
        
        # Run the society with timeout
        answer, chat_history, token_count = await asyncio.wait_for(
            asyncio.to_thread(run_society, society),
            timeout=timeout
        )
        
        # Update task with results
        tasks_store[task_id].update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "answer": answer,
            "chat_history": chat_history,
            "token_count": token_count
        })
        
    except asyncio.TimeoutError:
        tasks_store[task_id].update({
            "status": "timeout",
            "completed_at": datetime.now().isoformat(),
            "error": f"Task execution exceeded the timeout of {timeout} seconds"
        })
    except Exception as e:
        tasks_store[task_id].update({
            "status": "error",
            "completed_at": datetime.now().isoformat(),
            "error": str(e)
        })

# API endpoint to submit a question
@app.post("/api/questions", response_model=TaskResponse)
async def submit_question(request: QuestionRequest, background_tasks: BackgroundTasks):
    """
    Submit a question to be processed by the agent society.
    
    Args:
        request (QuestionRequest): The question request containing the question and optional parameters.
        background_tasks (BackgroundTasks): FastAPI background tasks manager.
        
    Returns:
        TaskResponse: The response containing the task ID and status.
    """
    # Generate a unique task ID
    task_id = str(uuid.uuid4())
    
    # Initialize task in the store
    tasks_store[task_id] = {
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "question": request.question,
        "parameters": request.parameters
    }
    
    # Add the task to background tasks
    background_tasks.add_task(
        process_question, 
        task_id=task_id, 
        question=request.question,
        timeout=request.timeout
    )
    
    # Return the task ID and status
    return TaskResponse(
        task_id=task_id,
        status="pending",
        created_at=tasks_store[task_id]["created_at"]
    )

# API endpoint to check the status of a task
@app.get("/api/tasks/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """
    Get the status of a task by its ID.
    
    Args:
        task_id (str): The ID of the task.
        
    Returns:
        TaskStatus: The status of the task.
    """
    if task_id not in tasks_store:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks_store[task_id]
    
    return TaskStatus(
        task_id=task_id,
        status=task["status"],
        created_at=task["created_at"],
        completed_at=task.get("completed_at"),
        answer=task.get("answer"),
        chat_history=task.get("chat_history"),
        token_count=task.get("token_count"),
        error=task.get("error")
    )

# List all tasks
@app.get("/api/tasks", response_model=List[TaskStatus])
async def list_tasks():
    """
    List all tasks.
    
    Returns:
        List[TaskStatus]: A list of all tasks.
    """
    return [
        TaskStatus(
            task_id=task_id,
            status=task["status"],
            created_at=task["created_at"],
            completed_at=task.get("completed_at"),
            answer=task.get("answer"),
            chat_history=task.get("chat_history"),
            token_count=task.get("token_count"),
            error=task.get("error")
        )
        for task_id, task in tasks_store.items()
    ]

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Dict: A dictionary containing the status of the API.
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Main function to run the app
if __name__ == "__main__":
    import uvicorn
    # Use environment variables for host and port if available
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
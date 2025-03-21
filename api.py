from fastapi import FastAPI, HTTPException, Depends, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import pathlib
import sys
import time
import os
import uuid
import asyncio
from datetime import datetime, timedelta
import logging
from logging.handlers import RotatingFileHandler
import json

# Rate limiting and caching
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
import redis.asyncio as redis

# Environment variables
from dotenv import load_dotenv

# Health checks
from fastapi_health import health

# CAMEL imports
from camel.models import ModelFactory
from camel.toolkits import (
    AudioAnalysisToolkit,
    CodeExecutionToolkit,
    ExcelToolkit,
    ImageAnalysisToolkit,
    SearchToolkit,
    VideoAnalysisToolkit,
    BrowserToolkit,
    FileWriteToolkit,
)
from camel.types import ModelPlatformType, ModelType
from camel.logger import set_log_level
from camel.societies import RolePlaying
from owl.utils import run_society, DocumentProcessingToolkit

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("camel_api")

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure file handler for rotating logs
file_handler = RotatingFileHandler(
    "logs/api.log", maxBytes=10 * 1024 * 1024, backupCount=5
)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

# Load environment variables
base_dir = pathlib.Path(__file__).parent.parent
env_path = base_dir / ".env"
load_dotenv(dotenv_path=str(env_path))

# Set CAMEL logging level from environment variable or default to INFO
camel_log_level = os.getenv("CAMEL_LOG_LEVEL", "INFO")
set_log_level(level=camel_log_level)

# Get Redis configuration from environment variables
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# API settings
API_RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", "10"))  # requests per minute
CACHE_EXPIRATION = int(os.getenv("CACHE_EXPIRATION", "86400"))  # in seconds
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

# Initialize FastAPI app
app = FastAPI(
    title="CAMEL API",
    description="API for interacting with CAMEL agent societies",
    version="1.0.0",
    debug=DEBUG_MODE,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    # Add request ID to request state
    request.state.request_id = request_id
    
    # Log request details
    logger.info(
        f"Request {request_id}: {request.method} {request.url.path} - "
        f"Client: {request.client.host if request.client else 'Unknown'}"
    )
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Add custom headers to response
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log response details
    logger.info(
        f"Response {request_id}: Status {response.status_code} - "
        f"Process time: {process_time:.4f}s"
    )
    
    return response

# Define request models with validation
class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000, description="The question to ask the agent society")
    cache_key: Optional[str] = Field(default=None, description="Optional custom cache key")
    model_config: Optional[Dict[str, Any]] = Field(default=None, description="Optional model configuration")

class QueryResponse(BaseModel):
    request_id: str
    answer: str
    tokens_used: int
    processing_time: float
    cached: bool = False
    timestamp: datetime

# Model configuration cache
model_cache = {}

# Function to construct the agent society
def construct_society(question: str, model_config: Optional[Dict[str, Any]] = None) -> RolePlaying:
    try:
        # Use cached model configs or default ones
        if not model_config:
            # Check if we have cached models
            if "default" in model_cache:
                logger.info("Using cached model configuration")
                models = model_cache["default"]
            else:
                logger.info("Creating new default model configuration")
                models = create_default_models()
                model_cache["default"] = models
        else:
            # Create custom models based on config
            logger.info("Creating custom model configuration")
            models = create_custom_models(model_config)
            
        # Create tools with the specified models
        tools = create_tools(models)
        
        # Create and return the society
        user_agent_kwargs = {"model": models["user"]}
        assistant_agent_kwargs = {"model": models["assistant"], "tools": tools}
        
        society = RolePlaying(
            task_prompt=question,
            with_task_specify=False,
            user_role_name="user",
            user_agent_kwargs=user_agent_kwargs,
            assistant_role_name="assistant",
            assistant_agent_kwargs=assistant_agent_kwargs,
        )
        
        return society
    except Exception as e:
        logger.error(f"Error constructing society: {str(e)}", exc_info=True)
        raise

def create_default_models():
    """Create default models configuration"""
    model_type = os.getenv("DEFAULT_MODEL_TYPE", "GPT_4O_MINI")
    model_platform = os.getenv("DEFAULT_MODEL_PLATFORM", "OPENAI")
    
    models = {
        "user": ModelFactory.create(
            model_platform=getattr(ModelPlatformType, model_platform),
            model_type=getattr(ModelType, model_type),
            model_config_dict={"temperature": 0},
        ),
        "assistant": ModelFactory.create(
            model_platform=getattr(ModelPlatformType, model_platform),
            model_type=getattr(ModelType, model_type),
            model_config_dict={"temperature": 0},
        ),
        "browsing": ModelFactory.create(
            model_platform=getattr(ModelPlatformType, model_platform),
            model_type=getattr(ModelType, model_type),
            model_config_dict={"temperature": 0},
        ),
        "planning": ModelFactory.create(
            model_platform=getattr(ModelPlatformType, model_platform),
            model_type=getattr(ModelType, model_type),
            model_config_dict={"temperature": 0},
        ),
        "video": ModelFactory.create(
            model_platform=getattr(ModelPlatformType, model_platform),
            model_type=getattr(ModelType, model_type),
            model_config_dict={"temperature": 0},
        ),
        "image": ModelFactory.create(
            model_platform=getattr(ModelPlatformType, model_platform),
            model_type=getattr(ModelType, model_type),
            model_config_dict={"temperature": 0},
        ),
        "document": ModelFactory.create(
            model_platform=getattr(ModelPlatformType, model_platform),
            model_type=getattr(ModelType, model_type),
            model_config_dict={"temperature": 0},
        ),
    }
    
    return models

def create_custom_models(model_config: Dict[str, Any]):
    """Create custom models based on provided configuration"""
    models = {}
    
    for role, config in model_config.items():
        models[role] = ModelFactory.create(
            model_platform=getattr(ModelPlatformType, config.get("platform", "OPENAI")),
            model_type=getattr(ModelType, config.get("type", "GPT_4O_MINI")),
            model_config_dict=config.get("config", {"temperature": 0}),
        )
    
    # Ensure all required roles are defined
    required_roles = ["user", "assistant", "browsing", "planning", "video", "image", "document"]
    for role in required_roles:
        if role not in models:
            # Use default for missing roles
            models[role] = create_default_models()[role]
    
    return models

def create_tools(models: Dict[str, Any]):
    """Create tools with the specified models"""
    # Get configuration from environment variables
    browser_headless = os.getenv("BROWSER_HEADLESS", "True").lower() == "true"
    code_sandbox = os.getenv("CODE_SANDBOX", "subprocess")
    code_verbose = os.getenv("CODE_VERBOSE", "True").lower() == "true"
    file_output_dir = os.getenv("FILE_OUTPUT_DIR", "./output")
    
    # Create output directory if it doesn't exist
    os.makedirs(file_output_dir, exist_ok=True)
    
    # Create and return tools
    tools = [
        *BrowserToolkit(
            headless=browser_headless,
            web_agent_model=models["browsing"],
            planning_agent_model=models["planning"],
        ).get_tools(),
        *VideoAnalysisToolkit(model=models["video"]).get_tools(),
        *AudioAnalysisToolkit().get_tools(),
        *CodeExecutionToolkit(sandbox=code_sandbox, verbose=code_verbose).get_tools(),
        *ImageAnalysisToolkit(model=models["image"]).get_tools(),
        SearchToolkit().search_duckduckgo,
        SearchToolkit().search_google,
        SearchToolkit().search_wiki,
        *ExcelToolkit().get_tools(),
        *DocumentProcessingToolkit(model=models["document"]).get_tools(),
        *FileWriteToolkit(output_dir=file_output_dir).get_tools(),
    ]
    
    return tools

@app.on_event("startup")
async def startup():
    # Initialize Redis for rate limiting
    redis_url = f"redis://{':' + REDIS_PASSWORD + '@' if REDIS_PASSWORD else ''}{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    
    try:
        # Initialize Redis clients for rate limiting and caching
        redis_instance = redis.from_url(redis_url, decode_responses=True)
        await FastAPILimiter.init(redis_instance)
        
        # Initialize cache
        redis_cache = redis.from_url(redis_url, encoding="utf8")
        FastAPICache.init(RedisBackend(redis_cache), prefix="camel_api_cache")
        
        logger.info("Successfully connected to Redis for rate limiting and caching")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {str(e)}", exc_info=True)
        logger.warning("Running without rate limiting and caching")

@app.on_event("shutdown")
async def shutdown():
    # Clean up resources
    logger.info("Shutting down API")
    
    # Clear model cache
    model_cache.clear()

def is_healthy():
    # Implement more sophisticated health checks if needed
    return {"status": "healthy"}

app.add_api_route("/health", health([is_healthy]), tags=["Health"])

@app.get("/", tags=["Root"])
async def root():
    return {
        "status": "online",
        "service": "CAMEL API",
        "version": "1.0.0",
        "documentation": "/docs"
    }

@app.post("/query", response_model=QueryResponse, dependencies=[Depends(RateLimiter(times=API_RATE_LIMIT, minutes=1))])
@cache(expire=CACHE_EXPIRATION, namespace="query_cache")
async def query(request: Request, query_request: QueryRequest):
    start_time = time.time()
    request_id = request.state.request_id
    cached = False
    
    try:
        # Generate cache key based on request or use custom key
        cache_key = query_request.cache_key or query_request.question
        
        # Check if response is in cache (even though @cache decorator handles this,
        # we want to set the 'cached' flag in the response)
        cache_backend = FastAPICache.get_backend()
        cached_response = await cache_backend.get(f"query_cache:{cache_key}")
        
        if cached_response:
            logger.info(f"Request {request_id}: Cache hit for key {cache_key}")
            cached_data = json.loads(cached_response)
            cached_data["request_id"] = request_id
            cached_data["cached"] = True
            cached_data["timestamp"] = datetime.now()
            return QueryResponse(**cached_data)
        
        logger.info(f"Request {request_id}: Processing question - {query_request.question[:50]}...")
        
        # Run the synchronous construct_society in a thread pool
        loop = asyncio.get_running_loop()
        society = await loop.run_in_executor(
            None, 
            lambda: construct_society(query_request.question, query_request.model_config)
        )

        # Run the synchronous run_society in a thread pool
        answer, chat_history, token_count = await loop.run_in_executor(None, run_society, society)

        # Calculate processing time
        process_time = time.time() - start_time
        
        response_data = {
            "request_id": request_id,
            "answer": answer,
            "tokens_used": token_count,
            "processing_time": process_time,
            "cached": cached,
            "timestamp": datetime.now()
        }
        
        logger.info(
            f"Request {request_id}: Successfully processed query in {process_time:.4f}s, "
            f"used {token_count} tokens"
        )
        
        return QueryResponse(**response_data)
    except Exception as e:
        # Log the exception
        logger.error(f"Request {request_id}: Error processing query: {str(e)}", exc_info=True)
        
        # Raise HTTPException with error details
        error_detail = str(e) if DEBUG_MODE else "An error occurred processing your request"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail
        )

@app.get("/stats", tags=["Admin"])
async def get_stats():
    """Get API statistics"""
    # This could be enhanced to retrieve actual stats from Redis or other monitoring tools
    return {
        "uptime": "N/A",  # Would need an actual implementation
        "requests_processed": "N/A",
        "cache_hit_ratio": "N/A",
        "average_response_time": "N/A"
    }

@app.post("/cache/clear", tags=["Admin"])
async def clear_cache():
    """Clear the API cache"""
    try:
        await FastAPICache.clear()
        return {"status": "success", "message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear cache"
        )

@app.post("/models/refresh", tags=["Admin"])
async def refresh_models():
    """Refresh the model cache"""
    try:
        model_cache.clear()
        model_cache["default"] = create_default_models()
        return {"status": "success", "message": "Model cache refreshed successfully"}
    except Exception as e:
        logger.error(f"Error refreshing models: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh models"
        )

# Run the FastAPI server if the script is executed directly
if __name__ == "__main__":
    import uvicorn
    
    # Get server configuration from environment variables
    HOST = os.getenv("API_HOST", "0.0.0.0")
    PORT = int(os.getenv("API_PORT", "8000"))
    WORKERS = int(os.getenv("API_WORKERS", "1"))
    LOG_LEVEL = os.getenv("API_LOG_LEVEL", "info")
    
    logger.info(f"Starting server on {HOST}:{PORT} with {WORKERS} workers")
    
    uvicorn.run(
        "api:app",  # Assuming this file is named api.py
        host=HOST,
        port=PORT,
        workers=WORKERS,
        log_level=LOG_LEVEL,
        reload=DEBUG_MODE
    )
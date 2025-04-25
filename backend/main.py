import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel 
from openai import OpenAI 
from dotenv import load_dotenv 
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time                     # Added for timing
from contextlib import contextmanager # Added for context manager

load_dotenv()

app = FastAPI()

origins = [
    "http://localhost",        
    "http://localhost:5173",  
    "http://127.0.0.1",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       
    allow_credentials=True,     
    allow_methods=["GET", "POST"], 
    allow_headers=["*"],          
)

# --- Token Usage Tracking --- (Copied from user input)
@contextmanager
def track_token_usage():
    """Context manager to track OpenAI API token usage."""
    class TokenUsage:
        def __init__(self):
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.start_time = time.time()
            self.end_time = None
            self.cost = 0 # Note: Based on approximate rates

        def update(self, response_dict):
            """Update token counts from OpenAI API response dictionary."""
            # Expects response_dict to be the dictionary form of the API response
            usage = response_dict.get("usage", {})
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)

            # Approximate cost calculation (rates depend heavily on the actual model)
            prompt_cost = (self.prompt_tokens / 1000) * 0.0015  # Sample rate
            completion_cost = (self.completion_tokens / 1000) * 0.002 # Sample rate
            self.cost = prompt_cost + completion_cost

        def finish(self):
            self.end_time = time.time()

        def __str__(self):
            duration = round(self.end_time - self.start_time, 2) if self.end_time else 0
            return (
                f"--- Token Usage ---\n"
                f"  Prompt Tokens:     {self.prompt_tokens}\n"
                f"  Completion Tokens: {self.completion_tokens}\n"
                f"  Total Tokens:      {self.total_tokens}\n"
                f"  Est. Cost (USD):   ${self.cost:.6f}\n" # Emphasize this is an estimate
                f"  API Call Duration: {duration}s\n"
                f"-------------------"
            )

    token_usage = TokenUsage()
    try:
        yield token_usage # Provides the tracker object to the 'with' block
    finally:
        token_usage.finish()
        print(token_usage) # Print stats when exiting the context

# --- Prompt Templating System ---
@dataclass
class MessageTemplate:
    role: str
    template: str # String with placeholders like {user_input}

    def format(self, **kwargs):
        """Format the template string with provided key-value pairs."""
        # Returns a dictionary like {"role": "user", "content": "formatted text"}
        return {"role": self.role, "content": self.template.format(**kwargs)}

# Note: The PromptTemplate class wasn't directly used here to keep history handling simple,
# but it's available if you want more complex template structures later.
class PromptTemplate:
    def __init__(self, messages: List[MessageTemplate]):
        self.messages = messages

    def format_messages(self, **kwargs):
        """Format all MessageTemplates in the list."""
        return [message.format(**kwargs) for message in self.messages]

# --- Data Models (Pydantic) ---
# Define expected data structures for requests

# Structure for one chat message (like a Python dictionary)
# Keys: 'role', 'content' | Values: string
# Used in the list sent to OpenAI
class Message(BaseModel):
    role: str
    content: str

# Structure for data expected from frontend POST request to /api/chat
# Keys: 'message', 'history' | Values: string, list of Message objects
class ChatRequest(BaseModel):
    message: str
    # history: Optional -> might be missing from request
    # List[Message] -> if present, must be a list of Message items
    # = [] -> if missing, use empty list by default
    history: Optional[List[Message]] = []

# --- OpenAI Client Setup ---
api_key = os.getenv("OPENAI_API_KEY") 
if not api_key:
    print("Error: OPENAI_API_KEY not found. Did you create a .env file?")
    client = None 
else:
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        print(f"Error setting up OpenAI client: {e}")
        client = None 

# --- API Routes --- 
@app.get("/")
def read_root():
    # Return simple dictionary (FastAPI sends as JSON)
    return {"message": "GPT Wrapper Backend is running"}

# Define a template for the user message part of the prompt
user_prompt_template = MessageTemplate(role="user", template="{user_input}")

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Handle chat request, call OpenAI, return response."""
    if not client:
         raise HTTPException(status_code=500, detail="OpenAI client not set up. Check API key.")

    try:
        conversation_history = request.history or []

        # 1. Start with the system message (direct dictionary)
        system_message = {"role": "system", "content": "You are a helpful assistant."}
        messages_payload = [system_message]

        # 2. Add historical messages (converting Pydantic models back to dicts)
        messages_payload.extend([msg.model_dump() for msg in conversation_history])

        # 3. Format the *new* user message using the template
        formatted_user_message = user_prompt_template.format(user_input=request.message)
        # Add the formatted message dictionary to the end of the list
        messages_payload.append(formatted_user_message)

        # --- Call OpenAI API with Token Tracking ---
        print("--- Calling OpenAI API ---") # Indicate API call start
        ai_response_content = None
        with track_token_usage() as tracker:
            # Make the API call inside the context manager
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages_payload,
                max_tokens=150
            )
            # Update tracker with usage info from the response
            # Need to convert the Pydantic response model to a dictionary first
            response_dict = response.model_dump()
            tracker.update(response_dict)
            # Extract the actual text content
            ai_response_content = response.choices[0].message.content

        # Check if we got content before returning
        if ai_response_content is None:
             raise HTTPException(status_code=500, detail="Failed to get valid response content from OpenAI.")

        # Return only the text content to the frontend
        return {"response": ai_response_content}

    except Exception as e:
        # Enhanced error logging
        print(f"Error during OpenAI call or processing: {e}")
        # Check if the exception is already an HTTPException, otherwise create one
        if isinstance(e, HTTPException):
            raise e
        else:
            raise HTTPException(status_code=500, detail=f"Failed to get response from OpenAI: {str(e)}")

if __name__ == "__main__":
    import uvicorn 
    print("Starting backend server at http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True) 
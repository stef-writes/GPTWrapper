import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel 
from openai import OpenAI 
from dotenv import load_dotenv 
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

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

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages_payload,
            max_tokens=150
        )

        ai_response = response.choices[0].message.content
        return {"response": ai_response}

    except Exception as e:
        print(f"Error calling OpenAI: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get response from OpenAI: {str(e)}")

if __name__ == "__main__":
    import uvicorn 
    print("Starting backend server at http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True) 
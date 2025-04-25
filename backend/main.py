import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

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
    allow_headers=["*"]
)

# --- Pydantic Models ---
# Define the structure of the request body for the chat endpoint
class ChatRequest(BaseModel):
    message: str
    # Optional: Add history if you want to maintain conversation context
    # history: list[dict[str, str]] = []

# --- OpenAI Client Initialization ---
# It's good practice to initialize the client once.
# The API key is read securely from environment variables.
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    # In a real application, you might want more robust error handling
    # or a way to prompt the user if the key is missing.
    print("Error: OPENAI_API_KEY environment variable not set.")
    # Exit or raise an exception if the key is critical for startup
    # raise ValueError("OPENAI_API_KEY environment variable not set.")
    # For now, we'll allow the app to run but OpenAI calls will fail.
    client = None
else:
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        client = None

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "GPT Wrapper Backend is running"}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Receives a message from the frontend, sends it to OpenAI GPT-4,
    and returns the AI's response.
    """
    if not client:
         raise HTTPException(status_code=500, detail="OpenAI client not initialized. Check API key.")

    try:
        # --- Construct the messages payload for OpenAI ---
        # For a simple wrapper, we might just send the current message.
        # For a conversational app, you'd include the history.
        messages_payload = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": request.message}
        ]
        # If including history:
        # messages_payload = [
        #     {"role": "system", "content": "You are a helpful assistant."},
        # ] + request.history + [{"role": "user", "content": request.message}]

        # --- Call OpenAI API ---
        response = client.chat.completions.create(
            model="gpt-4",  # Or specify gpt-4-turbo, gpt-3.5-turbo, etc.
            messages=messages_payload,
            max_tokens=150  # Adjust as needed
        )

        ai_response = response.choices[0].message.content
        return {"response": ai_response}

    except Exception as e:
        print(f"Error calling OpenAI: {e}")
        # Provide a more specific error message if possible
        raise HTTPException(status_code=500, detail=f"Failed to get response from OpenAI: {str(e)}")

# --- Run with Uvicorn (for development) ---
# You can run this file directly using `python main.py`
# But `uvicorn main:app --reload` is preferred for development.
if __name__ == "__main__":
    import uvicorn
    print("Running backend server. Access it at http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True) 
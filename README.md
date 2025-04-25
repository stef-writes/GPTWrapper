# Basic GPT Wrapper Project

This project contains a simple backend (FastAPI) and frontend (React)
to interact with the OpenAI GPT API.

## Setup

### Backend

1.  Navigate to the `backend` directory:
    ```bash
    cd backend
    ```
2.  Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Set up environment variables:
    *   Copy the example environment file: `cp .env.example .env`
    *   Edit the `.env` file and add your actual OpenAI API key:
      ```
      OPENAI_API_KEY='sk-...'
      ```
5.  Run the development server:
    ```bash
    uvicorn main:app --reload --port 8000
    ```
    The backend should now be running on `http://127.0.0.1:8000`.

### Frontend

1.  Navigate to the `frontend` directory (from the root):
    ```bash
    cd ../frontend
    # Or from backend: cd ../frontend
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Run the development server:
    ```bash
    npm run dev
    ```
    The frontend should now be running, likely on `http://localhost:5173` (check the terminal output).

## Usage

-   Open the frontend URL in your browser.
-   Type a message in the input box and press Enter or click Send.
-   The message will be sent to the backend, which forwards it to OpenAI.
-   The response from OpenAI will be displayed in the chat area. 
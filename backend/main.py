import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel 
from openai import OpenAI 
from dotenv import load_dotenv 
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time                     # Added for timing
from contextlib import contextmanager # Added for context manager
import networkx as nx # Added import for graph operations

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
    allow_methods=["GET", "POST", "OPTIONS"], 
    allow_headers=["*"],          
)

# --- LLM Configuration Class --- (Copied from user input)
class LLMConfig:
    def __init__(self, model="gpt-4", temperature=0.7, max_tokens=150):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

# --- Default Config Instance ---
# Create a default configuration instance to use
default_llm_config = LLMConfig(
    model="gpt-4",        # Or "gpt-3.5-turbo", etc.
    temperature=0.7,      # Controls randomness (0.0=deterministic, 1.0=more random)
    max_tokens=150        # Max length of the AI's generated response
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

# --- Node Structure --- (Copied from user input)
class Node:
    def __init__(self, node_id, node_type, input_keys=None, output_keys=None, model_config=None):
        self.node_id = node_id        # Unique name for this node
        self.node_type = node_type      # Type of operation (e.g., "text_generation")
        self.input_keys = input_keys or [] # List of data keys this node needs from storage
        self.output_keys = output_keys or []# List of data keys this node will produce
        self.data = {}                  # Internal data for the node (not currently used)
        self.model_config = model_config or default_llm_config # Use node-specific or default LLM config
        self.token_usage = None         # To store token usage from the process method

    def process(self, inputs):
        """Processes input data based on node type. Calls specific AI functions."""
        print(f"--- Processing Node: {self.node_id} ({self.node_type}) ---")
        result = None
        api_response_for_tracking = None # Store API response here for the tracker

        # The token tracker context manager is now placed *inside* relevant node types
        # to ensure it only runs when an actual API call is made.
        if self.node_type == "text_generation":
            with track_token_usage() as usage:
                result_data, api_response_for_tracking = generate_text(inputs, self.model_config)
                self.token_usage = usage # Store usage info
            result = result_data # Assign the content result

        elif self.node_type == "decision_making":
            with track_token_usage() as usage:
                result_data, api_response_for_tracking = process_decision(inputs, self.model_config)
                self.token_usage = usage
            result = result_data

        elif self.node_type == "retrieval":
            # Retrieval doesn't use LLM/tokens, so no tracking here
            result = retrieve_data(inputs)

        elif self.node_type == "logic_chain":
            with track_token_usage() as usage:
                result_data, api_response_for_tracking = logical_reasoning(inputs, self.model_config)
                self.token_usage = usage
            result = result_data

        else:
            print(f"Warning: Unknown node type '{self.node_type}' for node '{self.node_id}'")
            result = None

        # If an API call was made, update the tracker manually (since it's yielded)
        # The tracker printed automatically in its __exit__ method
        if self.token_usage and api_response_for_tracking:
             try:
                 # Ensure we have the dictionary form for update method
                 response_dict = api_response_for_tracking.model_dump()
                 self.token_usage.update(response_dict)
             except Exception as e:
                 print(f"Error updating token tracker: {e}")

        print(f"--- Finished Node: {self.node_id} ---")
        return result

# --- AI Functions with Enhanced Templating --- (Replacing placeholders)
# These functions now return a tuple: (content_dictionary, raw_api_response_object)
# The raw response is needed by the Node.process method to update the token tracker.

def generate_text(inputs: Dict[str, Any], config: LLMConfig) -> Tuple[Dict[str, Any], Any]:
    """Uses OpenAI to generate structured text based on inputs using templates."""
    if not client: raise ValueError("OpenAI client not initialized")
    context = inputs.get('context', '')
    query = inputs.get('query', '')

    system_message = MessageTemplate(role="system", template="You are an expert AI assistant. {context}")
    user_message = MessageTemplate(role="user", template="{query}")

    prompt = PromptTemplate([system_message, user_message])
    formatted_messages = prompt.format_messages(context=context, query=query)

    response = client.chat.completions.create(
        model=config.model,
        messages=formatted_messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens
    )
    # Return the content dict AND the raw response object
    return {"generated_text": response.choices[0].message.content}, response

def process_decision(inputs: Dict[str, Any], config: LLMConfig) -> Tuple[Dict[str, Any], Any]:
    """AI-powered ethical decision-making based on inputs."""
    if not client: raise ValueError("OpenAI client not initialized")
    scenario = inputs.get("situation", "")
    company_value = inputs.get("value", "")

    system_message = MessageTemplate(role="system", template="Analyze this scenario based on ethical and company values.")
    user_message = MessageTemplate(role="user", template="In the given scenario: {scenario}, how does it align with the value: {company_value}?")

    prompt = PromptTemplate([system_message, user_message])
    formatted_messages = prompt.format_messages(scenario=scenario, company_value=company_value)

    response = client.chat.completions.create(
        model=config.model,
        messages=formatted_messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens
    )
    # Return the content dict AND the raw response object
    return {"decision_output": response.choices[0].message.content}, response

def retrieve_data(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieves stored data from previous nodes."""
    # Expects 'storage' (the graph's data store) and 'key' (which data to get)
    storage = inputs.get("storage", {})
    key_to_retrieve = inputs.get("key", "")
    retrieved_value = storage.get(key_to_retrieve, "No data found.")
    # Return value using a standard output key for retrieval nodes
    return {"retrieved_data": retrieved_value}

def logical_reasoning(inputs: Dict[str, Any], config: LLMConfig) -> Tuple[Dict[str, Any], Any]:
    """Processes multi-step logical AI reasoning chains."""
    if not client: raise ValueError("OpenAI client not initialized")
    premise = inputs.get("premise", "")
    supporting_evidence = inputs.get("supporting_evidence", "")

    system_message = MessageTemplate(role="system", template="Perform structured logical reasoning.")
    user_message = MessageTemplate(role="user", template="Given the premise: {premise}, and supporting evidence: {supporting_evidence}, logically conclude the next step.")

    prompt = PromptTemplate([system_message, user_message])
    formatted_messages = prompt.format_messages(premise=premise, supporting_evidence=supporting_evidence)

    response = client.chat.completions.create(
        model=config.model,
        messages=formatted_messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens
    )
    # Return the content dict AND the raw response object
    return {"reasoning_result": response.choices[0].message.content}, response

# --- Callback System --- (Copied from user input)
class Callback:
    """Base class for callbacks during graph execution."""
    def on_node_start(self, node_id, node_type, inputs):
        """Called before a node starts processing."""
        pass # Base implementation does nothing

    def on_node_complete(self, node_id, node_type, result, token_usage):
        """Called after a node finishes processing."""
        pass # Base implementation does nothing

    def on_chain_complete(self, final_results: Dict[str, Any], total_tokens: int, total_cost: float):
        """Called after the entire graph/chain finishes execution."""
        pass # Base implementation does nothing

class LoggingCallback(Callback):
    """Simple callback that logs events to the console."""
    def on_node_start(self, node_id, node_type, inputs):
        print(f"[Callback] START Node '{node_id}' ({node_type}) with inputs: {list(inputs.keys())}")

    def on_node_complete(self, node_id, node_type, result, token_usage):
        # Result is expected to be a dictionary
        output_keys = list(result.keys()) if isinstance(result, dict) else []
        print(f"[Callback] END   Node '{node_id}' ({node_type}) producing outputs: {output_keys}")
        if token_usage:
            # Access attributes directly from the stored TokenUsage object
            print(f"  [Callback] Usage: {token_usage.total_tokens} tokens (${token_usage.cost:.6f}) Est. Cost")

    def on_chain_complete(self, final_results: Dict[str, Any], total_tokens: int, total_cost: float):
        print(f"[Callback] === Chain Complete ===")
        print(f"  [Callback] Final Results Keys: {list(final_results.keys())}")
        print(f"  [Callback] Total Estimated Cost: ${total_cost:.6f}")
        print(f"  [Callback] Total Tokens Used: {total_tokens}")
        print(f"[Callback] ======================")

# --- Graph-Based Execution --- (Copied from user input)
class ScriptChain:
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.graph = nx.DiGraph() # Directed graph to hold nodes and connections
        self.storage = {}          # Dictionary to store outputs from executed nodes
        self.callbacks = callbacks or [] # List of callback objects to notify

    def add_node(self, node_id: str, node_type: str, input_keys: Optional[List[str]] = None, output_keys: Optional[List[str]] = None, model_config: Optional[LLMConfig] = None):
        """Adds a node (processing step) to the graph."""
        # Associates a Node object with the node_id in the networkx graph
        node_instance = Node(node_id, node_type, input_keys, output_keys, model_config)
        self.graph.add_node(node_id, node=node_instance)

    def add_edge(self, from_node: str, to_node: str):
        """Adds a directed connection (dependency) between two nodes."""
        # Ensures that 'from_node' must execute before 'to_node'
        self.graph.add_edge(from_node, to_node)

    def add_callback(self, callback: Callback):
        """Registers a callback object to receive execution events."""
        if isinstance(callback, Callback):
            self.callbacks.append(callback)
        else:
            print(f"Warning: Attempted to add non-Callback object: {callback}")

    def execute(self):
        """Executes the graph nodes in topological (dependency) order."""
        try:
            # Calculate the order nodes must run based on edges
            execution_order = list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            print("Error: Graph contains a cycle, cannot determine execution order.")
            return {"error": "Graph contains a cycle"}
        except Exception as e:
             print(f"Error during topological sort: {e}")
             return {"error": f"Failed to determine execution order: {e}"}

        results = {} # Stores the final output of each node by node_id
        total_tokens = 0
        total_cost = 0.0 # Use float for cost

        print(f"--- Executing Chain (Order: {execution_order}) ---")

        for node_id in execution_order:
            if node_id not in self.graph:
                 print(f"Error: Node '{node_id}' found in execution order but not in graph.")
                 continue # Or handle error more formally

            node_instance = self.graph.nodes[node_id].get("node")
            if not isinstance(node_instance, Node):
                 print(f"Error: Node '{node_id}' in graph does not contain a valid Node object.")
                 continue # Or handle error more formally

            # --- Prepare Inputs for Node --- 
            # Gather required inputs from the storage (outputs of previous nodes)
            inputs_for_node = {key: self.storage.get(key) for key in node_instance.input_keys}
            # Filter out None values if a previous node didn't produce expected output
            # Alternatively, could raise an error here if an input is missing.
            inputs_for_node = {k: v for k, v in inputs_for_node.items() if v is not None}
            # Also pass the entire storage for potential direct access (e.g., retrieval node)
            inputs_for_node["storage"] = self.storage

            # --- Trigger on_node_start Callbacks --- 
            for callback in self.callbacks:
                try:
                    callback.on_node_start(node_id, node_instance.node_type, inputs_for_node)
                except Exception as e:
                    print(f"Error in callback {type(callback).__name__}.on_node_start for node {node_id}: {e}")

            # --- Process the Node --- 
            try:
                node_result = node_instance.process(inputs_for_node)
            except Exception as e:
                 print(f"Error processing node {node_id}: {e}")
                 # Decide how to handle node errors - stop chain? continue? store error?
                 results[node_id] = {"error": str(e)} # Example: store error and continue
                 self.storage[node_id] = {"error": str(e)}
                 continue # Skip storing result and callbacks for this failed node

            # --- Store Result --- 
            # Store the direct result (expected to be a dictionary) in results and storage
            # Note: The previous logic extracting 'content' might be too specific.
            # Storing the whole result dict might be more flexible.
            # Adjust based on how output_keys are intended to be used.
            if isinstance(node_result, dict):
                results[node_id] = node_result
                self.storage.update(node_result) # Update storage with all keys from result dict
            else:
                 # Handle non-dict results if necessary
                 results[node_id] = {"output": node_result} # Wrap non-dict result
                 self.storage[node_id] = {"output": node_result}

            # --- Aggregate Token Stats --- 
            if node_instance.token_usage:
                # Ensure usage object exists and has attributes before accessing
                try:
                    total_tokens += getattr(node_instance.token_usage, 'total_tokens', 0)
                    total_cost += getattr(node_instance.token_usage, 'cost', 0.0)
                except AttributeError:
                     print(f"Warning: token_usage object for node {node_id} missing expected attributes.")

            # --- Trigger on_node_complete Callbacks --- 
            for callback in self.callbacks:
                try:
                    callback.on_node_complete(node_id, node_instance.node_type, results.get(node_id), node_instance.token_usage)
                except Exception as e:
                    print(f"Error in callback {type(callback).__name__}.on_node_complete for node {node_id}: {e}")

        # --- Trigger on_chain_complete Callbacks --- 
        print("--- Chain Execution Finished ---")
        for callback in self.callbacks:
             try:
                 callback.on_chain_complete(results, total_tokens, total_cost)
             except Exception as e:
                 print(f"Error in callback {type(callback).__name__}.on_chain_complete: {e}")

        # Return final results and aggregated stats
        return {
            "results": results, # Dictionary mapping node_id to its result dictionary
            "stats": {
                "total_tokens": total_tokens,
                "total_cost": total_cost
            }
        }

# --- Data Models (Pydantic) ---
# Define expected data structures for requests

# Structure for one chat message (like a Python dictionary)
# Keys: 'role', 'content' | Values: string
# Used in the list sent to OpenAI
class Message(BaseModel):
    role: str
    content: str

# DEPRECATED - Replaced by graph API models
# class ChatRequest(BaseModel):
#     message: str
#     history: Optional[List[Message]] = []

# --- OpenAI Client Setup --- (Re-adding this section)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY not found. Did you create a .env file?")
    client = None # Mark client as unusable
else:
    # Try to set up the OpenAI client object
    try:
        client = OpenAI(api_key=api_key)
        print("--- OpenAI Client Initialized Successfully ---")
    except Exception as e:
        print(f"Error setting up OpenAI client: {e}")
        client = None # Mark client as unusable

# --- API Interface for UI Integration --- 

# Pydantic models for the *new* graph API
class ModelConfigInput(BaseModel):
    # Input model for specifying LLM config in API requests
    model: str = default_llm_config.model
    temperature: float = default_llm_config.temperature
    max_tokens: Optional[int] = default_llm_config.max_tokens

class NodeInput(BaseModel):
    # Input model for adding/defining a node via API
    node_id: str
    node_type: str
    input_keys: List[str] = []
    output_keys: List[str] = []
    # Renamed field from model_config to llm_config due to Pydantic V2 conflict
    llm_config: Optional[ModelConfigInput] = None

class EdgeInput(BaseModel):
    # Input model for adding an edge via API
    from_node: str
    to_node: str

class GenerateTextNodeRequest(BaseModel):
    # Input model for the NEW single-node text generation endpoint
    prompt_text: str # The final, already formatted prompt text
    # Renamed from model_config to avoid Pydantic v2 conflict
    llm_config: Optional[ModelConfigInput] = None # Optional config override

class GenerateTextNodeResponse(BaseModel):
    # Output model for the NEW single-node text generation endpoint
    generated_text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost: Optional[float] = None
    duration: Optional[float] = None

# !!! IMPORTANT: Global ScriptChain instance !!!
# This instance persists across requests for the server's lifetime.
# Good for demos, but NOT suitable for multiple concurrent users.
# In a real application, you would manage chain instances per user/session.
print("--- Initializing Global ScriptChain Instance ---")
global_script_chain = ScriptChain()
global_script_chain.add_callback(LoggingCallback())

# --- API Routes --- 

@app.get("/")
def read_root():
    return {"message": "ScriptChain Backend Running"}

# --- Graph Construction Endpoints --- 

@app.post("/add_node", status_code=201)
async def add_node_api(node: NodeInput):
    """Adds a node to the *global* script chain via API."""
    llm_config_for_node = default_llm_config
    # Use the renamed field 'llm_config' here
    if node.llm_config:
        llm_config_for_node = LLMConfig(
            # And here
            model=node.llm_config.model,
            temperature=node.llm_config.temperature,
            max_tokens=node.llm_config.max_tokens
        )
    try:
        global_script_chain.add_node(
            # ... (arguments remain the same, model_config arg is correct here) ...
            node_id=node.node_id,
            node_type=node.node_type,
            input_keys=node.input_keys,
            output_keys=node.output_keys,
            model_config=llm_config_for_node # This maps to the Node class __init__ param
        )
        print(f"Added node: {node.node_id}")
        return {"message": f"Node '{node.node_id}' added successfully."}
    except Exception as e:
        print(f"Error adding node {node.node_id}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to add node: {str(e)}")

@app.post("/add_edge", status_code=201)
async def add_edge_api(edge: EdgeInput):
    """Adds an edge to the *global* script chain via API."""
    # Basic validation: Check if nodes exist before adding edge
    if edge.from_node not in global_script_chain.graph or edge.to_node not in global_script_chain.graph:
        raise HTTPException(status_code=404, detail=f"Node(s) not found: '{edge.from_node}' or '{edge.to_node}'")
    # --- CYCLE PREVENTION ---
    # Temporarily add the edge and check for cycles
    global_script_chain.graph.add_edge(edge.from_node, edge.to_node)
    if not nx.is_directed_acyclic_graph(global_script_chain.graph):
        global_script_chain.graph.remove_edge(edge.from_node, edge.to_node)
        raise HTTPException(status_code=400, detail="Adding this edge would create a cycle. Please check your node connections.")
    print(f"Added edge: {edge.from_node} -> {edge.to_node}")
    return {"message": f"Edge from '{edge.from_node}' to '{edge.to_node}' added successfully."}

# --- Single Node Execution Endpoint --- (NEW)
@app.post("/generate_text_node", response_model=GenerateTextNodeResponse)
async def generate_text_node_api(request: GenerateTextNodeRequest):
    """Executes a single text generation call based on provided prompt text."""
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI client not set up. Check API key.")

    # Determine config: use request's config or fallback to global default
    node_config = default_llm_config
    # Use the renamed field 'llm_config' here
    if request.llm_config:
        node_config = LLMConfig(
            # And here
            model=request.llm_config.model,
            temperature=request.llm_config.temperature,
            max_tokens=request.llm_config.max_tokens
        )

    print(f"--- Executing Single Text Generation (Model: {node_config.model}) ---")

    # Prepare simple messages list (system prompt + user prompt text)
    # Note: No history or complex templating needed here as per request
    messages_payload = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": request.prompt_text} # Use the direct prompt text
    ]

    response_content = None
    tracker_instance = None

    try:
        # Use the token tracker context manager
        with track_token_usage() as tracker:
            response = client.chat.completions.create(
                model=node_config.model,
                messages=messages_payload,
                temperature=node_config.temperature,
                max_tokens=node_config.max_tokens
            )
            # Update the tracker with the response
            response_dict = response.model_dump()
            tracker.update(response_dict)
            response_content = response.choices[0].message.content
            tracker_instance = tracker # Store the tracker instance

        if response_content is None:
            raise ValueError("Received no content from OpenAI.")

        # Prepare the response using the Pydantic model
        return GenerateTextNodeResponse(
            generated_text=response_content,
            prompt_tokens=getattr(tracker_instance, 'prompt_tokens', None),
            completion_tokens=getattr(tracker_instance, 'completion_tokens', None),
            total_tokens=getattr(tracker_instance, 'total_tokens', None),
            cost=getattr(tracker_instance, 'cost', None),
            duration=round(tracker_instance.end_time - tracker_instance.start_time, 2) if tracker_instance and tracker_instance.end_time else None
        )

    except Exception as e:
        print(f"Error during single text generation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed text generation: {str(e)}")

# --- Graph Execution Endpoint --- (Potentially less used now)
@app.post("/execute")
async def execute_api(initial_inputs: Optional[Dict[str, Any]] = None):
    """Executes the *global* AI-driven node chain."""
    print("--- Received /execute request (Full Graph) ---")
    global_script_chain.storage = initial_inputs or {}
    print(f"Initial storage set to: {global_script_chain.storage}")

    try:
        results = global_script_chain.execute()
        if results and "error" in results:
             raise HTTPException(status_code=400, detail=results["error"])
        return results
    except Exception as e:
        print(f"Error during chain execution: {e}")
        raise HTTPException(status_code=500, detail=f"Chain execution failed: {str(e)}")

# --- Deprecated Chat Endpoint --- 
# Commenting out the old chat endpoint as it's replaced by the graph execution model
# @app.post("/api/chat")
# async def chat_endpoint(request: ChatRequest):
#    """Handle chat request, call OpenAI, return response."""
#    ...

# --- Run Server --- (Existing code)
if __name__ == "__main__":
    import uvicorn 
    print("Starting backend server at http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True) 
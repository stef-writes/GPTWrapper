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
import re # Added for regex pattern matching
import json # Added for JSON parsing

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
    def __init__(self, node_id, node_type, input_keys=None, output_keys=None, model_config=None, template=None):
        self.node_id = node_id        # Unique name for this node
        self.node_type = node_type      # Type of operation (e.g., "text_generation")
        self.input_keys = input_keys or [] # List of data keys this node needs from storage
        self.output_keys = output_keys or []# List of data keys this node will produce
        self.data = {}                  # Internal data for the node (not currently used)
        self.model_config = model_config or default_llm_config # Use node-specific or default LLM config
        self.token_usage = None         # To store token usage from the process method
        self.template = template        # Optional template configuration for the node

    def process(self, inputs):
        """Processes input data based on node type. Calls specific AI functions."""
        print(f"--- Processing Node: {self.node_id} ({self.node_type}) ---")
        result = None
        api_response_for_tracking = None # Store API response here for the tracker

        # Apply template if available
        processed_inputs = self._apply_template(inputs)

        # The token tracker context manager is now placed *inside* relevant node types
        # to ensure it only runs when an actual API call is made.
        if self.node_type == "text_generation":
            with track_token_usage() as usage:
                result_data, api_response_for_tracking = generate_text(processed_inputs, self.model_config)
                self.token_usage = usage # Store usage info
            result = result_data # Assign the content result

        elif self.node_type == "decision_making":
            with track_token_usage() as usage:
                result_data, api_response_for_tracking = process_decision(processed_inputs, self.model_config)
                self.token_usage = usage
            result = result_data

        elif self.node_type == "retrieval":
            # Retrieval doesn't use LLM/tokens, so no tracking here
            result = retrieve_data(processed_inputs)

        elif self.node_type == "logic_chain":
            with track_token_usage() as usage:
                result_data, api_response_for_tracking = logical_reasoning(processed_inputs, self.model_config)
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
    
    def _apply_template(self, inputs):
        """Apply node template if defined, otherwise return original inputs."""
        if not self.template:
            return inputs
            
        # Use the global template processor for consistent processing
        return template_processor.process_node_template(self.template, inputs, self.node_id)

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

# --- Enhanced Storage and Context Management ---
class ContextVersion:
    """Tracks versions of context data for each node."""
    def __init__(self):
        self.versions = {}  # {node_id: version_number}
    
    def update(self, node_id):
        """Increment the version number for a node."""
        self.versions[node_id] = self.versions.get(node_id, 0) + 1
        return self.versions[node_id]
    
    def get(self, node_id):
        """Get the current version number for a node."""
        return self.versions.get(node_id, 0)

class NamespacedStorage:
    """
    A storage system that namespaces data by node ID to prevent key collisions.
    Allows for retrieving outputs from specific nodes or by key across all nodes.
    """
    
    def __init__(self):
        self.data = {}  # Main storage: {node_id: {key: value}}
        
    def store(self, node_id, data):
        """Store data dictionary under node_id"""
        if not isinstance(data, dict):
            raise ValueError(f"Data must be a dictionary, got {type(data)}")
        
        if node_id not in self.data:
            self.data[node_id] = {}
            
        # Store all key-value pairs from the data dict
        for key, value in data.items():
            self.data[node_id][key] = value
        
    def get(self, node_id, key=None):
        """
        Get a value from storage.
        If key is None, return all data for the node.
        If key is provided, return the specific value.
        """
        if node_id not in self.data:
            return None if key else {}
            
        if key is None:
            return self.data[node_id]
        
        return self.data[node_id].get(key)
    
    def get_all_data(self):
        """Return a flat dictionary with node_id:key as the keys"""
        flat_data = {}
        for node_id, node_data in self.data.items():
            for key, value in node_data.items():
                flat_data[f"{node_id}:{key}"] = value
        return flat_data
        
    def has_node(self, node_id):
        """Check if a node has any data stored"""
        return node_id in self.data
    
    def get_node_output(self, node_id, key=None):
        """Helper method to get output from a specific node"""
        return self.get(node_id, key)
        
    def get_by_key(self, key):
        """
        Scan all nodes for a key and return the first value found.
        This is used for backward compatibility with non-namespaced keys.
        """
        for node_data in self.data.values():
            if key in node_data:
                return node_data[key]
        return None
        
    def get_flattened(self):
        """
        Return a flattened view of all data without namespacing.
        Used for backward compatibility.
        If there are key collisions, the last value encountered wins.
        """
        flat_data = {}
        for node_data in self.data.values():
            flat_data.update(node_data)
        return flat_data

class InputValidator:
    """Validates inputs for a node before processing."""
    @staticmethod
    def validate(node, available_inputs):
        """Validate that all required inputs for a node are available."""
        missing = []
        for key in node.input_keys:
            if key not in available_inputs or available_inputs[key] is None:
                missing.append(key)
        
        if missing:
            raise ValueError(f"Missing required inputs for node '{node.node_id}': {missing}")
        
        return True

# --- Update ScriptChain class to use namespaced storage ---
class ScriptChain:
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.graph = nx.DiGraph()  # Directed graph to hold nodes and connections
        self.storage = NamespacedStorage()  # Namespaced storage to prevent key collisions
        self.callbacks = callbacks or []  # List of callback objects to notify

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

        results = {}  # Stores the final output of each node by node_id
        total_tokens = 0
        total_cost = 0.0  # Use float for cost

        print(f"--- Executing Chain (Order: {execution_order}) ---")

        for node_id in execution_order:
            if node_id not in self.graph:
                print(f"Error: Node '{node_id}' found in execution order but not in graph.")
                continue  # Or handle error more formally

            node_instance = self.graph.nodes[node_id].get("node")
            if not isinstance(node_instance, Node):
                print(f"Error: Node '{node_id}' in graph does not contain a valid Node object.")
                continue  # Or handle error more formally

            # --- Prepare Inputs for Node --- 
            # Get required inputs that aren't node-specific
            inputs_for_node = {}
            
            # Find direct upstream nodes that provide inputs
            upstream_nodes = list(self.graph.predecessors(node_id))
            
            # Collect outputs from each upstream node
            for upstream_id in upstream_nodes:
                node_outputs = self.storage.get_node_output(upstream_id)
                if node_outputs:
                    # Add each output with a namespaced key: {node_id}:{output_key}
                    for output_key, output_value in node_outputs.items():
                        namespaced_key = f"{upstream_id}:{output_key}"
                        inputs_for_node[namespaced_key] = output_value
                        
                        # Also provide direct access to keys specified in input_keys
                        if output_key in node_instance.input_keys:
                            inputs_for_node[output_key] = output_value
            
            # Add non-namespaced keys for backward compatibility
            for key in node_instance.input_keys:
                if key not in inputs_for_node:
                    # Try to find from any node via flattened view
                    value = self.storage.get_by_key(key)
                    if value is not None:
                        inputs_for_node[key] = value
            
            # Provide the flattened storage for backward compatibility
            inputs_for_node["storage"] = self.storage.get_flattened()
            
            # Add namespaced accessors for more precision
            inputs_for_node["get_node_output"] = self.storage.get_node_output
            
            # --- Validate inputs ---
            try:
                InputValidator.validate(node_instance, inputs_for_node)
            except ValueError as e:
                print(f"Input validation error for node {node_id}: {e}")
                results[node_id] = {"error": str(e)}
                self.storage.store(node_id, {"error": str(e)})
                continue  # Skip processing this node

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
                # Store error and continue
                results[node_id] = {"error": str(e)}
                self.storage.store(node_id, {"error": str(e)})
                continue  # Skip storing result and callbacks for this failed node

            # --- Store Result --- 
            if isinstance(node_result, dict):
                results[node_id] = node_result
                self.storage.store(node_id, node_result)  # Store with namespace
            else:
                # Handle non-dict results
                results[node_id] = {"output": node_result}  # Wrap non-dict result
                self.storage.store(node_id, {"output": node_result})

            # --- Aggregate Token Stats --- 
            if node_instance.token_usage:
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
            "results": results,  # Dictionary mapping node_id to its result dictionary
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
    # Add context_data to hold selected node outputs
    context_data: Optional[Dict[str, str]] = None # Map of node names to their outputs

class GenerateTextNodeResponse(BaseModel):
    # Output model for the NEW single-node text generation endpoint
    generated_text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost: Optional[float] = None
    duration: Optional[float] = None

# --- Content Parser for Structured Data ---
class ContentParser:
    """Parser for extracting structured data from node outputs"""
    
    @staticmethod
    def parse_numbered_list(content):
        """Extracts items from a numbered list into a dictionary"""
        if not content or not isinstance(content, str):
            return {}
        
        items = {}
        # Match patterns like "1. Item" or "1) Item" or "1: Item"
        pattern = r'(\d+)[.):]\s+(.*?)(?=\n\d+[.):]\s+|\Z)'
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            num, text = match.groups()
            items[int(num)] = text.strip()
        
        return items
    
    @staticmethod
    def extract_item(content, item_num):
        """Extract a specific numbered item from content"""
        if not isinstance(item_num, int):
            try:
                item_num = int(item_num)
            except (ValueError, TypeError):
                return None
        
        items = ContentParser.parse_numbered_list(content)
        return items.get(item_num)
    
    @staticmethod
    def try_parse_json(content):
        """Attempt to parse content as JSON"""
        if not content or not isinstance(content, str):
            return None
        
        try:
            # Find JSON-like structures (between { } or [ ])
            json_pattern = r'(\{.*\}|\[.*\])'
            match = re.search(json_pattern, content, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            return None
        except Exception:
            return None
    
    @staticmethod
    def extract_table(content):
        """Extract tabular data if present"""
        if not content or not isinstance(content, str):
            return None
        
        # Simple markdown table detection
        lines = content.split('\n')
        table_start = None
        table_end = None
        
        # Look for table markers (| --- |)
        for i, line in enumerate(lines):
            if '|' in line and '---' in line and table_start is None:
                table_start = i - 1  # Header row is usually above
                continue
            if table_start is not None and ('|' not in line or line.strip() == ''):
                table_end = i
                break
        
        if table_start is not None:
            table_end = table_end or len(lines)
            if table_start >= 0:
                table_rows = lines[table_start:table_end]
                # Convert to list of dictionaries if it has a header
                if len(table_rows) >= 2:  # Need at least header + separator
                    return table_rows
        return None

# --- DataAccessor Helper Class ---
class DataAccessor:
    """
    Helper class to provide structured access to node data.
    Used for advanced data extraction from node outputs.
    """
    
    def __init__(self, node_data):
        """Initialize with a dictionary of node outputs"""
        self.node_data = node_data
        self.parser = ContentParser()
        
    def get_node_content(self, node_name):
        """Get raw content from a node"""
        return self.node_data.get(node_name)
        
    def get_item(self, node_name, item_num):
        """Get a specific numbered item from a node's output"""
        if node_name not in self.node_data:
            return None
            
        content = self.node_data[node_name]
        return self.parser.extract_item(content, item_num)
        
    def get_json(self, node_name):
        """Try to parse node output as JSON"""
        if node_name not in self.node_data:
            return None
            
        content = self.node_data[node_name]
        return self.parser.try_parse_json(content)
        
    def get_table(self, node_name):
        """Extract table data from a node if present"""
        if node_name not in self.node_data:
            return None
            
        content = self.node_data[node_name]
        return self.parser.extract_table(content)
        
    def get_all_nodes(self):
        """Get list of all available node names"""
        return list(self.node_data.keys())
        
    def has_node(self, node_name):
        """Check if a node exists"""
        return node_name in self.node_data
        
    def analyze_content(self, node_name):
        """Perform comprehensive analysis of a node's content"""
        if not self.has_node(node_name):
            return None
            
        content = self.node_data[node_name]
        result = {
            "has_numbered_list": False,
            "numbered_items_count": 0,
            "has_json": False,
            "has_table": False,
            "content_length": len(content) if content else 0
        }
        
        # Check for numbered list
        numbered_items = self.parser.parse_numbered_list(content)
        if numbered_items:
            result["has_numbered_list"] = True
            result["numbered_items_count"] = len(numbered_items)
            
        # Check for JSON
        json_data = self.parser.try_parse_json(content)
        if json_data:
            result["has_json"] = True
            
        # Check for table
        table_data = self.parser.extract_table(content)
        if table_data:
            result["has_table"] = True
            
        return result

# --- Template Processor for consistent variable substitution ---
class TemplateProcessor:
    """
    Unified template processing system that handles all node variable substitutions.
    This replaces both Node._apply_template and the template processing in generate_text_node_api.
    """
    
    def __init__(self, debug_mode=True):
        """Initialize the template processor"""
        self.debug_mode = debug_mode  # Enable detailed logging
        
    def log(self, message):
        """Log messages when debug mode is enabled"""
        if self.debug_mode:
            print(message)
    
    def validate_node_references(self, template_text, available_nodes):
        """
        Validate that all node references in the template exist in available_nodes.
        Returns a tuple: (is_valid, list_of_missing_nodes, list_of_found_nodes)
        """
        if not template_text or not isinstance(template_text, str):
            return True, [], []
            
        # Find all {NodeName} references in the template
        # This pattern matches both simple {NodeName} and indexed {NodeName[n]} references
        reference_pattern = r'\{([^:\}\[]+)(?:\[(\d+)\]|\:item\((\d+)\))?\}'
        matches = re.findall(reference_pattern, template_text)
        
        # Create normalized versions of available nodes for case-insensitive matching
        normalized_available_nodes = {node.lower().strip(): node for node in available_nodes}
        
        missing_nodes = []
        found_nodes = []
        
        for match in matches:
            node_name = match[0]
            normalized_node_name = node_name.lower().strip()
            
            # First try exact match
            if node_name in available_nodes:
                found_nodes.append(node_name)
            # Then try case-insensitive match
            elif normalized_node_name in normalized_available_nodes:
                original_node = normalized_available_nodes[normalized_node_name]
                found_nodes.append(node_name)  # Add the referenced name
                self.log(f"Found node '{node_name}' using case-insensitive matching (original: '{original_node}')")
            else:
                missing_nodes.append(node_name)
        
        is_valid = len(missing_nodes) == 0
        return is_valid, missing_nodes, found_nodes
    
    def process_node_template(self, template, inputs, node_id=None):
        """
        Process a node's template configuration (used by Node._apply_template).
        Returns processed inputs dictionary.
        """
        if not template:
            return inputs
            
        # Create a copy to avoid modifying the original
        processed_inputs = inputs.copy()
        
        try:
            self.log(f"Processing node template for node: {node_id or 'unknown'}")
            
            # First validate we have all required inputs
            missing_inputs = []
            for field_name, template_string in template.items():
                if not isinstance(template_string, str):
                    continue
                    
                # Find all input references in this template
                # Standard Python format variables like {variable_name}
                var_pattern = r'\{([^{}]+)\}'
                variables = re.findall(var_pattern, template_string)
                
                for var in variables:
                    # Skip function references like get_output
                    if '(' in var:
                        continue
                        
                    if var not in inputs and ":" not in var:
                        missing_inputs.append(var)
            
            if missing_inputs:
                self.log(f"Warning: Template missing inputs: {missing_inputs}")
            
            # Process each template field
            for field_name, template_string in template.items():
                # Skip if the field is not a string template
                if not isinstance(template_string, str):
                    continue
                    
                try:
                    # Replace namespaced keys with their values for template formatting
                    template_context = {}
                    
                    # Add non-namespaced values first
                    for key, value in inputs.items():
                        if ":" not in key and not callable(value):
                            template_context[key] = value
                    
                    # Add access to namespaced values through helper functions
                    if "get_node_output" in inputs and callable(inputs["get_node_output"]):
                        get_node_output = inputs["get_node_output"]
                        
                        # Add a function to access node outputs in templates
                        def get_output(node_id, output_key=None):
                            return get_node_output(node_id, output_key)
                        
                        template_context["get_output"] = get_output
                    
                    formatted_value = template_string.format(**template_context)
                    processed_inputs[field_name] = formatted_value
                    
                    self.log(f"Processed template field '{field_name}': {formatted_value[:50]}...")
                    
                except KeyError as e:
                    self.log(f"Warning: Missing key {e} in template for node {node_id}")
                except Exception as e:
                    self.log(f"Error formatting template for node {node_id}: {e}")
        
        except Exception as e:
            self.log(f"Error applying template for node {node_id}: {e}")
            # Fall back to original inputs on error
            return inputs
            
        return processed_inputs
    
    def process_node_references(self, prompt_text, context_data, data_accessor=None):
        """
        Process a prompt text with node references (used by generate_text_node_api).
        Returns processed prompt and a dictionary of processed node values.
        """
        if not prompt_text or not context_data:
            return prompt_text, {}
            
        processed_prompt = prompt_text
        processed_node_values = {}  # Track which nodes were processed and their values
        
        try:
            self.log(f"Processing node references in prompt")
            
            # Extract node mapping if available
            node_mapping = context_data.get('__node_mapping', {})
            if node_mapping:
                self.log(f"Found node name-to-ID mapping: {node_mapping}")
                
            # Create a normalized version of context_data keys for case-insensitive matching
            normalized_context_data = {}
            for key, value in context_data.items():
                # Skip the special mapping key
                if key == '__node_mapping':
                    continue
                    
                # Store both the original key and a normalized version (lowercase, trimmed)
                normalized_key = key.lower().strip()
                normalized_context_data[normalized_key] = (key, value)
            
            self.log(f"Normalized context data keys: {list(normalized_context_data.keys())}")
            
            # Validate node references
            is_valid, missing_nodes, found_nodes = self.validate_node_references(
                prompt_text, [k for k in context_data.keys() if k != '__node_mapping']
            )
            
            if not is_valid:
                self.log(f"Warning: Template references non-existent nodes: {missing_nodes}")
                # Continue processing anyway with the nodes we do have
                
                # Try to resolve missing nodes using the mapping
                resolved_nodes = []
                for missing_node in missing_nodes:
                    # Check if any node maps to this name
                    for name, node_id in node_mapping.items():
                        if name.lower().strip() == missing_node.lower().strip():
                            # Found a match by name
                            resolved_nodes.append(missing_node)
                            self.log(f"Resolved missing node '{missing_node}' via mapping to ID '{node_id}'")
                            break
                            
                        # Also check for ID-prefixed keys
                        if f"id:{node_id}" in context_data:
                            resolved_nodes.append(missing_node)
                            self.log(f"Resolved missing node '{missing_node}' via ID lookup")
                            break
                
                if resolved_nodes:
                    self.log(f"Resolved {len(resolved_nodes)} of {len(missing_nodes)} missing nodes via mapping")
                    # Remove resolved nodes from missing list
                    missing_nodes = [n for n in missing_nodes if n not in resolved_nodes]
            
            if found_nodes:
                self.log(f"Found references to nodes: {found_nodes}")
            
            # Step 1: Create data accessor if not provided
            if not data_accessor and context_data:
                # Filter out the special mapping key
                filtered_context_data = {k: v for k, v in context_data.items() if k != '__node_mapping'}
                data_accessor = DataAccessor(filtered_context_data)
            
            # Step 2: Process advanced references like {NodeName[n]} first
            if data_accessor:
                # Handle patterns like {Node1[2]} or {Node1:item(2)}
                def replace_item_reference(match):
                    full_match = match.group(0)
                    node_name = match.group(1)
                    item_num_str = match.group(2) or match.group(3)  # Either [2] or :item(2) format
                    
                    self.log(f"Processing item reference: {full_match}")
                    self.log(f"  Node: {node_name}")
                    self.log(f"  Item: {item_num_str}")
                    
                    # Priority 1: Try exact match by name
                    if data_accessor.has_node(node_name):
                        self.log(f"  Found node '{node_name}' by exact name match")
                    else:
                        # Priority 2: Try case-insensitive matching by name
                        normalized_node_name = node_name.lower().strip()
                        if normalized_node_name in normalized_context_data:
                            # Use the original key and value
                            original_key, node_output = normalized_context_data[normalized_node_name]
                            self.log(f"  Found node '{node_name}' using normalized matching (original key: '{original_key}')")
                            
                            # Update data_accessor with the correct key if needed
                            if node_name != original_key:
                                data_accessor.node_data[node_name] = node_output
                        
                        # Priority 3: Try lookup via node mapping
                        elif node_mapping:
                            # Check if this name is in the mapping
                            node_id = None
                            for mapped_name, mapped_id in node_mapping.items():
                                if mapped_name.lower().strip() == normalized_node_name:
                                    node_id = mapped_id
                                    self.log(f"  Found node '{node_name}' via mapping to ID '{node_id}'")
                                    break
                                    
                            if node_id and f"id:{node_id}" in context_data:
                                # Add this data to the accessor
                                data_accessor.node_data[node_name] = context_data[f"id:{node_id}"]
                    
                    if not data_accessor.has_node(node_name):
                        self.log(f"  Node '{node_name}' not found")
                        return full_match  # Node not found, return unchanged
                    
                    # Convert item number to int
                    try:
                        item_num = int(item_num_str)
                    except ValueError:
                        self.log(f"  Invalid item number: {item_num_str}")
                        return full_match  # Not a valid number
                    
                    # Get the specific item
                    item_content = data_accessor.get_item(node_name, item_num)
                    if item_content:
                        self.log(f"  Found item {item_num}: {item_content}")
                        processed_node_values[f"{node_name}[{item_num}]"] = item_content
                        return item_content
                    
                    self.log(f"  Item {item_num} not found in node {node_name}")
                    # Fallback to original reference if item not found
                    return full_match
                
                # Process item references like {Node1[2]} or {Node1:item(2)}
                item_ref_pattern = r'\{([^:\}\[]+)(?:\[(\d+)\]|\:item\((\d+)\))\}'
                processed_prompt = re.sub(item_ref_pattern, replace_item_reference, processed_prompt)
                self.log(f"After item reference processing: {processed_prompt[:100]}...")
            
            # Step 3: Process normal node references
            for key, value in context_data.items():
                # Skip special keys and empty values
                if key == '__node_mapping' or not value or (isinstance(value, str) and value.strip() == ""):
                    continue
                    
                # Get the node name (or ID-prefixed key)
                node_name = key
                node_output = value
                
                # Replace any remaining direct node references (exact match)
                template_marker = "{" + node_name + "}"
                if template_marker in processed_prompt:
                    self.log(f"Processing node reference: {template_marker}")
                    self.log(f"Original node output: '{node_output}'")
                    
                    final_value = node_output  # Default value
                    
                    # Check if the node output is a simple number (for calculations)
                    try:
                        # First try to see if it's already a number type
                        if isinstance(node_output, (int, float)):
                            final_value = str(node_output)
                            self.log(f"Numeric value detected (direct): {final_value}")
                            processed_prompt = processed_prompt.replace(template_marker, final_value)
                            processed_node_values[node_name] = final_value
                            continue
                            
                        # Next try to convert string to number if it looks like one
                        if isinstance(node_output, str) and node_output.strip().replace('.', '', 1).isdigit():
                            # This will catch numbers like "2" or "3.14"
                            final_value = node_output.strip()
                            self.log(f"Numeric value detected (string): {final_value}")
                            processed_prompt = processed_prompt.replace(template_marker, final_value)
                            processed_node_values[node_name] = final_value
                            continue
                            
                        # Try to extract just the number if it's a simple text answer
                        if isinstance(node_output, str):
                            # Look for patterns where the output is just a number with maybe some text
                            number_pattern = r'^\s*(\d+(\.\d+)?)\s*$'
                            match = re.search(number_pattern, node_output)
                            if match:
                                final_value = match.group(1)
                                self.log(f"Numeric value detected (pattern): {final_value}")
                                processed_prompt = processed_prompt.replace(template_marker, final_value)
                                processed_node_values[node_name] = final_value
                                continue
                    except Exception as e:
                        self.log(f"Error processing numeric node output: {e}")
                        # Fall back to normal processing on error
                    
                    # Default: use the raw content value directly
                    self.log(f"Using default string value: '{node_output}'")
                    processed_prompt = processed_prompt.replace(template_marker, node_output)
                    processed_node_values[node_name] = node_output
                
                # Also try case-insensitive matching
                normalized_node_name = node_name.lower().strip()
                for potential_reference in re.findall(r'\{([^:\}\[]+)(?:\[(\d+)\]|\:item\((\d+)\))?\}', processed_prompt):
                    potential_node_name = potential_reference[0]
                    if potential_node_name.lower().strip() == normalized_node_name and "{" + potential_node_name + "}" in processed_prompt:
                        template_marker = "{" + potential_node_name + "}"
                        self.log(f"Processing node reference using case-insensitive matching: {template_marker}")
                        self.log(f"  Matched to node: {node_name}")
                        self.log(f"  Original node output: '{node_output}'")
                        processed_prompt = processed_prompt.replace(template_marker, node_output)
                        processed_node_values[potential_node_name] = node_output
            
            # Step 4: Try to resolve any remaining references using the mapping
            if node_mapping:
                # Find all remaining unprocessed references
                for potential_reference in re.findall(r'\{([^:\}\[]+)(?:\[(\d+)\]|\:item\((\d+)\))?\}', processed_prompt):
                    potential_node_name = potential_reference[0]
                    template_marker = "{" + potential_node_name + "}"
                    
                    # Skip if we've already processed this reference
                    if potential_node_name in processed_node_values:
                        continue
                        
                    self.log(f"Trying to resolve remaining reference: {template_marker} using mapping")
                    
                    # Try to find a match in the mapping
                    for mapped_name, mapped_id in node_mapping.items():
                        if mapped_name.lower().strip() == potential_node_name.lower().strip():
                            # Found a match by name, now check if we have the ID data
                            id_key = f"id:{mapped_id}"
                            if id_key in context_data:
                                node_output = context_data[id_key]
                                self.log(f"  Resolved via mapping to ID {mapped_id}")
                                self.log(f"  Original node output: '{node_output}'")
                                processed_prompt = processed_prompt.replace(template_marker, node_output)
                                processed_node_values[potential_node_name] = node_output
                                break
            
            # Print summary 
            self.log(f"\n--- Template Processing Summary ---")
            self.log(f"Original prompt: {prompt_text}")
            self.log(f"Processed nodes:")
            for node_name, value in processed_node_values.items():
                self.log(f"  - {node_name}: '{value}'")
            self.log(f"Final processed prompt: {processed_prompt}")
            self.log(f"--- End Template Processing Summary ---\n")
            
        except Exception as e:
            self.log(f"Error processing template variables: {e}")
            import traceback
            traceback.print_exc()
            # Return original on severe error
            return prompt_text, {}
            
        return processed_prompt, processed_node_values

# Create a global instance of the template processor
template_processor = TemplateProcessor(debug_mode=True)

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
    if request.llm_config:
        node_config = LLMConfig(
            model=request.llm_config.model,
            temperature=request.llm_config.temperature,
            max_tokens=request.llm_config.max_tokens
        )

    print(f"--- Executing Single Text Generation (Model: {node_config.model}) ---")
    print(f"--- Original Prompt: {request.prompt_text} ---")
    if request.context_data:
        print(f"--- Context Data: ---")
        for node_name, content in request.context_data.items():
            print(f"Node: {node_name}")
            print(f"Content: {content[:200]}..." if len(content) > 200 else f"Content: {content}")
            print("-" * 40)

    # Enhanced template processing with better context
    processed_prompt, processed_node_values = template_processor.process_node_references(
        request.prompt_text, request.context_data, DataAccessor(request.context_data)
    )

    # Enhanced system message with guidance about node relationships and data structures
    system_content = "You are a helpful AI assistant working with connected nodes of information."
    
    if request.context_data:
        system_content += "\n\nYou have access to content from these nodes:\n" + "\n".join(
            f"- {node_name}: {content[:50]}..." if len(content) > 50 else f"- {node_name}: {content}"
            for node_name, content in request.context_data.items()
        )
        
        # Add explicit JSON data if detected
        for node_name, content in request.context_data.items():
            json_data = ContentParser.try_parse_json(content)
            if json_data:
                system_content += f"\n\nJSON data from {node_name}:\n```json\n{json.dumps(json_data, indent=2)}\n```"
                # Add specific instructions for this JSON
                if isinstance(json_data, dict) and "countries" in json_data:
                    countries = json_data.get("countries", [])
                    country_names = [country.get("name") for country in countries if isinstance(country, dict) and "name" in country]
                    if country_names:
                        system_content += f"\n\nCountries found in {node_name}: {', '.join(country_names)}"
                            
                        # Add specific guidance on accessing country populations
                        if any("population" in country for country in countries if isinstance(country, dict)):
                            system_content += f"\n\nTo access population data for a specific country, look up the country by name in the JSON data."
        
        system_content += "\n\nWhen answering:"
        system_content += "\n- Reference the specific content from nodes directly"
        system_content += "\n- If content contains numbered items, extract and use the exact items being referenced"
        system_content += "\n- If a question asks about a specific numbered item (e.g., 'item #2 from Node 1'), provide the exact corresponding item"
        system_content += "\n- When working with JSON data, use the exact values from the JSON structure"
        system_content += "\n- Analyze any lists, data, or information thoroughly"
        system_content += "\n- When working with tables, maintain proper formatting"
        system_content += "\n- If you can't find the referenced data, clearly state that it's not available"
        system_content += "\n- Ensure you're looking at the correct node when extracting information"

    print(f"\n=== FULL SYSTEM CONTENT ===\n{system_content}\n=== END SYSTEM CONTENT ===\n")
    
    # Prepare messages with enhanced system content
    messages_payload = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": processed_prompt}
    ]

    print(f"\n=== FULL MESSAGE PAYLOAD ===")
    for idx, msg in enumerate(messages_payload):
        print(f"Message {idx+1} ({msg['role']}):\n{msg['content']}\n---")
    print(f"=== END MESSAGE PAYLOAD ===\n")

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
            
            # Log the response content
            print(f"\n=== RESPONSE CONTENT ===\n{response_content}\n=== END RESPONSE CONTENT ===\n")

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

# --- Debug Endpoints ---
@app.get("/debug/node_content")
async def debug_node_content(node_content: str):
    """Debug endpoint to test content parsing functionality."""
    result = {
        "original_content": node_content,
        "analysis": {},
        "parsed_data": {}
    }
    
    # Analyze using ContentParser
    numbered_items = ContentParser.parse_numbered_list(node_content)
    json_data = ContentParser.try_parse_json(node_content)
    table_data = ContentParser.extract_table(node_content)
    
    # Build analysis
    result["analysis"] = {
        "has_numbered_list": bool(numbered_items),
        "numbered_items_count": len(numbered_items) if numbered_items else 0,
        "has_json": json_data is not None,
        "has_table": table_data is not None
    }
    
    # Add parsed data
    result["parsed_data"] = {
        "numbered_items": numbered_items,
        "json_data": json_data,
        "table_data": table_data
    }
    
    return result

@app.post("/debug/process_template")
async def debug_process_template(request: dict):
    """Debug endpoint to test template processing directly."""
    if "prompt" not in request or "context_data" not in request:
        raise HTTPException(status_code=400, detail="Request must include 'prompt' and 'context_data' fields")
    
    prompt = request["prompt"]
    context_data = request["context_data"]
    
    print(f"Debug process template request:")
    print(f"Prompt: {prompt}")
    print(f"Context data: {context_data}")
    
    # Process the template using our unified processor
    processed_prompt, processed_node_values = template_processor.process_node_references(
        prompt, context_data
    )
    
    # Return detailed results for debugging
    return {
        "original_prompt": prompt,
        "context_data": context_data,
        "processed_prompt": processed_prompt,
        "processed_node_values": processed_node_values,
        "validation": {
            "is_valid": len(template_processor.validate_node_references(prompt, context_data.keys())[1]) == 0,
            "missing_nodes": template_processor.validate_node_references(prompt, context_data.keys())[1],
            "found_nodes": template_processor.validate_node_references(prompt, context_data.keys())[2],
        }
    }

# --- Template Validation Endpoint ---
class TemplateValidationRequest(BaseModel):
    prompt_text: str
    available_nodes: List[str]

class TemplateValidationResponse(BaseModel):
    is_valid: bool
    missing_nodes: List[str]
    found_nodes: List[str]
    warnings: Optional[List[str]] = None

@app.post("/validate_template", response_model=TemplateValidationResponse)
async def validate_template_api(request: TemplateValidationRequest):
    """Validates that all node references in a template exist in the available nodes."""
    is_valid, missing_nodes, found_nodes = template_processor.validate_node_references(
        request.prompt_text, set(request.available_nodes)
    )
    
    warnings = []
    if not is_valid:
        for node in missing_nodes:
            warnings.append(f"Node reference '{node}' not found in available nodes.")
    
    return TemplateValidationResponse(
        is_valid=is_valid,
        missing_nodes=missing_nodes,
        found_nodes=found_nodes,
        warnings=warnings
    )

@app.post("/debug/test_reference")
async def debug_test_reference(request: dict):
    """Test endpoint for reference extraction."""
    if "content" not in request or "reference" not in request:
        raise HTTPException(status_code=400, detail="Request must include 'content' and 'reference' fields")
    
    content = request["content"]
    reference = request["reference"]
    
    # Create fake context with Node1 containing the content
    fake_context = {"Node1": content}
    data_accessor = DataAccessor(fake_context)
    
    # Try to parse the reference
    result = {
        "original_content": content,
        "reference": reference,
        "parsed_result": None,
        "details": {}
    }
    
    # Check if it's an item reference
    item_ref_pattern = r'\{([^:\}\[]+)(?:\[(\d+)\]|\:item\((\d+)\))\}'
    match = re.search(item_ref_pattern, reference)
    
    if match:
        node_name = match.group(1)
        item_num_str = match.group(2) or match.group(3)
        
        result["details"] = {
            "node_name": node_name,
            "item_num": item_num_str,
            "is_valid_node": data_accessor.has_node(node_name)
        }
        
        try:
            item_num = int(item_num_str)
            result["details"]["valid_item_num"] = True
            
            # Get the specific item
            item_content = data_accessor.get_item(node_name, item_num)
            result["parsed_result"] = item_content
            
        except ValueError:
            result["details"]["valid_item_num"] = False
    else:
        result["details"]["is_reference_pattern"] = False
    
    return result

# --- Run Server --- (Existing code)
if __name__ == "__main__":
    import uvicorn 
    print("Starting backend server at http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True) 
# ScriptChain: AI Processing Chain Backend

This application provides a backend service for creating, managing, and executing AI-driven processing chains. It allows users to define a series of interconnected "nodes," where each node performs a specific task, often leveraging Large Language Models (LLMs) like OpenAI's GPT series. The output of one node can be used as input for subsequent nodes, enabling complex, multi-step AI workflows.

## Core Concepts

### 1. Nodes
A **Node** is the fundamental building block of a processing chain. Each node represents a single unit of work and has the following key properties:

-   `node_id`: A unique identifier for the node (e.g., "summarizer", "translator", "data_extractor").
-   `node_type`: Specifies the kind of operation the node performs. Current supported types include:
    -   `text_generation`: Generates text using an LLM based on a prompt and context.
    -   `decision_making`: Uses an LLM to analyze a scenario and make a decision based on provided values or criteria.
    -   `retrieval`: Retrieves data that was output by a previous node in the chain.
    -   `logic_chain`: Performs multi-step logical reasoning using an LLM.
-   `input_keys`: A list of data keys that this node expects as input. These inputs are typically sourced from the outputs of upstream nodes or provided as initial data to the chain.
-   `output_keys`: A list of data keys that this node will produce as output. This output is stored and can be used by downstream nodes.
-   `model_config`: (Optional) Allows specifying a custom LLM configuration (model, temperature, max_tokens) for this particular node. If not provided, a default configuration is used.
-   `template`: (Optional) A template string that can be used to format the input data for the node, often used to construct prompts for LLMs by dynamically inserting values from previous nodes.

### 2. Edges (Connections)
**Edges** define the dependencies and flow of data between nodes. An edge from `NodeA` to `NodeB` means that `NodeA` must execute before `NodeB`, and the output of `NodeA` can potentially be used as input for `NodeB`.

The application uses a directed acyclic graph (DAG) to represent the chain, ensuring that there are no circular dependencies.

### 3. ScriptChain
The `ScriptChain` class orchestrates the execution of the defined graph of nodes. It performs the following key functions:

-   **Node Management**: Allows adding nodes and defining edges between them.
-   **Topological Sort**: Before execution, it determines the correct order in which nodes must be processed based on their dependencies. This ensures that a node only runs after all its prerequisite input nodes have completed.
-   **Data Storage & Propagation**: It manages a `NamespacedStorage` system where each node's output is stored under its unique `node_id`. This allows downstream nodes to access specific outputs from specific upstream nodes.
    -   **Namespacing**: Data is stored like `storage[node_id][output_key] = value`.
    -   **Input Assembly**: When a node is about to be processed, the `ScriptChain` gathers the necessary inputs for that node from the storage, based on its `input_keys` and the outputs of its direct predecessors.
-   **Execution**: Iterates through the nodes in the determined topological order, calling the `process()` method of each node.
-   **Callbacks**: Supports a callback system (`on_node_start`, `on_node_complete`, `on_chain_complete`) that allows external functions to be notified about the progress and results of the chain execution. The `LoggingCallback` is a default implementation that prints this information to the console.
-   **Token Usage & Cost Tracking**: For nodes that make calls to LLMs (like OpenAI), it tracks the number of prompt and completion tokens used and estimates the cost of the API calls.
-   **Versioning & Caching (Basic)**: Includes a basic versioning system (`increment_node_version`, `node_needs_update`) to track changes in node outputs. If an upstream node's output changes, dependent downstream nodes might be marked for re-execution. (This is a foundational aspect for potential future caching improvements).

### 4. API Endpoints
The application exposes a FastAPI backend with several key endpoints to interact with the `ScriptChain`:

-   **`/add_node`**: Adds a new node to a session's script chain.
-   **`/add_edge`**: Defines a dependency (edge) between two existing nodes.
-   **`/execute`**: Triggers the execution of the entire defined script chain for a session. It takes optional initial inputs that can be fed into the first node(s).
-   **`/generate_text_node`**: A specialized endpoint for executing a single text generation task without defining a full chain. It can take context data from other (hypothetical or existing) nodes. This is useful for more direct interactions or when a UI wants to dynamically update a single node's content.
-   **`/get_node_outputs`**: Retrieves the stored output for specified nodes.
-   **Debugging Endpoints**: Several `/debug/...` endpoints are available for testing template processing and content parsing.

## How Nodes are Connected and Data Flows

1.  **Define Nodes**: You first define individual nodes, specifying their ID, type, and what input data keys they expect and what output keys they will produce.
    *Example*:
    -   Node A (`text_generation`): Takes `original_text` as input, produces `summary` as output.
    -   Node B (`text_generation`): Takes `summary` as input (from Node A), produces `translation` as output.

2.  **Define Edges**: You then connect these nodes by adding edges. An edge from Node A to Node B signifies that Node B depends on Node A.
    *Example*: `add_edge("NodeA", "NodeB")`

3.  **Execution**: When the `/execute` endpoint is called:
    -   The `ScriptChain` determines that Node A must run before Node B.
    -   Node A processes its input (e.g., `original_text` provided initially or from another source).
    -   The output of Node A (e.g., `{"summary": "This is a summary."}`) is stored in the namespaced storage associated with "NodeA".
    -   When Node B is processed, the `ScriptChain` looks at its `input_keys` (which includes "summary"). It retrieves the value of "summary" from "NodeA"'s output in the storage.
    -   Node B then processes this summary to produce its `translation`.
    -   The output of Node B is stored similarly.

4.  **Templating**: Nodes, especially those performing text generation, can use templates. A template for Node B might look like: `"Translate this summary into French: {summary}"`.
    -   During Node B's processing, the `{summary}` placeholder is dynamically replaced with the actual summary content retrieved from Node A's output. The `TemplateProcessor` handles resolving these references, looking up values in the `context_data` which includes outputs from upstream nodes.

## Key Functionality Summary

-   **Modular AI Workflows**: Break down complex AI tasks into smaller, manageable, and reusable nodes.
-   **Dependency Management**: Automatically handles the order of execution based on how nodes are connected.
-   **Data Propagation**: Seamlessly passes data (outputs) from one node to the next (inputs).
-   **LLM Integration**: Primarily designed for tasks involving LLMs, with built-in support for OpenAI models.
-   **Configuration Flexibility**: Allows default and per-node LLM configurations.
-   **Extensibility**: The `Node` types and `ScriptChain` logic can be extended to support new operations or integrations.
-   **Session Management (Basic)**: Chains are managed per `session_id`, allowing multiple independent chains to exist.

This system is designed to be a flexible backend for applications that need to orchestrate sequences of AI-driven tasks, particularly those involving natural language processing and generation.

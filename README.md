# ScriptChain: Visual AI Workflow Builder

ScriptChain is a powerful visual interface for creating, connecting, and executing AI-powered workflows. It allows you to build directed acyclic graphs (DAGs) of AI processing nodes that can pass context and content between each other, enabling sophisticated multi-step AI processing pipelines.



## Features

- **Visual Workflow Builder:** Drag, connect, and configure AI nodes in an intuitive interface.
- **Multi-Input Node Support:** Nodes can receive input from multiple upstream nodes.
- **Context Passing:** Output from one node can be used as context for downstream nodes.
- **Template Variable System:** Insert `{NodeName}` references that get replaced with node outputs.
- **Variable Clicking:** Simply click on connected nodes to insert their variable templates.
- **DAG Enforcement:** Strict directed acyclic graph rules prevent circular dependencies.
- **OpenAI Integration:** Powered by OpenAI's GPT models with full configuration options.
- **Token Usage Tracking:** Monitor token usage and estimated costs for each operation.

## Installation

### Prerequisites

- Node.js (v16+)
- Python (v3.8+)
- OpenAI API key

### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create `.env` file with your OpenAI API key:**
   ```bash
   echo "OPENAI_API_KEY=your-api-key" > .env
   ```

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

## Running the Application

### Start the Backend

```bash
cd backend
python main.py
```

The API server will run at http://localhost:8000.

### Start the Frontend

```bash
cd frontend
npm run dev
```

The web interface will be available at http://localhost:5173.

## Usage Guide

### Creating Your First Workflow

1. **Add Nodes:** Click "Add Node" to create processing nodes in your workflow.
2. **Connect Nodes:** Drag connections from a node's output (bottom) to another node's input (top).
3. **Configure Nodes:** Enter prompts in each node's text area.
4. **Reference Other Nodes:** For a node that receives input, check the box beside an input node and click its button to insert a `{NodeName}` reference into your prompt.
5. **Run Nodes:** Click "Run" to execute a single node or "Execute Flow" to run the entire workflow.

### Example: Creating a Story Generator

1. Create "Character" node with prompt: "Create a protagonist character profile for a sci-fi story."
2. Create "Setting" node with prompt: "Describe a futuristic setting for a sci-fi story."
3. Connect both to a new "Story" node.
4. In "Story" node, click to insert references: "Write a short story using {Character} in the setting of {Setting}."
5. Check both input nodes.
6. Run "Story" node to generate a cohesive narrative that incorporates both elements.

## Architecture

ScriptChain is built on a modern stack with a clear separation between frontend and backend:

### Backend

- **FastAPI:** High-performance API framework.
- **NetworkX:** Graph management and operations.
- **OpenAI API:** Integration with AI language models.
- **Pydantic:** Request/response validation and serialization.

### Frontend

- **React:** Component-based UI framework.
- **ReactFlow:** Interactive node-based interface.
- **CSS Variables:** For consistent theming and styling.
- **Fetch API:** For backend communication.

## Key Components

### Nodes

Each node represents a processing step with:
- Inputs (from upstream nodes)
- Prompt template
- Processing logic
- Output generation

### Edges

Connections between nodes define:
- Data flow direction
- Execution order dependencies
- Context availability

### Template System

The `{NodeName}` syntax provides:
- References to other nodes' outputs
- Clear visual indication of dependencies
- Automatic content substitution at runtime

## Plans and Vision for Enhancements

### Immediate Roadmap

1. **Save and Load Workflows**
   - Save graph configurations to JSON
   - Load previously created workflows
   - Share workflows between users

2. **Additional Node Types**
   - Data retrieval nodes (web, databases)
   - Branching/conditional nodes based on content analysis
   - Aggregator nodes that combine multiple inputs with custom logic

3. **Enhanced Context Processing**
   - Smarter template processing with nested references
   - Context windowing for handling large outputs
   - Metadata retention across node connections

### Medium-Term Vision

1. **User Management**
   - Multi-user support with authentication
   - Shared workflow libraries
   - Permission systems for collaborative workflows

2. **Advanced UI Features**
   - Node grouping and subflows
   - Visual history/version control
   - Real-time collaboration

3. **Performance Optimizations**
   - Caching frequently used outputs
   - Smart re-execution (only affected nodes)
   - Parallel execution where possible

### Long-Term Possibilities

1. **AI Agent Integration**
   - Autonomous workflow adjustment
   - Self-healing error recovery
   - Auto-generation of optimal workflows

2. **Multimodal Support**
   - Image generation/processing nodes
   - Audio input/output nodes
   - Video processing capabilities

3. **Enterprise Features**
   - Workflow approval processes
   - Usage monitoring and quotas
   - Compliance and audit logging

### Community-Focused Developments

1. **Workflow Marketplace**
   - Share and discover useful workflows
   - Rating and commenting system
   - Monetization options for workflow creators

2. **Plugin System**
   - Extend functionality with custom nodes
   - Third-party integrations
   - API connectors for external services

3. **Educational Resources**
   - Tutorial workflows
   - Best practices documentation
   - Pattern libraries for common AI tasks

---

ScriptChain aims to become the definitive platform for visual AI workflow creation, making complex AI orchestration accessible to technical and non-technical users alike. By focusing on intuitive UI, robust execution, and extensibility, we're building a system that grows with the rapidly evolving AI landscape. 
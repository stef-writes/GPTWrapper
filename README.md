# ScriptChain: Visual AI Workflow Builder

ScriptChain is a powerful visual interface for creating, connecting, and executing AI-powered workflows. It allows you to build directed acyclic graphs (DAGs) of AI processing nodes that can pass context and content between each other, enabling sophisticated multi-step AI processing pipelines.



## Features

- **Visual Workflow Builder:** Drag, connect, and configure AI nodes in an intuitive interface.
- **Multi-Input Node Support:** Nodes can receive input from multiple upstream nodes.
- **Context Passing:** Output from one node can be used as context for downstream nodes.
- **Template Variable System:** Insert `{NodeName}` references that get replaced with node outputs.
- **Advanced Data References:** Access specific items with `{NodeName[n]}` syntax for lists, tables and JSON.
- **Intelligent Data Parsing:** Automatic detection and handling of structured data (lists, JSON, tables).
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

### Working with Structured Data

ScriptChain now intelligently handles structured data between nodes:

1. **Numbered Lists:** Access specific items with `{NodeName[n]}` syntax
   - Example: `{CityList[2]}` will extract item #2 from the "CityList" node

2. **JSON Data:** The system automatically parses JSON in node outputs
   - Reference JSON directly in your prompts
   - The system will present structured JSON data to the LLM with proper formatting

3. **Combined References:** Mix and match structured data from multiple nodes
   - Example: `What is the population of the country where {CityList[3]} is located?`

### Example: Multi-Step Data Analysis

1. Create a "DataGenerator" node that outputs a numbered list or JSON
2. Create an "Analyzer" node that references specific items: `Analyze item {DataGenerator[2]}`
3. The system will automatically extract just the referenced item

## Architecture

ScriptChain is built on a modern stack with a clear separation between frontend and backend:

### Backend

- **FastAPI:** High-performance API framework.
- **NetworkX:** Graph management and operations.
- **OpenAI API:** Integration with AI language models.
- **Pydantic:** Request/response validation and serialization.
- **ContentParser:** Intelligent parsing and extraction of structured data.
- **DataAccessor:** Advanced data manipulation helpers for node interactions.

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

The template system provides multiple ways to reference data:
- Basic references: `{NodeName}` - inserts full node output
- Item references: `{NodeName[n]}` - inserts specific numbered items
- The system automatically detects data structures in node outputs
- LLM is guided with context-aware instructions based on data types

### Data Handling

ScriptChain's data parsing capabilities include:
- **Numbered Lists:** Automatically detected and indexed 
- **JSON Data:** Parsed and made accessible by structure
- **Tables:** Identified and preserved in formatting
- **Content Analysis:** Automatic detection of data types

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
   - ✅ Smarter template processing with nested references
   - ✅ Advanced data structure handling (lists, JSON, tables)
   - ✅ Intelligent data extraction and referencing
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
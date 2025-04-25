import React, { useState, useCallback } from 'react';
import ReactFlow, {
  Controls, // Adds zoom/pan controls
  Background, // Adds a background pattern
  applyNodeChanges, // Helper for node state updates
  applyEdgeChanges, // Helper for edge state updates
  addEdge, // Helper for adding edges
  ReactFlowProvider, // Import the provider
} from 'reactflow';

// Import React Flow CSS
import 'reactflow/dist/style.css';

// Import our App specific CSS
import './App.css';

// Initial node (example)
const initialNodes = [
  {
    id: '1', // Needs to be unique string
    position: { x: 100, y: 100 },
    data: { label: 'Node 1 (Placeholder)' }, // Data associated with the node
    // type: 'customNode' // We'll define custom nodes later
  },
];

// Initial edges (example)
const initialEdges = [];

function App() {
  // State for nodes and edges
  const [nodes, setNodes] = useState(initialNodes);
  const [edges, setEdges] = useState(initialEdges);

  // Handlers for node/edge changes (drag, select, remove)
  const onNodesChange = useCallback(
    (changes) => setNodes((nds) => applyNodeChanges(changes, nds)),
    [setNodes]
  );
  const onEdgesChange = useCallback(
    (changes) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    [setEdges]
  );

  // Handler for connecting nodes
  const onConnect = useCallback(
    (connection) => setEdges((eds) => addEdge(connection, eds)),
    [setEdges]
  );

  return (
    // Wrap the main div with ReactFlowProvider
    <ReactFlowProvider>
      <div className="reactflow-wrapper">
        <ReactFlow
          nodes={nodes}               // Pass nodes state
          edges={edges}               // Pass edges state
          onNodesChange={onNodesChange} // Handler for node changes
          onEdgesChange={onEdgesChange} // Handler for edge changes
          onConnect={onConnect}       // Handler for creating edges
          fitView                     // Zooms/pans to fit nodes on initial render
          // nodeTypes={nodeTypes} // We will add custom node types here later
        >
          <Background /> {/* Adds dot pattern background */}
          <Controls />   {/* Adds zoom/pan controls */}
          {/* <MiniMap /> Optional minimap */}
        </ReactFlow>
      </div>
    </ReactFlowProvider>
  );
}

export default App;

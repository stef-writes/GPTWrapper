import React, { useState, useCallback, useRef } from 'react';
import ReactFlow, {
  Controls, // Adds zoom/pan controls
  Background, // Adds a background pattern
  applyNodeChanges, // Helper for node state updates
  applyEdgeChanges, // Helper for edge state updates
  addEdge, // Helper for adding edges
  ReactFlowProvider, // Import the provider
  Panel,
  useReactFlow,
} from 'reactflow';
import { v4 as uuidv4 } from 'uuid';

// Import React Flow CSS
import 'reactflow/dist/style.css';

// Import our App specific CSS
import './App.css';

// Import custom node types
import nodeTypes from './components/CustomNodeTypes';

// Import NodeService for API calls
import NodeService from './services/NodeService';

import Node from './components/Node';

function Flow() {
  const reactFlowWrapper = useRef(null);
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [reactFlowInstance, setReactFlowInstance] = useState(null);
  const [nodeOutputs, setNodeOutputs] = useState({});
  const [isConnecting, setIsConnecting] = useState(false);

  // Handler for node changes (drag, select, remove)
  const onNodesChange = useCallback(
    (changes) => setNodes((nds) => applyNodeChanges(changes, nds)),
    []
  );

  // Handler for edge changes (select, remove)
  const onEdgesChange = useCallback(
    (changes) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    []
  );

  // Handler for connecting nodes
  const onConnect = useCallback(
    async (connection) => {
      try {
        setIsConnecting(true);
        // Create the edge in the UI
        setEdges((eds) => addEdge(connection, eds));
        const sourceNodeId = connection.source;
        const targetNodeId = connection.target;
        // Add the edge to the backend
        await NodeService.addEdge(sourceNodeId, targetNodeId);
        // Update available variables for the target node (multi-input support)
        setNodes((nds) => 
          nds.map((node) => {
            if (node.id === targetNodeId) {
              const sourceNode = nds.find(n => n.id === sourceNodeId);
              const sourceNodeName = sourceNode?.data?.nodeName || 'Unnamed Node';
              const updatedVariables = [
                ...(node.data.variables || []),
                {
                  id: sourceNodeId,
                  name: sourceNodeName
                }
              ];
              // Remove duplicates
              const uniqueVariables = updatedVariables.filter(
                (variable, index, self) => 
                  index === self.findIndex(v => v.id === variable.id)
              );
              return {
                ...node,
                data: {
                  ...node.data,
                  variables: uniqueVariables
                }
              };
            }
            return node;
          })
        );
      } catch (error) {
        // Show backend error (cycle prevention, etc.)
        alert(error.message || 'Failed to connect nodes.');
        // Revert the UI change on error
        setEdges((eds) => eds.filter(e => 
          !(e.source === connection.source && e.target === connection.target)
        ));
      } finally {
        setIsConnecting(false);
      }
    },
    []
  );

  // Add a new node when the button is clicked
  const onAddNode = useCallback(async () => {
    if (!reactFlowInstance) return;
    
    try {
      // Get the center position of the viewport
      const { x, y, zoom } = reactFlowInstance.getViewport();
      const centerX = (window.innerWidth / 2 - x) / zoom;
      const centerY = (window.innerHeight / 2 - y) / zoom;
      
      const id = uuidv4();
      const nodeName = `Node ${nodes.length + 1}`;
      
      // Add the node to the backend first
      await NodeService.addNode(
        id,
        'text_generation', // Default node type
        ['context', 'query'], // Default input keys
        ['generated_text'], // Default output keys
      );
      
      // Then add to the UI if successful
      const newNode = {
        id,
        type: 'llmNode',
        position: { x: centerX, y: centerY },
        data: {
          nodeName,
          prompt: '',
          output: '',
          variables: [],
          onNameChange: (name) => {
            // Update node name
            setNodes((nds) =>
              nds.map((node) => {
                if (node.id === id) {
                  return {
                    ...node,
                    data: {
                      ...node.data,
                      nodeName: name,
                    },
                  };
                }
                return node;
              })
            );
            
            // Update node name in other nodes' variables
            setNodes((nds) =>
              nds.map((node) => {
                if (node.id !== id && node.data.variables) {
                  const updatedVariables = node.data.variables.map(variable => {
                    if (variable.id === id) {
                      return {
                        ...variable,
                        name: name
                      };
                    }
                    return variable;
                  });
                  
                  return {
                    ...node,
                    data: {
                      ...node.data,
                      variables: updatedVariables
                    }
                  };
                }
                return node;
              })
            );
          },
          onPromptChange: (prompt) => {
            setNodes((nds) =>
              nds.map((node) => {
                if (node.id === id) {
                  return {
                    ...node,
                    data: {
                      ...node.data,
                      prompt,
                    },
                  };
                }
                return node;
              })
            );
          },
          onOutputChange: (output) => {
            setNodeOutputs(prev => ({
              ...prev,
              [id]: output
            }));
          },
          onVariableSelect: (variableIds) => {
            // Store the selected variable IDs for state management
            setNodes((nds) =>
              nds.map((node) => {
                if (node.id === id) {
                  return {
                    ...node,
                    data: {
                      ...node.data,
                      selectedVariableIds: variableIds, // Store which are selected
                    },
                  };
                }
                return node;
              })
            );
          },
          onRun: async (prompt, activeInputNodeIds) => {
            console.log(`Running node ${id} (name: ${nodeName}) with active inputs:`, activeInputNodeIds);
            
            // Prepare context data based ONLY on actively selected nodes (checked in UI)
            let contextData = {};
            if (activeInputNodeIds && activeInputNodeIds.length > 0) {
              console.log("Active input nodes detail:");
              activeInputNodeIds.forEach(inputId => {
                const inputNodeOutput = nodeOutputs[inputId] || '';
                const sourceNode = nodes.find(n => n.id === inputId);
                const inputName = sourceNode?.data?.nodeName || inputId;
                
                console.log(`  - Node ID: ${inputId}`);
                console.log(`    Node Name: ${inputName}`);
                console.log(`    Node Output: ${inputNodeOutput.substring(0, 50)}${inputNodeOutput.length > 50 ? '...' : ''}`);
              
                // Build the context data with the output from selected nodes
                contextData[inputName] = inputNodeOutput;
              });
            }
            
            console.log("Context data being sent to backend:", contextData);
            console.log("Prompt with templates:", prompt);
            
            try {
              // Send the prompt text (with {NodeName} templates) and context data separately
              // The backend can use contextData to replace templates or as additional context
              const response = await NodeService.generateText(prompt, null, contextData);
              
              setNodeOutputs(prev => ({
                ...prev,
                [id]: response.generated_text
              }));
              
              return response.generated_text;
            } catch (error) {
              console.error('Error generating text:', error);
              return `Error: ${error.message}`;
            }
          },
        },
      };
      
      setNodes((nds) => [...nds, newNode]);
    } catch (error) {
      console.error('Failed to add node:', error);
    }
  }, [reactFlowInstance, nodes, nodeOutputs]);

  const onDragOver = useCallback((event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  // Execute the entire flow
  const executeFlow = useCallback(async () => {
    try {
      // Prepare initial data for the chain execution
      const initialInputs = {};
      
      // For each node, find if it has any connected input nodes
      // and prepare the context accordingly
      edges.forEach(edge => {
        const sourceNode = nodes.find(n => n.id === edge.source);
        const targetNode = nodes.find(n => n.id === edge.target);
        
        if (sourceNode && targetNode) {
          const sourceOutput = nodeOutputs[sourceNode.id] || '';
          
          // Add this as input for the ScriptChain execution
          // Using the proper input keys that the ScriptChain expects
          initialInputs[`context_${targetNode.id}`] = sourceOutput;
        }
      });
      
      // Execute the entire flow with these inputs
      const result = await NodeService.executeChain(initialInputs);
      console.log('Flow execution results:', result);
      
      // Update node outputs based on results
      if (result.results) {
        const newNodeOutputs = { ...nodeOutputs };
        
        Object.entries(result.results).forEach(([nodeId, nodeResult]) => {
          // Only update if the node exists in our flow
          if (nodes.some(node => node.id === nodeId)) {
            // Update the output in the state
            const outputText = nodeResult.generated_text || 
                              nodeResult.decision_output || 
                              nodeResult.reasoning_result || 
                              nodeResult.retrieved_data || 
                              JSON.stringify(nodeResult);
            
            newNodeOutputs[nodeId] = outputText;
            
            // Update node display
            setNodes(prevNodes => 
              prevNodes.map(node => {
                if (node.id === nodeId) {
                  return {
                    ...node,
                    data: {
                      ...node.data,
                      output: outputText
                    }
                  };
                }
                return node;
              })
            );
          }
        });
        
        setNodeOutputs(newNodeOutputs);
      }
    } catch (error) {
      console.error('Error executing flow:', error);
    }
  }, [nodes, edges, nodeOutputs]);

  // Helper to find DIRECTLY connected input nodes
  const findDirectInputNodes = (nodeId, nodes, edges) => {
    const inputEdges = edges.filter(edge => edge.target === nodeId);
    const inputNodeIds = inputEdges.map(edge => edge.source);
    return nodes
      .filter(node => inputNodeIds.includes(node.id))
      .map(node => ({ id: node.id, name: node.data.nodeName || 'Unnamed Node' }));
  };

  // Helper: Find all downstream nodes from a given node (DFS) - Keep for potential future use or validation
  const findDownstreamNodes = (nodeId, edges) => {
    const visited = new Set();
    const stack = [nodeId];
    while (stack.length > 0) {
      const current = stack.pop();
      edges.forEach(edge => {
        if (edge.source === current && !visited.has(edge.target)) {
          visited.add(edge.target);
          stack.push(edge.target);
        }
      });
    }
    return visited;
  };

  return (
    <div className="reactflow-wrapper" ref={reactFlowWrapper}>
      <ReactFlow
        nodes={nodes.map(node => {
          // *** CHANGE HERE: Use findDirectInputNodes ***
          const directInputNodes = findDirectInputNodes(node.id, nodes, edges);
          // Compute valid input nodes for this node (all nodes not self, not downstream) - No longer used for dropdown, kept for reference
          // const downstream = findDownstreamNodes(node.id, edges);
          // const potentialUpstreamNodes = nodes
          //   .filter(n => n.id !== node.id && !downstream.has(n.id))
          //   .map(n => ({ id: n.id, name: n.data.nodeName || 'Unnamed Node' }));
          return {
            ...node,
            data: {
              ...node.data,
              validInputNodes: directInputNodes, // Pass only DIRECT inputs
            },
            type: node.type,
          };
        })}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onInit={setReactFlowInstance}
        onDragOver={onDragOver}
        nodeTypes={nodeTypes}
        fitView
      >
        <Background />
        <Controls />
        <Panel position="top-right">
          <button className="add-node-button" onClick={onAddNode}>
            Add Node
          </button>
          {nodes.length > 1 && (
            <button 
              className="execute-flow-button" 
              onClick={executeFlow}
              style={{ marginLeft: '10px' }}
            >
              Execute Flow
            </button>
          )}
        </Panel>
      </ReactFlow>
    </div>
  );
}

function App() {
  return (
    <ReactFlowProvider>
      <Flow />
    </ReactFlowProvider>
  );
}

export default App;

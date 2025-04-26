import React, { useState, useEffect, useRef } from 'react';
import { Handle, Position } from 'reactflow';
import './Node.css';

const Node = ({ data, isConnectable }) => {
  const validInputNodes = data.validInputNodes || [];
  const [nodeName, setNodeName] = useState(data.nodeName || 'Node Name');
  const [prompt, setPrompt] = useState(data.prompt || '');
  const [output, setOutput] = useState(data.output || '');
  const [isRunning, setIsRunning] = useState(false);
  const [selectedInputNodes, setSelectedInputNodes] = useState([]);
  const [showTooltip, setShowTooltip] = useState(false);
  
  const promptRef = useRef(null);
  const outputRef = useRef(null);
  
  // Handle node name change
  const handleNameChange = (e) => {
    const newName = e.target.value;
    setNodeName(newName);
    if (data.onNameChange) {
      data.onNameChange(newName);
    }
  };
  
  // Handle prompt change
  const handlePromptChange = (e) => {
    const newPrompt = e.target.value;
    setPrompt(newPrompt);
    if (data.onPromptChange) {
      data.onPromptChange(newPrompt);
    }
  };
  
  // Handle run button click
  const handleRun = async () => {
    if (isRunning) return;
    setIsRunning(true);
    if (data.onRun) {
      try {
        // Pass the prompt AND the list of actively selected input node IDs
        const result = await data.onRun(prompt, selectedInputNodes);
        setOutput(result);
        if (data.onOutputChange) {
          data.onOutputChange(result);
        }
      } catch (error) {
        setOutput(`Error: ${error.message}`);
      } finally {
        setIsRunning(false);
      }
    } else {
      // Mock response for testing
      setTimeout(() => {
        setOutput('AI generated output would appear here...');
        setIsRunning(false);
      }, 1000);
    }
  };
  
  // Insert variable template into prompt textarea
  const insertVariable = (varName) => {
    if (!promptRef.current) return;
    
    const textarea = promptRef.current;
    const cursorPos = textarea.selectionStart;
    const textBefore = prompt.substring(0, cursorPos);
    const textAfter = prompt.substring(cursorPos);
    
    // Create template with spaces for better formatting
    const template = `{${varName}}`;
    
    // Create new prompt with inserted template
    const newPrompt = textBefore + template + textAfter;
    
    // Update prompt state
    setPrompt(newPrompt);
    
    // Notify parent of the change
    if (data.onPromptChange) {
      data.onPromptChange(newPrompt);
    }
    
    // Set focus back to textarea and position cursor after inserted template
    setTimeout(() => {
      textarea.focus();
      const newCursorPos = cursorPos + template.length;
      textarea.setSelectionRange(newCursorPos, newCursorPos);
    }, 0);
  };
  
  // Toggle whether a node is selected for context
  const toggleInputNode = (nodeId) => {
    setSelectedInputNodes(prev => {
      const isSelected = prev.includes(nodeId);
      const newSelection = isSelected 
        ? prev.filter(id => id !== nodeId) 
        : [...prev, nodeId];
      
      // Notify parent of the change
      if (data.onVariableSelect) {
        data.onVariableSelect(newSelection);
      }
      
      return newSelection;
    });
  };
  
  // Auto-resize textareas
  useEffect(() => {
    if (promptRef.current) {
      promptRef.current.style.height = 'auto';
      promptRef.current.style.height = `${promptRef.current.scrollHeight}px`;
    }
    if (outputRef.current) {
      outputRef.current.style.height = 'auto';
      outputRef.current.style.height = `${outputRef.current.scrollHeight}px`;
    }
  }, [prompt, output]);
  
  // Initialize selected input nodes from prop
  useEffect(() => {
    if (data.selectedVariableIds && data.selectedVariableIds.length > 0) {
      setSelectedInputNodes(data.selectedVariableIds);
    }
  }, [data.selectedVariableIds]);
  
  return (
    <div className="node">
      {/* Target handle (Input) - Top */}
      <Handle
        type="target"
        position={Position.Top}
        id="top"
        style={{ background: 'var(--color-primary)', width: '10px', height: '10px' }}
        isConnectable={isConnectable}
      />
      
      {/* Header with node name and run button */}
      <div className="node-header">
        <input
          type="text"
          className="node-name-input"
          value={nodeName}
          onChange={handleNameChange}
          placeholder="Node Name"
        />
        <button 
          className={`run-button ${isRunning ? 'running' : ''}`}
          onClick={handleRun}
          disabled={isRunning}
        >
          {isRunning ? 'Running...' : 'Run'}
        </button>
      </div>
      
      {/* New Connected Nodes Section */}
      <div className="connected-nodes-section">
        <div className="section-header">
          <span>Input from connected nodes:</span>
          <button 
            className="help-button" 
            onClick={() => setShowTooltip(!showTooltip)}
            type="button"
          >
            ?
          </button>
        </div>
        
        {showTooltip && (
          <div className="tooltip">
            Click any connected node below to insert its variable <code>{"{NodeName}"}</code> into your prompt. 
            The checked nodes will have their outputs included as context when you run this node.
          </div>
        )}
        
        {validInputNodes.length > 0 ? (
          <div className="connected-nodes-list">
            {validInputNodes.map(node => (
              <div key={node.id} className="connected-node-item">
                <label className="connected-node-checkbox">
                  <input
                    type="checkbox"
                    checked={selectedInputNodes.includes(node.id)}
                    onChange={() => toggleInputNode(node.id)}
                  />
                  <span>Use as input</span>
                </label>
                <button
                  className="insert-variable-button"
                  onClick={() => insertVariable(node.name)}
                  title={`Click to insert {${node.name}} at cursor position`}
                >
                  {node.name} <span className="insert-icon">→</span>
                </button>
              </div>
            ))}
          </div>
        ) : (
          <div className="no-connections-message">
            No connected input nodes. Connect nodes to this node's input first.
          </div>
        )}
      </div>
      
      {/* Prompt textarea */}
      <textarea
        ref={promptRef}
        className="prompt-textarea"
        value={prompt}
        onChange={handlePromptChange}
        placeholder="Type your prompt here. Click on a connected node above to insert it into your prompt."
      />
      
      {/* Output textarea */}
      <textarea
        ref={outputRef}
        className="output-textarea"
        value={output}
        readOnly
        placeholder="AI Generated Output Here..."
      />
      
      {/* Source handle (Output) - Bottom */}
      <Handle
        type="source"
        position={Position.Bottom}
        id="bottom"
        style={{ background: 'var(--color-primary)', width: '10px', height: '10px' }}
        isConnectable={isConnectable}
      />
    </div>
  );
};

export default Node; 
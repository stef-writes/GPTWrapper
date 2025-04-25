import React, { useState, useEffect, useRef } from 'react';
import { Handle, Position } from 'reactflow';
import './Node.css';

const Node = ({ data, isConnectable }) => {
  const [nodeName, setNodeName] = useState(data.nodeName || 'Node Name');
  const [prompt, setPrompt] = useState(data.prompt || '');
  const [output, setOutput] = useState(data.output || '');
  const [isRunning, setIsRunning] = useState(false);
  const [variables, setVariables] = useState(data.variables || []);
  const [selectedVariable, setSelectedVariable] = useState(null);
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
        const result = await data.onRun(prompt, selectedVariable);
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
  
  // Handle variable selection
  const handleVariableSelect = (variableId) => {
    if (variableId) {
      setSelectedVariable(variableId);
      // Insert variable template into prompt if not already present
      const selectedVar = variables.find(v => v.id === variableId);
      if (selectedVar && !prompt.includes(`{${selectedVar.name}}`)) {
        const newPrompt = prompt ? 
          `${prompt}\n\nInclude the following: {${selectedVar.name}}` : 
          `Include the following: {${selectedVar.name}}`;
        setPrompt(newPrompt);
        if (data.onPromptChange) {
          data.onPromptChange(newPrompt);
        }
      }
    } else {
      setSelectedVariable(null);
    }
    
    if (data.onVariableSelect) {
      data.onVariableSelect(variableId);
    }
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
  
  // Update variables when they change from parent
  useEffect(() => {
    if (data.variables && data.variables !== variables) {
      setVariables(data.variables);
    }
  }, [data.variables]);
  
  return (
    <div className="node">
      {/* Source handle (top) */}
      <Handle
        type="source"
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
      
      {/* Variable dropdown */}
      <div className="variable-dropdown">
        <div className="variable-dropdown-header">
          <label htmlFor="variable-select">Input from connected nodes:</label>
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
            Select a connected node to use its output. 
            The pattern <code>{"{NodeName}"}</code> in your prompt will be 
            replaced with that node's output when you run this node.
          </div>
        )}
        
        <select 
          id="variable-select"
          value={selectedVariable || ''}
          onChange={(e) => handleVariableSelect(e.target.value)}
        >
          <option value="">-- Select an input source --</option>
          {variables.map((variable) => (
            <option key={variable.id} value={variable.id}>
              {variable.name}
            </option>
          ))}
        </select>
        
        {selectedVariable && variables.length > 0 && (
          <div className="variable-usage-hint">
            Use <code>{`{${variables.find(v => v.id === selectedVariable)?.name || ''}}`}</code> in your prompt
          </div>
        )}
      </div>
      
      {/* Prompt textarea */}
      <textarea
        ref={promptRef}
        className="prompt-textarea"
        value={prompt}
        onChange={handlePromptChange}
        placeholder="Type your prompt here. Include {NodeName} to reference a connected node's output."
      />
      
      {/* Output textarea */}
      <textarea
        ref={outputRef}
        className="output-textarea"
        value={output}
        readOnly
        placeholder="AI Generated Output Here..."
      />
      
      {/* Target handle (bottom) */}
      <Handle
        type="target"
        position={Position.Bottom}
        id="bottom"
        style={{ background: 'var(--color-primary)', width: '10px', height: '10px' }}
        isConnectable={isConnectable}
      />
    </div>
  );
};

export default Node; 
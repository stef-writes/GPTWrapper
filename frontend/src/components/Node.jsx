import React, { useState, useEffect, useRef } from 'react';
import { Handle, Position } from 'reactflow';
import './Node.css';

const Node = ({ data, isConnectable }) => {
  const validInputNodes = data.validInputNodes || [];
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
  
  // Handle variable selection (now supports multiple and inserts at cursor)
  const handleVariableSelect = (event) => {
    const newlySelectedIds = Array.from(event.target.selectedOptions, option => option.value);
    const previouslySelectedIds = selectedVariable || [];
    setSelectedVariable(newlySelectedIds);

    const addedIds = newlySelectedIds.filter(id => !previouslySelectedIds.includes(id));

    if (addedIds.length > 0 && promptRef.current) {
        const validInputNodes = data.validInputNodes || [];
        const nodesToAdd = validInputNodes.filter(v => addedIds.includes(v.id));
        const textarea = promptRef.current;
        const start = textarea.selectionStart;
        const end = textarea.selectionEnd;
        const currentText = textarea.value; // Use direct value for insertion calculation

        let textToInsert = '';
        nodesToAdd.forEach(selectedVar => {
            const template = `{${selectedVar.name}}`;
            // Only prepare template if not already present (simple check, might need refinement)
            if (!currentText.includes(template)) {
                // Add a space before if inserting mid-text and no space exists
                const prefix = (start > 0 && currentText[start - 1] !== ' ') ? ' ' : '';
                // Add a space after if not at the end and no space exists
                const suffix = (end < currentText.length && currentText[end] !== ' ') ? ' ' : '';
                textToInsert += `${prefix}${template}${suffix}`;
            }
        });

        if (textToInsert) {
            // Construct the new prompt value by inserting/replacing at cursor
            const newPromptValue = 
                currentText.substring(0, start) + 
                textToInsert + 
                currentText.substring(end);

            // Update the state
            setPrompt(newPromptValue);

            // Set cursor position *after* the state update might render
            // Use a small timeout to ensure the DOM has updated (common React pattern)
            setTimeout(() => {
                const newCursorPos = start + textToInsert.length;
                textarea.focus(); // Ensure textarea has focus before setting selection
                textarea.setSelectionRange(newCursorPos, newCursorPos);
            }, 0);

            // Notify parent
            if (data.onPromptChange) {
                data.onPromptChange(newPromptValue);
            }
        }
    }

    // Always notify parent about the overall selection change
    if (data.onVariableSelect) {
      data.onVariableSelect(newlySelectedIds);
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
      {/* Target handle (Input) - Moved to Top */}
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
            Select a connected node from the dropdown below.
            Its template, like <code>{"{NodeName}"}</code>, will be automatically
            inserted into your prompt at the current cursor position.
          </div>
        )}
        
        <select 
          id="variable-select"
          multiple
          value={selectedVariable || []}
          onChange={handleVariableSelect}
          style={{ minHeight: '40px' }}
        >
          {validInputNodes.map((variable) => (
            <option key={variable.id} value={variable.id}>
              {variable.name}
            </option>
          ))}
        </select>
        
        {selectedVariable && selectedVariable.length > 0 && validInputNodes.length > 0 && (
          <div className="variable-usage-hint">
            Use {selectedVariable.map(varId => {
              const v = validInputNodes.find(v => v.id === varId);
              return v ? <code key={varId}>{`{${v.name}}`}</code> : null;
            })} in your prompt
          </div>
        )}
      </div>
      
      {/* Prompt textarea */}
      <textarea
        ref={promptRef}
        className="prompt-textarea"
        value={prompt}
        onChange={handlePromptChange}
        placeholder="Type your prompt here. Select connected nodes from the dropdown to include their output."
      />
      
      {/* Output textarea */}
      <textarea
        ref={outputRef}
        className="output-textarea"
        value={output}
        readOnly
        placeholder="AI Generated Output Here..."
      />
      
      {/* Source handle (Output) - Moved to Bottom */}
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
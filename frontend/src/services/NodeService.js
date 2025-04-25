// NodeService.js - Handles API calls to the backend

const API_URL = 'http://localhost:8000';

/**
 * Sends a prompt to the backend API to generate text
 * @param {string} prompt - The prompt text
 * @param {object} config - Optional LLM config parameters
 * @returns {Promise<object>} - The API response with generated text
 */
export const generateText = async (prompt, config = null) => {
  try {
    const response = await fetch(`${API_URL}/generate_text_node`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt_text: prompt,
        llm_config: config,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Error generating text');
    }

    return response.json();
  } catch (error) {
    console.error('Error calling generateText API:', error);
    throw error;
  }
};

/**
 * Adds a node to the backend script chain
 * @param {string} nodeId - Unique identifier for the node
 * @param {string} nodeType - Type of node (e.g., 'text_generation')
 * @param {Array<string>} inputKeys - Input keys required by the node
 * @param {Array<string>} outputKeys - Output keys produced by the node
 * @param {object} config - Optional LLM config parameters
 * @returns {Promise<object>} - The API response
 */
export const addNode = async (nodeId, nodeType, inputKeys = [], outputKeys = [], config = null) => {
  try {
    const response = await fetch(`${API_URL}/add_node`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        node_id: nodeId,
        node_type: nodeType,
        input_keys: inputKeys,
        output_keys: outputKeys,
        llm_config: config,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Error adding node');
    }

    return response.json();
  } catch (error) {
    console.error('Error calling addNode API:', error);
    throw error;
  }
};

/**
 * Adds an edge between two nodes in the backend script chain
 * @param {string} fromNode - Source node ID
 * @param {string} toNode - Target node ID
 * @returns {Promise<object>} - The API response
 */
export const addEdge = async (fromNode, toNode) => {
  try {
    const response = await fetch(`${API_URL}/add_edge`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from_node: fromNode,
        to_node: toNode,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Error adding edge');
    }

    return response.json();
  } catch (error) {
    console.error('Error calling addEdge API:', error);
    throw error;
  }
};

/**
 * Executes the script chain in the backend
 * @param {object} initialInputs - Optional initial inputs for the chain
 * @returns {Promise<object>} - The execution results
 */
export const executeChain = async (initialInputs = null) => {
  try {
    const response = await fetch(`${API_URL}/execute`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(initialInputs),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Error executing chain');
    }

    return response.json();
  } catch (error) {
    console.error('Error calling executeChain API:', error);
    throw error;
  }
};

// Export all functions as a service object
const NodeService = {
  generateText,
  addNode,
  addEdge,
  executeChain,
};

export default NodeService; 
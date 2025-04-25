import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './App.css'

function App() {
  const [inputValue, setInputValue] = useState('')
  const [chatHistory, setChatHistory] = useState([]) // Stores { role: 'user' | 'assistant', content: string }
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleInputChange = (event) => {
    setInputValue(event.target.value)
  }

  const handleSubmit = async (event) => {
    event.preventDefault() // Prevent default form submission (page reload)
    if (!inputValue.trim() || isLoading) return // Don't send empty messages or while loading

    const userMessage = { role: 'user', content: inputValue }

    // Create the history to send *before* adding the new user message to the local state
    const historyToSend = [...chatHistory]

    // Update local state immediately for responsiveness
    setChatHistory((prevHistory) => [...prevHistory, userMessage])
    setInputValue('') // Clear input field
    setIsLoading(true)
    setError(null)

    try {
      // --- Send message and history to backend ---
      const response = await fetch('http://localhost:8000/api/chat', { // Ensure this matches your backend URL/port
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        // Send the current message and the history
        body: JSON.stringify({ 
          message: userMessage.content, 
          history: historyToSend // Send the history *before* the latest user message was added
        }), 
      })

      if (!response.ok) {
        // Try to get error details from backend response body
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error occurred' }))
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      const assistantMessage = { role: 'assistant', content: data.response }
      setChatHistory((prevHistory) => [...prevHistory, assistantMessage])

    } catch (err) {
      console.error("Error fetching response:", err)
      setError(err.message || 'Failed to fetch response from the backend.')
      // Optionally add the error message to chat history
      // setChatHistory((prevHistory) => [...prevHistory, { role: 'assistant', content: `Error: ${err.message}` }]);
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="app-container">
      <h1>Simple GPT Wrapper</h1>
      <div className="chat-window">
        {chatHistory.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            <strong>{msg.role === 'user' ? 'You' : 'Assistant'}:</strong>
            {msg.role === 'assistant' ? (
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
            ) : (
              msg.content // Render user messages directly
            )}
          </div>
        ))}
        {isLoading && <div className="message assistant"><em>Assistant is thinking...</em></div>}
        {error && <div className="message error"><strong>Error:</strong> {error}</div>}
      </div>
      <form onSubmit={handleSubmit} className="input-area">
        <input
          type="text"
          value={inputValue}
          onChange={handleInputChange}
          placeholder="Type your message..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>
    </div>
  )
}

export default App

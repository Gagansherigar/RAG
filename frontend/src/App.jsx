import React, { useState, useRef, useEffect } from 'react';
import { Send, FileText, Loader2, Trash2, Settings, BookOpen } from 'lucide-react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [ragType, setRagType] = useState('advanced');
  const [topK, setTopK] = useState(5);
  const [showSettings, setShowSettings] = useState(false);
  const [apiUrl, setApiUrl] = useState('http://localhost:8000');
  const [systemStatus, setSystemStatus] = useState(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    checkHealth();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const checkHealth = async () => {
    try {
      const response = await fetch(`${apiUrl}/health`);
      const data = await response.json();
      setSystemStatus(data);
    } catch (error) {
      console.error('Health check failed:', error);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage = {
      role: 'user',
      content: input,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch(`${apiUrl}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: input,
          rag_type: ragType,
          top_k: topK,
          min_score: 0.2,
          return_context: false,
          summarize: ragType === 'pipeline'
        })
      });

      if (!response.ok) throw new Error('Query failed');

      const data = await response.json();

      const assistantMessage = {
        role: 'assistant',
        content: data.answer,
        sources: data.sources || [],
        confidence: data.confidence,
        summary: data.summary,
        ragType: ragType,
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage = {
        role: 'error',
        content: 'Failed to get response. Make sure the API is running at ' + apiUrl,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const clearHistory = async () => {
    try {
      await fetch(`${apiUrl}/history`, { method: 'DELETE' });
      setMessages([]);
    } catch (error) {
      console.error('Failed to clear history:', error);
    }
  };

  const exampleQueries = [
    'What is image processing?',
    'Explain Digital Image Representation ',
    'What is Image Topology',
    'explain computer vision'
  ];

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <div className="logo">
            <BookOpen size={24} />
          </div>
          <div className="header-title">
            <h1>RAG Assistant</h1>
            <p>Powered by AI & Vector Search</p>
          </div>
        </div>
        
        <div className="header-right">
          {systemStatus && (
            <div className="status-badge">
              <div className="status-dot"></div>
              <span>{systemStatus.documents} docs</span>
            </div>
          )}
          
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="icon-button"
            title="Settings"
          >
            <Settings size={20} />
          </button>
          
          <button
            onClick={clearHistory}
            className="icon-button"
            title="Clear chat"
          >
            <Trash2 size={20} />
          </button>
        </div>
      </header>

      {/* Settings Panel */}
      {showSettings && (
        <div className="settings-panel">
          <div className="settings-grid">
            <div className="settings-field">
              <label>RAG Type</label>
              <select
                value={ragType}
                onChange={(e) => setRagType(e.target.value)}
              >
                <option value="simple">Simple</option>
                <option value="advanced">Advanced</option>
                <option value="pipeline">Summarized</option>
              </select>
            </div>
            
            <div className="settings-field">
              <label>Top K Results</label>
              <input
                type="number"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value))}
                min="1"
                max="20"
              />
            </div>
            
           
            </div>
          </div>
       
      )}

      {/* Messages Area */}
      <div className="messages-container">
        <div className="messages-wrapper">
          {messages.length === 0 && (
            <div className="empty-state">
              <div className="empty-logo">
                <BookOpen size={40} />
              </div>
              <h2>Ask me anything</h2>
              <p>I'll search through your documents to find the answer</p>
              
              <div className="examples-grid">
                {exampleQueries.map((example, i) => (
                  <button
                    key={i}
                    onClick={() => setInput(example)}
                    className="example-button"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message, index) => (
            <div
              key={index}
              className={`message-row ${message.role}`}
            >
              {message.role === 'assistant' && (
                <div className="message-avatar assistant">
                  <BookOpen size={20} />
                </div>
              )}
              
              <div className={`message-content-wrapper ${message.role}`}>
                <div className={`message-bubble ${message.role}`}>
                  <div className="message-text">{message.content}</div>
                  
                  {message.sources && message.sources.length > 0 && (
                    <div className="sources-section">
                      <div className="sources-header">
                        <FileText size={16} />
                        <span className="sources-title">Sources</span>
                        {message.confidence && (
                          <span className="sources-confidence">
                            Confidence: {(message.confidence * 100).toFixed(0)}%
                          </span>
                        )}
                      </div>
                      <div className="sources-list">
                        {message.sources.slice(0, 3).map((source, i) => (
                          <div key={i} className="source-item">
                            <div className="source-header">
                              <span className="source-name">{source.source}</span>
                              <span className="source-score">
                                Score: {(source.score * 100).toFixed(0)}%
                              </span>
                            </div>
                            <div className="source-page">Page {source.page}</div>
                            <div className="source-preview">{source.preview}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {message.summary && (
                    <div className="summary-section">
                      <div className="summary-title">Summary</div>
                      <div className="summary-text">{message.summary}</div>
                    </div>
                  )}

                  <div className="message-timestamp">
                    {new Date(message.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              </div>

              {message.role === 'user' && (
                <div className="message-avatar user">
                  You
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="loading-message">
              <div className="message-avatar assistant">
                <BookOpen size={20} />
              </div>
              <div className="loading-bubble">
                <div className="loading-content">
                  <Loader2 size={16} className="loading-spinner" />
                  <span className="loading-text">Searching documents...</span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="input-container">
        <div className="input-wrapper">
          <div className="input-row">
            <div className="input-field-wrapper">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask a question about your documents..."
                rows="1"
                className="input-field"
              />
              <div className="input-indicator">
                {ragType === 'simple' ? '⚡ Simple' : ragType === 'advanced' ? '🎯 Advanced' : '🚀 Pipeline'}
              </div>
            </div>
            
            <button
              onClick={handleSend}
              disabled={!input.trim() || loading}
              className="send-button"
            >
              {loading ? (
                <Loader2 size={20} className="loading-spinner" />
              ) : (
                <Send size={20} />
              )}
              Send
            </button>
          </div>
          
          <div className="input-hint">
            Press Enter to send, Shift + Enter for new line
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
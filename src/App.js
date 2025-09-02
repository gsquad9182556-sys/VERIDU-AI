import React, { useState, useEffect } from 'react';
import './App.css';

const App = () => {
  const [activeTab, setActiveTab] = useState('analyze');
  const [contentType, setContentType] = useState('text');
  const [textContent, setTextContent] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState(null);

  const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

  useEffect(() => {
    if (activeTab === 'history') {
      fetch(`${backendUrl}/api/history`)
        .then(res => res.json())
        .then(data => setHistory(data))
        .catch(() => setHistory([]));
    }
    if (activeTab === 'stats') {
      fetch(`${backendUrl}/api/stats`)
        .then(res => res.json())
        .then(data => setStats(data))
        .catch(() => setStats(null));
    }
  }, [activeTab, backendUrl]);

  const handleAnalyze = async (e) => {
    e.preventDefault();
    setAnalyzing(true);
    setResult(null);

    const formData = new FormData();
    formData.append('content_type', contentType);
    if (contentType === 'text') {
      formData.append('text_content', textContent);
    } else if (selectedFile) {
      formData.append('file', selectedFile);
    }

    try {
      const res = await fetch(`${backendUrl}/api/analyze`, {
        method: 'POST',
        body: formData
      });
      const json = await res.json();
      setResult(json);
    } catch (error) {
      setResult({ error: 'Failed to analyze content.' });
    }
    setAnalyzing(false);
  };

  return (
    <div className="app">
      <h1>Truth Detector</h1>
      <div className="tabs">
        <button onClick={() => setActiveTab('analyze')} className={activeTab === 'analyze' ? 'active' : ''}>Analyze</button>
        <button onClick={() => setActiveTab('history')} className={activeTab === 'history' ? 'active' : ''}>History</button>
        <button onClick={() => setActiveTab('stats')} className={activeTab === 'stats' ? 'active' : ''}>Statistics</button>
      </div>

      {activeTab === 'analyze' && (
        <form className="analyze-form" onSubmit={handleAnalyze}>
          <label>Content Type:</label>
          <select value={contentType} onChange={e => setContentType(e.target.value)}>
            <option value="text">Text</option>
            <option value="image">Image</option>
            <option value="video">Video</option>
          </select>
          {contentType === 'text' && (
            <textarea value={textContent} onChange={e => setTextContent(e.target.value)} placeholder="Enter text for analysis..." />
          )}
          {(contentType === 'image' || contentType === 'video') && (
            <input type="file" accept={contentType === 'image' ? 'image/*' : 'video/*'}
              onChange={e => setSelectedFile(e.target.files)} />
          )}
          <button type="submit" disabled={analyzing}>Analyze</button>
          {analyzing && <p>Analyzing...</p>}
          {result && (
            <div className="result">
              {result.error ? (
                <p>{result.error}</p>
              ) : (
                <>
                  <h2>Analysis Result</h2>
                  <p><strong>Authenticity Score:</strong> {result.authenticity_score}</p>
                  <p><strong>Confidence Level:</strong> {result.confidence_level}</p>
                  <p><strong>Risk Factors:</strong> {result.risk_factors && result.risk_factors.join(', ')}</p>
                  <p><strong>Recommendations:</strong> {result.recommendations}</p>
                  <pre>{JSON.stringify(result.analysis_details, null, 2)}</pre>
                </>
              )}
            </div>
          )}
        </form>
      )}

      {activeTab === 'history' && (
        <div className="history">
          <h2>Recent Analysis History</h2>
          {history.length === 0 && <p>No history found.</p>}
          {history.length > 0 && (
            <table>
              <thead>
                <tr>
                  <th>Type</th>
                  <th>Score</th>
                  <th>Confidence</th>
                  <th>Date</th>
                  <th>Risk Factors</th>
                </tr>
              </thead>
              <tbody>
                {history.map(item => (
                  <tr key={item.id}>
                    <td>{item.content_type}</td>
                    <td>{item.authenticity_score}</td>
                    <td>{item.confidence_level}</td>
                    <td>{new Date(item.timestamp).toLocaleString()}</td>
                    <td>{item.risk_factors.join(', ')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {activeTab === 'stats' && (
        <div className="stats">
          <h2>Analysis Statistics</h2>
          {!stats ? (
            <p>No statistics available.</p>
          ) : (
            <ul>
              <li>Total Analyses: {stats.total_analyses}</li>
              <li>Text: {stats.content_types.text}</li>
              <li>Image: {stats.content_types.image}</li>
              <li>Video: {stats.content_types.video}</li>
              <li>Average Authenticity Score: {stats.averages.authenticity_score}</li>
              <li>Average Confidence Level: {stats.averages.confidence_level}</li>
            </ul>
          )}
        </div>
      )}
    </div>
  );
};

export default App;

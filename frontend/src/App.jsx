import { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(null);
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [asking, setAsking] = useState(false);

  const [contradictions, setContradictions] = useState([]);
  const [checkingContradictions, setCheckingContradictions] = useState(false);

  const analyzePaper = async () => {
    if (!file) return;

    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:8000/analyze", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({ error: "Could not connect to BioNexus backend." });
    }

    setLoading(false);
  };

  const askQuestion = async () => {
    if (!question.trim()) return;

    setAsking(true);
    setAnswer(null);
    setSources([]);

    try {
      const response = await fetch(
        `http://127.0.0.1:8000/ask?query=${encodeURIComponent(
          question
        )}&top_k=5`
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error("Failed to get answer");
      }

      setAnswer(data.answer);
      setSources(data.sources || []);
    } catch (error) {
      setAnswer("Could not get an answer from BioNexus.");
    }

    setAsking(false);
  };

  const checkContradictions = async () => {
    if (!question.trim()) return;

    setCheckingContradictions(true);
    setContradictions([]);

    try {
      const response = await fetch(
        `http://127.0.0.1:8000/contradictions?query=${encodeURIComponent(
          question
        )}&top_k=5`
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error("Failed to check contradictions");
      }

      setContradictions(data.comparisons || []);
    } catch (error) {
      setContradictions([]);
    }

    setCheckingContradictions(false);
  };

  return (
    <div className="app">
      <header>
        <h1>BioNexus</h1>
        <p>AI-Powered Medical Research Intelligence</p>
      </header>

      <main>
        <div className="upload-card">
          <h2>Research Paper Analysis</h2>

          <p>
            Upload a medical research PDF to analyze it using AI and Machine
            Learning.
          </p>

          <input
            type="file"
            accept=".pdf"
            onChange={(e) => setFile(e.target.files[0])}
          />

          {file && (
            <p className="file-name">
              Selected: {file.name}
            </p>
          )}

          <button onClick={analyzePaper} disabled={!file || loading}>
            {loading ? "Analyzing..." : "Analyze Paper"}
          </button>
        </div>

        {result && (
          <div className="result-card">
            <h2>Analysis Result</h2>

            {result.error ? (
              <p>{result.error}</p>
            ) : (
              <>
                <p>
                  <strong>Title:</strong> {result.title}
                </p>

                <p>
                  <strong>Pages:</strong> {result.num_pages}
                </p>

                <p>
                  <strong>Domain:</strong> {result.predicted_domain}
                </p>

                <p>
                  <strong>Chunks:</strong> {result.num_chunks}
                </p>
              </>
            )}
          </div>
        )}

        <div className="upload-card">
          <h2>Ask BioNexus</h2>

          <p>
            Ask a question about the uploaded research evidence.
          </p>

          <input
            type="text"
            placeholder="e.g. What are the main findings?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />

          <button
            onClick={askQuestion}
            disabled={!question.trim() || asking}
          >
            {asking ? "Thinking..." : "Ask Question"}
          </button>

          <button
            onClick={checkContradictions}
            disabled={!question.trim() || checkingContradictions}
          >
            {checkingContradictions
              ? "Checking..."
              : "Check Contradictions"}
          </button>
        </div>

        {answer && (
          <div className="result-card">
            <h2>AI Answer</h2>

            <p>{answer}</p>

            <h3>Research Sources</h3>

            {sources.map((source, index) => (
              <div key={index}>
                <p>
                  <strong>Source {index + 1}:</strong> {source.title}
                </p>

                <p>{source.text}</p>
              </div>
            ))}
          </div>
        )}

        {contradictions.length > 0 && (
          <div className="result-card">
            <h2>Evidence Comparison</h2>

            {contradictions.map((item, index) => (
              <div key={index}>
                <p>
                  <strong>Source 1:</strong> {item.source_1}
                </p>

                <p>
                  <strong>Source 2:</strong> {item.source_2}
                </p>

                <p>
                  <strong>Relationship:</strong>{" "}
                  {item.label.toUpperCase()}
                </p>

                <p>
                  <strong>Confidence:</strong>{" "}
                  {(item.score * 100).toFixed(1)}%
                </p>

                <hr />
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
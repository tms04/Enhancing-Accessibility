// App.js
import React, { useState } from "react";
import "./App.css";
import FileUpload from './components/FileUpload';
import PhraseInputs from './components/PhraseInputs';
import StatusMessage from './components/StatusMessage';

function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [phrase1, setPhrase1] = useState("");
  const [phrase2, setPhrase2] = useState("");
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [downloadUrl, setDownloadUrl] = useState(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    resetState();
    if (file && file.type.startsWith("video")) {
      setVideoFile(file);
    } else {
      setError("Please select a valid video file.");
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setDragging(true);
  };

  const handleDragLeave = () => {
    setDragging(false);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files[0];
    resetState();
    if (file && file.type.startsWith("video")) {
      setVideoFile(file);
    } else {
      setError("Please drop a valid video file.");
    }
  };

  const resetState = () => {
    setLoading(false);
    setError(null);
    setSuccess(false);
    setDownloadUrl(null);
  };

  const handleSubmit = async () => {
    resetState();

    if (!videoFile) {
      setError("Please provide a video file.");
      return;
    }
    if (!phrase1 || !phrase2) {
      setError("Please enter both phrases.");
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append('video', videoFile);
    formData.append('start_phrase', phrase1);
    formData.append('end_phrase', phrase2);

    try {
      const response = await fetch('http://127.0.0.1:5000/analyze_video', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        setDownloadUrl(url);
        setSuccess(true);
      } else {
        const errorData = await response.json();
        setError(errorData.error || 'An error occurred.');
      }
    } catch (error) {
      setError('Network error. Please check your connection.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1 className="title">Enhancing Accessibility</h1>
      <p className="subtitle">A Textual Transformation of Video Frames</p>

      <FileUpload
        videoFile={videoFile}
        onFileSelect={handleFileSelect}
        dragging={dragging}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      />

      <PhraseInputs
        phrase1={phrase1}
        phrase2={phrase2}
        onPhrase1Change={(e) => setPhrase1(e.target.value)}
        onPhrase2Change={(e) => setPhrase2(e.target.value)}
      />

      <StatusMessage
        error={error}
        loading={loading}
        success={success}
        downloadUrl={downloadUrl}
      />

      {!loading && (
        <button className="button" onClick={handleSubmit}>
          Analyze Video
        </button>
      )}
    </div>
  );
}

export default App;
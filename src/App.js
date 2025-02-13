// App.js
import React, { useState } from "react";
import { FaCloudUploadAlt, FaCheckCircle, FaExclamationTriangle } from 'react-icons/fa'; // Import icons
import "./App.css";

function App() {
    const [videoFile, setVideoFile] = useState(null);
    const [phrase1, setPhrase1] = useState("");
    const [phrase2, setPhrase2] = useState("");
    const [dragging, setDragging] = useState(false);
    const [loading, setLoading] = useState(false); // Loading state
    const [error, setError] = useState(null); // Error state
    const [success, setSuccess] = useState(false); //for animation
    const [downloadUrl, setDownloadUrl] = useState(null); //for download link

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
        setDownloadUrl(null)
    }

    const handleSubmit = async () => {  // Make the function async
        resetState();

        if (!videoFile) {
            setError("Please provide a video file.");
            return;
        }
        if (!phrase1 || !phrase2) {
            setError("Please enter both phrases.");
            return;
        }

        setLoading(true); // Start loading

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
                setSuccess(true)

            } else {
                // Handle errors from the backend
                const errorData = await response.json();
                setError(errorData.error || 'An error occurred.');
            }
        } catch (error) {
            setError('Network error. Please check your connection.');
        } finally {
            setLoading(false); // Stop loading, regardless of success/failure
        }
    };



  return (
    <div className="container">
      <h1 className="title">Video Lecture Analyzer</h1>
      <p className="subtitle">Extract key insights from your lectures.</p>

      <div
        className={`drag-drop-area ${dragging ? "dragging" : ""} ${
          videoFile ? "uploaded" : ""
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => document.getElementById("videoInput").click()}
      >
        {videoFile ? (
          <>
            <FaCheckCircle className="success-icon" />
            <span className="file-name">{videoFile.name}</span>
          </>
        ) : (
          <>
            <FaCloudUploadAlt className="upload-icon" />
            <span className="drag-drop-text">
              Drag & Drop or Click to Upload
            </span>
          </>
        )}
        <input
          type="file"
          id="videoInput"
          accept="video/*"
          style={{ display: "none" }}
          onChange={handleFileSelect}
        />
      </div>

      <div className="input-group">
        <input
          type="text"
          className="input-box"
          id="phrase1"
          value={phrase1}
          onChange={(e) => setPhrase1(e.target.value)}
          placeholder="Enter start phrase"
        />
        <input
          type="text"
          className="input-box"
          id="phrase2"
          value={phrase2}
          onChange={(e) => setPhrase2(e.target.value)}
          placeholder="Enter end phrase"
        />
      </div>

      {error && (
        <div className="error-message">
          <FaExclamationTriangle /> {error}
        </div>
      )}
      {loading ? (
        <div className="loader">Loading...</div> // Display loader
      ) : (
          <button className="button" onClick={handleSubmit}>
              Analyze Video
          </button>
      )}

        {downloadUrl && (
            <div className={`download-section ${success ? "success-animation" : ""}`}>
                <a href={downloadUrl} download="lecture_analysis.docx" className="download-button">
                   Download Analysis
                </a>
            </div>
        )}
    </div>
  );
}

export default App;
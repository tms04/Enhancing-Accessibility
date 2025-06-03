import React from 'react';
import { FaCloudUploadAlt, FaCheckCircle } from 'react-icons/fa';

const FileUpload = ({ videoFile, onFileSelect, dragging, onDragOver, onDragLeave, onDrop }) => {
    return (
        <div
            className={`drag-drop-area ${dragging ? "dragging" : ""} ${videoFile ? "uploaded" : ""
                }`}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
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
                onChange={onFileSelect}
            />
        </div>
    );
};

export default FileUpload; 
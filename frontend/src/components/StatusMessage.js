import React from 'react';
import { FaExclamationTriangle } from 'react-icons/fa';

const StatusMessage = ({ error, loading, success, downloadUrl }) => {
    return (
        <>
            {error && (
                <div className="error-message">
                    <FaExclamationTriangle /> {error}
                </div>
            )}
            {loading && <div className="loader"></div>}
            {downloadUrl && (
                <div className={`download-section ${success ? "success-animation" : ""}`}>
                    <a href={downloadUrl} download="lecture_analysis.docx" className="download-button">
                        Download Analysis
                    </a>
                </div>
            )}
        </>
    );
};

export default StatusMessage; 
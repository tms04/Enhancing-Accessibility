# 🎥 Enhancing Accessibility: A Textual Transformation of Video Frames

A full-stack web application that allows users to upload a **video lecture**, specify a **start** and **end phrase**, and receive a **detailed DOCX report** summarizing the extracted textual and visual content between those phrases using advanced OCR and Gemini AI.

---

## 🚀 Features

- 📤 Drag-and-drop video file uploader (with progress and error handling)
- ✍️ Smart OCR for both typed and handwritten content
- 🔍 Phrase-based video frame filtering
- 🧠 AI-powered explanation using Google Gemini
- 🖼️ Image extraction, classification (math/diagram/text), and visual description
- 📄 Generates a downloadable `.docx` report of findings

---

## 🛠️ Technologies Used

### Frontend
- **React.js** with Hooks
- **FontAwesome Icons**
- **CSS** for styling and animations

### Backend
- **Flask** (Python)
- **OpenCV** for video & image processing
- **Tesseract OCR** for text recognition
- **Google Gemini API** for AI summarization
- **python-docx** for Word report generation
- **ThreadPoolExecutor** for frame analysis

---

## 🧠 How It Works

1. **Upload a Lecture Video** via drag-and-drop or file input.
2. Enter a **start phrase** and an **end phrase**.
3. The backend:
   - Converts video to frames at intervals.
   - Extracts text using OCR (typed + handwritten).
   - Detects and saves key visual content (images, diagrams).
   - Uses Gemini AI to summarize extracted content.
4. A **Word document** is created containing:
   - Timestamped OCR text
   - Extracted images with descriptions
   - AI-generated explanation
5. A **Download Analysis** button appears once the process completes.

---

## 📦 Setup Instructions

### ✅ Prerequisites

- Python 3.8+
- Node.js 16+
- Tesseract OCR installed (and its path known)
- Google Gemini API key

### ⚙️ Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Run Flask server
python backend.py
```
### 💻 Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start React development server
npm start
```
### 📌 Future Enhancements
 1. Email delivery of generated report

 2. Support for regional languages in OCR

 3. Upload multiple videos in batch mode

 4. Add login and history tracking
### 👨‍💻 Author
Tanay Shah

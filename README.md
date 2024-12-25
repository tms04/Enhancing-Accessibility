# Video Lecture Analyzer

## Overview
Video Lecture Analyzer is a Python-based tool that automatically processes educational video content, extracting both text and visual elements to create comprehensive lecture analysis reports. The tool uses OCR (Optical Character Recognition), image processing, and AI-powered analysis to generate detailed documentation of lecture content.

## Features
- Automated text extraction from video frames using OCR
- Intelligent image detection and extraction
- Language detection and content filtering
- Text similarity comparison to avoid redundant content
- Image similarity detection to prevent duplicate visuals
- AI-powered content analysis using Groq API
- Comprehensive Word document report generation
- Multi-threaded processing for improved performance

## Prerequisites
```bash
pip install -r requirements.txt
```

Required packages:
- opencv-python
- pytesseract
- langdetect
- python-docx
- numpy
- scikit-learn
- scipy
- groq
- Pillow

Additionally, you need to:
1. Install Tesseract OCR on your system
2. Set up a Groq API key
3. Have sufficient storage space for temporary image files

## Configuration
Create a `config.py` file with your settings:

```python
# Configuration
TESSERACT_PATH = "path/to/tesseract"  # Update with your Tesseract installation path
INTERVAL_MINUTES = 3  # Interval between frame captures
MAX_WORKERS = 4  # Number of concurrent processing threads
TEXT_SIMILARITY_THRESHOLD = 0.7
IMAGE_SIMILARITY_THRESHOLD = 0.95
OUTPUT_DIR = "output"  # Directory for output files
IMAGE_OUTPUT_DIR = "extracted_images"  # Directory for extracted images
```

## Usage

1. Import the necessary modules and set up your configuration:
```python
from video_analyzer import VideoAnalyzer

analyzer = VideoAnalyzer(
    video_path="path/to/your/video.mp4",
    start_phrase="Your Start Phrase",
    end_phrase="Your End Phrase"
)
```

2. Run the analysis:
```python
analyzer.process_video()
```

3. Find the generated report in the output directory:
- A Word document containing the complete analysis
- Extracted images in the images directory
- Text content with timestamps

## Customization
You can customize the analysis by:
- Modifying the frame capture interval
- Adjusting similarity thresholds for text and images
- Customizing the AI analysis prompts
- Modifying the output document format
- Adding custom image processing filters

## Output Format
The tool generates:
1. A comprehensive Word document containing:
   - Timestamped lecture content
   - Extracted images with descriptions
   - AI-powered analysis of the content
   - Technical appendix with processing details

2. Organized image files:
   - Automatically filtered for relevance
   - Named with timestamps
   - Stored in a dedicated directory

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Tesseract OCR for text extraction
- OpenCV for image processing
- Groq for AI-powered analysis

## Disclaimer
This tool is designed for educational purposes. Please ensure you have the necessary rights to process any video content.
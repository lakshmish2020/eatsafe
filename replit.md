# Food Package Label OCR & Ingredient Analyzer

## Overview

This is a Streamlit-based web application that uses computer vision and AI to analyze food package labels. The system extracts text from uploaded images using OCR (Optical Character Recognition) technology, then uses OpenAI's API to analyze and summarize the ingredients. The application helps users understand what's in their food by providing detailed ingredient analysis, allergen identification, and nutritional insights from package photos.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for rapid web app development
- **Layout**: Two-column design for image upload and results display
- **User Interface**: File uploader with image preview and validation
- **Session Management**: Streamlit's built-in session state management

### Backend Architecture
- **Modular Design**: Separated into specialized processors for different concerns
- **OCR Processing**: Dedicated `OCRProcessor` class using Tesseract for text extraction
- **AI Analysis**: `IngredientAnalyzer` class for intelligent ingredient interpretation
- **Image Processing**: Computer vision utilities for image preprocessing and enhancement
- **Caching**: Streamlit resource caching for processor initialization to improve performance

### Data Processing Pipeline
- **Image Preprocessing**: OpenCV-based enhancement including denoising, contrast adjustment, and binarization
- **Text Extraction**: Tesseract OCR with confidence thresholding and character filtering
- **AI Enhancement**: OpenAI integration for intelligent ingredient analysis and summarization
- **Validation**: Multi-stage validation for images and extracted text

### Error Handling
- **Graceful Degradation**: Comprehensive error handling with user-friendly messages
- **Input Validation**: Image format and quality validation before processing
- **API Resilience**: Fallback mechanisms for OCR and AI processing failures

## External Dependencies

### Core Technologies
- **Streamlit**: Web application framework
- **OpenCV (cv2)**: Computer vision and image processing
- **PIL (Pillow)**: Image handling and manipulation
- **NumPy**: Numerical operations and array processing

### OCR Technology
- **Tesseract (pytesseract)**: Open-source OCR engine for text extraction
- **Custom Configuration**: Optimized character recognition settings for food labels

### AI Services
- **OpenAI API**: GPT-based ingredient analysis and interpretation
- **Environment Variables**: Secure API key management

### Image Processing
- **Multiple Format Support**: JPEG, PNG, BMP, TIFF compatibility
- **Base64 Encoding**: Image data conversion for processing
- **IO Operations**: Stream-based image handling

### Development Tools
- **Logging**: Python logging for debugging and monitoring
- **Regular Expressions**: Text pattern matching and extraction
- **JSON**: Data serialization for AI responses
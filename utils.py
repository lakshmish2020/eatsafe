import cv2
import numpy as np
from PIL import Image
import io

def preprocess_image(image, enhance_contrast=True, denoise=True, resize_factor=1.5):
    """
    Preprocess image for better OCR results
    
    Args:
        image: OpenCV image array
        enhance_contrast: Whether to enhance contrast
        denoise: Whether to remove noise
        resize_factor: Factor to resize image
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    try:
        # Make a copy to avoid modifying original
        processed = image.copy()
        
        # Convert to grayscale if colored
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        # Resize image for better OCR
        if resize_factor != 1.0:
            height, width = processed.shape[:2]
            new_width = int(width * resize_factor)
            new_height = int(height * resize_factor)
            processed = cv2.resize(processed, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Denoise
        if denoise:
            processed = cv2.fastNlMeansDenoising(processed)
        
        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(processed)
        
        # Apply slight Gaussian blur to smooth out artifacts
        processed = cv2.GaussianBlur(processed, (1, 1), 0)
        
        # Threshold to create binary image (helps with text extraction)
        _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return processed
        
    except Exception as e:
        raise Exception(f"Image preprocessing failed: {str(e)}")

def enhance_text_regions(image):
    """
    Specifically enhance text regions in the image
    
    Args:
        image: OpenCV image array
        
    Returns:
        numpy.ndarray: Enhanced image
    """
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply morphological operations to enhance text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # Close operation to connect text components
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Open operation to remove noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        
        return opened
        
    except Exception as e:
        return image

def is_valid_image(image):
    """
    Check if uploaded image is valid
    
    Args:
        image: PIL Image object
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        if image is None:
            return False
            
        # Check image format
        if image.format not in ['JPEG', 'PNG', 'BMP', 'TIFF']:
            return False
        
        # Check image size (minimum requirements)
        width, height = image.size
        if width < 100 or height < 100:
            return False
        
        # Check if image can be processed
        image.verify()
        
        return True
        
    except Exception:
        return False

def convert_pil_to_cv2(pil_image):
    """
    Convert PIL Image to OpenCV format
    
    Args:
        pil_image: PIL Image object
        
    Returns:
        numpy.ndarray: OpenCV image array
    """
    try:
        # Convert PIL to numpy array
        image_array = np.array(pil_image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_array.shape) == 3:
            cv2_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            cv2_image = image_array
            
        return cv2_image
        
    except Exception as e:
        raise Exception(f"Image conversion failed: {str(e)}")

def convert_cv2_to_pil(cv2_image):
    """
    Convert OpenCV image to PIL format
    
    Args:
        cv2_image: OpenCV image array
        
    Returns:
        PIL.Image: PIL Image object
    """
    try:
        # Convert BGR to RGB
        if len(cv2_image.shape) == 3:
            rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv2_image
            
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        return pil_image
        
    except Exception as e:
        raise Exception(f"Image conversion failed: {str(e)}")

def calculate_image_quality_score(image):
    """
    Calculate a quality score for the image to help with OCR success
    
    Args:
        image: OpenCV image array
        
    Returns:
        float: Quality score between 0 and 1
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate various quality metrics
        
        # 1. Contrast (standard deviation of pixel values)
        contrast = np.std(gray) / 255.0
        
        # 2. Sharpness (variance of Laplacian)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(laplacian_var / 1000.0, 1.0)  # Normalize
        
        # 3. Brightness distribution
        mean_brightness = np.mean(gray) / 255.0
        brightness_score = 1.0 - abs(mean_brightness - 0.5) * 2  # Penalize very dark/bright images
        
        # 4. Text-like regions (areas with high gradient)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        text_score = min(np.mean(gradient_magnitude) / 100.0, 1.0)
        
        # Combine scores (weighted average)
        quality_score = (contrast * 0.3 + sharpness * 0.3 + brightness_score * 0.2 + text_score * 0.2)
        
        return min(max(quality_score, 0.0), 1.0)  # Clamp between 0 and 1
        
    except Exception:
        return 0.5  # Default moderate score if calculation fails

def get_image_info(image):
    """
    Get information about the uploaded image
    
    Args:
        image: PIL Image object
        
    Returns:
        dict: Image information
    """
    try:
        info = {
            'format': image.format,
            'mode': image.mode,
            'size': image.size,
            'width': image.size[0],
            'height': image.size[1]
        }
        
        # Calculate file size estimate
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format or 'JPEG')
        info['estimated_size_mb'] = len(img_byte_arr.getvalue()) / (1024 * 1024)
        
        return info
        
    except Exception as e:
        return {'error': str(e)}

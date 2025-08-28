import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
from ocr_processor import OCRProcessor
from ingredient_analyzer import IngredientAnalyzer
from utils import preprocess_image, is_valid_image

# Initialize processors
@st.cache_resource
def get_processors():
    """Initialize OCR and AI processors with caching"""
    ocr = OCRProcessor()
    analyzer = IngredientAnalyzer()
    return ocr, analyzer

def add_pwa_components():
    """Add PWA components to the app"""
    # PWA manifest and service worker
    st.markdown("""
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
        <meta name="theme-color" content="#ff6b6b">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="default">
        <meta name="apple-mobile-web-app-title" content="EatSafe">
        <link rel="manifest" href="/static/manifest.json">
        <link rel="apple-touch-icon" href="/static/icon-192.png">
        <link rel="shortcut icon" href="/static/icon-192.png">
    </head>
    
    <script>
        if ('serviceWorker' in navigator) {
            window.addEventListener('load', function() {
                navigator.serviceWorker.register('/static/sw.js')
                    .then(function(registration) {
                        console.log('SW registered: ', registration);
                    }, function(registrationError) {
                        console.log('SW registration failed: ', registrationError);
                    });
            });
        }
    </script>
    
    <style>
        /* Mobile-optimized styles */
        .main > div {
            padding-top: 1rem;
        }
        
        .stButton > button {
            width: 100%;
            height: 3rem;
            font-size: 1.1rem;
        }
        
        .stFileUploader > div > div {
            text-align: center;
        }
        
        @media (max-width: 768px) {
            .main .block-container {
                padding: 1rem;
                max-width: 100%;
            }
            
            .stColumns > div {
                width: 100% !important;
                flex: none !important;
            }
            
            .stTabs [data-baseweb="tab-list"] {
                gap: 0.5rem;
            }
        }
        
        /* Install prompt */
        .install-prompt {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: #ff6b6b;
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 1000;
            display: none;
        }
        
        .install-prompt.show {
            display: block;
            animation: slideUp 0.3s ease;
        }
        
        @keyframes slideUp {
            from { transform: translateX(-50%) translateY(100px); }
            to { transform: translateX(-50%) translateY(0); }
        }
    </style>
    
    <div id="installPrompt" class="install-prompt">
        <span>📱 Install this app on your phone!</span>
        <button onclick="installApp()" style="margin-left: 10px; background: white; color: #ff6b6b; border: none; padding: 5px 10px; border-radius: 15px; cursor: pointer;">Install</button>
        <button onclick="hideInstallPrompt()" style="margin-left: 5px; background: transparent; color: white; border: 1px solid white; padding: 5px 10px; border-radius: 15px; cursor: pointer;">×</button>
    </div>
    
    <script>
        let deferredPrompt;
        
        window.addEventListener('beforeinstallprompt', (e) => {
            e.preventDefault();
            deferredPrompt = e;
            showInstallPrompt();
        });
        
        function showInstallPrompt() {
            const prompt = document.getElementById('installPrompt');
            if (prompt) {
                prompt.classList.add('show');
            }
        }
        
        function hideInstallPrompt() {
            const prompt = document.getElementById('installPrompt');
            if (prompt) {
                prompt.classList.remove('show');
            }
        }
        
        function installApp() {
            if (deferredPrompt) {
                deferredPrompt.prompt();
                deferredPrompt.userChoice.then((choiceResult) => {
                    if (choiceResult.outcome === 'accepted') {
                        console.log('User accepted the install prompt');
                    }
                    deferredPrompt = null;
                    hideInstallPrompt();
                });
            }
        }
    </script>
    """, unsafe_allow_html=True)

def main():
    # Configure page for PWA
    st.set_page_config(
        page_title="EatSafe",
        page_icon="🍎",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Add PWA components
    add_pwa_components()
    
    st.title("🍎 EatSafe")
    st.markdown("**AI-powered food safety and ingredient analysis**")
    st.markdown("Upload a clear photo of a food package's ingredient list for instant analysis.")
    
    # Initialize processors
    try:
        ocr_processor, ingredient_analyzer = get_processors()
    except Exception as e:
        st.error(f"Failed to initialize processors: {str(e)}")
        st.stop()
    
    # Mobile-responsive layout
    if st.session_state.get('mobile_view', False) or 'mobile' in st.query_params:
        # Single column for mobile
        col1 = st.container()
        col2 = st.container()
    else:
        # Two columns for desktop
        col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "📸 Choose or take a photo",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Take a photo or upload an image of the ingredients list"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Your Photo", use_column_width=True)
            
            # Image validation
            if not is_valid_image(image):
                st.error("Invalid image format or corrupted file. Please upload a valid image.")
                st.stop()
    
    with col2:
        st.subheader("🔍 Processing Options")
        
        # Image preprocessing options
        enhance_contrast = st.checkbox("Enhance Contrast", value=True, help="Improve text clarity")
        denoise = st.checkbox("Remove Noise", value=True, help="Clean up image artifacts")
        resize_factor = st.slider("Resize Factor", 0.5, 3.0, 1.5, 0.1, help="Scale image for better OCR")
        
        # OCR confidence threshold
        confidence_threshold = st.slider("OCR Confidence Threshold", 0, 100, 30, 5, 
                                       help="Minimum confidence for text extraction")
    
    # Process button
    if uploaded_file is not None:
        if st.button("🔬 Analyze Package Label", type="primary"):
            with st.spinner("Processing image and extracting ingredients..."):
                try:
                    # Convert PIL image to OpenCV format
                    image = Image.open(uploaded_file)
                    image_array = np.array(image)
                    if len(image_array.shape) == 3:
                        image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    else:
                        image_cv = image_array
                    
                    # Preprocess image
                    processed_image = preprocess_image(
                        image_cv, 
                        enhance_contrast=enhance_contrast,
                        denoise=denoise,
                        resize_factor=resize_factor
                    )
                    
                    # Extract text using OCR
                    extracted_text = ocr_processor.extract_text(processed_image, confidence_threshold)
                    
                    if not extracted_text or len(extracted_text.strip()) < 10:
                        st.warning("⚠️ Unable to extract sufficient text from the image. Try:")
                        st.write("• Upload a clearer image")
                        st.write("• Ensure the ingredients list is visible")
                        st.write("• Adjust preprocessing options")
                        st.stop()
                    
                    # Display extracted text
                    st.subheader("📝 Extracted Text")
                    with st.expander("View Raw OCR Output", expanded=False):
                        st.text_area("Raw Text", extracted_text, height=200, disabled=True)
                    
                    # Extract and analyze ingredients
                    ingredients_data = ingredient_analyzer.analyze_ingredients(extracted_text)
                    
                    if not ingredients_data:
                        st.warning("⚠️ No ingredients found in the extracted text. The image may not contain a clear ingredients list.")
                        st.stop()
                    
                    # Display results
                    st.subheader("🧪 Ingredient Analysis")
                    
                    # Create tabs for different views
                    tab1, tab2, tab3 = st.tabs(["📋 Ingredients List", "🍃 Nutritional Insights", "⚠️ Allergen Information"])
                    
                    with tab1:
                        if ingredients_data.get('ingredients'):
                            st.write("**Identified Ingredients:**")
                            for i, ingredient in enumerate(ingredients_data['ingredients'], 1):
                                st.write(f"{i}. **{ingredient['name']}**")
                                if ingredient.get('description'):
                                    st.write(f"   _{ingredient['description']}_")
                        else:
                            st.info("No individual ingredients identified.")
                    
                    with tab2:
                        if ingredients_data.get('nutritional_insights'):
                            insights = ingredients_data['nutritional_insights']
                            
                            # Health score
                            if insights.get('health_score'):
                                score = insights['health_score']
                                st.metric("Health Score", f"{score}/10")
                                
                            # Categories
                            if insights.get('categories'):
                                st.write("**Nutritional Categories:**")
                                for category in insights['categories']:
                                    st.write(f"• {category}")
                            
                            # Key nutrients
                            if insights.get('key_nutrients'):
                                st.write("**Key Nutrients:**")
                                for nutrient in insights['key_nutrients']:
                                    st.write(f"• {nutrient}")
                            
                            # Health notes
                            if insights.get('health_notes'):
                                st.write("**Health Notes:**")
                                st.write(insights['health_notes'])
                        else:
                            st.info("No nutritional insights available.")
                    
                    with tab3:
                        if ingredients_data.get('allergens'):
                            st.write("**Potential Allergens:**")
                            for allergen in ingredients_data['allergens']:
                                st.warning(f"⚠️ {allergen}")
                        else:
                            st.success("✅ No common allergens identified.")
                        
                        if ingredients_data.get('dietary_flags'):
                            st.write("**Dietary Information:**")
                            for flag in ingredients_data['dietary_flags']:
                                st.info(f"ℹ️ {flag}")
                    
                    # Summary
                    if ingredients_data.get('summary'):
                        st.subheader("📄 AI Summary")
                        st.write(ingredients_data['summary'])
                
                except Exception as e:
                    st.error(f"❌ Error processing image: {str(e)}")
                    st.write("Please try:")
                    st.write("• Uploading a different image")
                    st.write("• Adjusting the preprocessing options")
                    st.write("• Ensuring the image contains visible text")

    # Help section
    with st.expander("💡 Tips for Best Results", expanded=False):
        st.write("""
        **For optimal OCR results:**
        • Use well-lit, clear photos
        • Ensure the ingredients list is clearly visible
        • Avoid blurry or angled images
        • Make sure text is not obscured by shadows or reflections
        
        **Image requirements:**
        • Supported formats: JPG, PNG, BMP, TIFF
        • Maximum file size: 200MB
        • Minimum resolution: 300x300 pixels recommended
        
        **📱 Mobile tip:**
        • You can install this as an app on your phone!
        • Look for the "Add to Home Screen" or "Install" option in your browser
        """)
    
    # Mobile-friendly bottom spacing
    st.markdown("<div style='margin-bottom: 100px;'></div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("📱 **Install this app on your phone for easy access!**")
    st.markdown("*Powered by AI and OCR technology*")

if __name__ == "__main__":
    main()

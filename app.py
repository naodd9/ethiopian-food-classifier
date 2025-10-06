# app.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import gdown

# === CONFIG - GOOGLE DRIVE ===
FILE_ID = "1z1D_CX9HBT5HVcQvf0bvaETXtr5jd4pF"  # ← REPLACE WITH YOUR FILE ID
MODEL_URL = f"https://drive.google.com/uc?id=1z1D_CX9HBT5HVcQvf0bvaETXtr5jd4pF&export=download"
MODEL_PATH = "ethiopian_food_classifier_11_classes.pt"

@st.cache_resource
def download_and_load_model():
    """Download and load the TorchScript model"""
    # Download model if not exists
    if not os.path.exists(MODEL_PATH):
        try:
            with st.spinner('📥 Downloading model from Google Drive (first time only, this may take a minute)...'):
                gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
                
            # Verify download was successful
            if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1000000:  # >1MB
                st.success("✅ Model downloaded successfully!")
            else:
                st.error("❌ Download failed - file is too small or missing")
                return None
                
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            return None
    
    # Load as TorchScript model
    try:
        model = torch.jit.load(MODEL_PATH, map_location='cpu')
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Ethiopian food class names
CLASS_NAMES = [
    'Beyaynetu', 'Chechebsa', 'Doro Wat', 'Firfir', 'Genfo',
    'Kikil', 'Kitfo', 'Shekla Tibs', 'Shiro Wat', 'Tihlo', 'Tire Siga'
]

# Image preprocessing (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main():
    # Page configuration
    st.set_page_config(
        page_title="Ethiopian Food Classifier",
        page_icon="🌮",
        layout="wide"
    )
    
    # Header
    st.title("🌮 Ethiopian Food Classifier")
    st.markdown("**Upload a photo of Ethiopian food and I'll identify it using AI!**")
    
    # Load the model
    model = download_and_load_model()
    
    if model is None:
        st.error("""
        **Unable to load the model. Please check:**
        - Your Google Drive file is shared publicly
        - The file ID is correct: `12TMelvEjhi2fb9T64EayBQ`
        - The file is a valid TorchScript model (.pt file)
        """)
        return
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Upload Food Photo")
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=["jpg", "jpeg", "png"],
            help="Select a clear photo of Ethiopian food for best accuracy"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Your Food Photo", use_column_width=True)
            
            # Show image info
            st.info(f"Image size: {image.size[0]}x{image.size[1]} pixels")
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        if uploaded_file is not None:
            if st.button('🧠 Classify Food', type='primary', use_container_width=True):
                with st.spinner('🔍 Analyzing your food...'):
                    try:
                        # Preprocess the image
                        image_tensor = transform(image).unsqueeze(0)
                        
                        # Make prediction
                        with torch.no_grad():
                            outputs = model(image_tensor)
                            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                        
                        # Get top 3 predictions
                        top3_probs, top3_indices = torch.topk(probabilities, 3)
                        
                        # Display results
                        st.success("## 🍽️ Prediction Results")
                        
                        # Top prediction with progress bar
                        top_confidence = top3_probs[0].item() * 100
                        top_food = CLASS_NAMES[top3_indices[0].item()]
                        
                        st.markdown(f"### 🥇 {top_food}")
                        st.markdown(f"**Confidence: {top_confidence:.1f}%**")
                        st.progress(top_confidence / 100)
                        
                        # Show other possibilities
                        if len(top3_probs) > 1:
                            st.markdown("---")
                            st.subheader("Other Possibilities:")
                            
                            for i in range(1, min(3, len(top3_probs))):
                                confidence = top3_probs[i].item() * 100
                                food_name = CLASS_NAMES[top3_indices[i].item()]
                                
                                medal = "🥈" if i == 1 else "🥉"
                                st.markdown(f"**{medal} {food_name}** - {confidence:.1f}%")
                            
                    except Exception as e:
                        st.error(f"❌ Error during classification: {str(e)}")
                        st.info("Please try with a different image or check the image format.")
        
        else:
            # Show instructions when no image is uploaded
            st.info("👆 **Upload a food photo to get started!**")
            
            # Display supported foods
            st.markdown("---")
            st.subheader("🍴 Supported Ethiopian Foods")
            
            # Display foods in 2 columns
            col_a, col_b = st.columns(2)
            with col_a:
                for food in CLASS_NAMES[:6]:
                    st.write(f"• {food}")
            with col_b:
                for food in CLASS_NAMES[6:]:
                    st.write(f"• {food}")
            
            # Tips for best results
            st.markdown("---")
            st.subheader("💡 Tips for Best Results:")
            st.markdown("""
            - Use **clear, well-lit** photos
            - Focus on the **food** (not people or background)
            - Take photos from **directly above** or at a **slight angle**
            - Ensure the food is **clearly visible**
            - Avoid blurry or dark images
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center'>"
        "Built with ❤️ using PyTorch & Streamlit | "
        "AI Model trained to recognize 11 Ethiopian dishes"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

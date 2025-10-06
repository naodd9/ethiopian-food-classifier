# app.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import requests

# === CONFIG - GOOGLE DRIVE ===
# REPLACE THIS WITH YOUR ACTUAL GOOGLE DRIVE FILE ID
FILE_ID = "1z1D_CX9HBT5HVcQvf0bvaETXtr5jd4pF"  # ← REPLACE WITH YOUR FILE ID
MODEL_URL = f"https://drive.google.com/uc?id=1z1D_CX9HBT5HVcQvf0bvaETXtr5jd4pF&export=download"
MODEL_PATH = "ethiopian_food_classifier_11_classes.pt"

class EthiopianFoodClassifier(nn.Module):
    def __init__(self, num_classes=11):
        super(EthiopianFoodClassifier, self).__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

@st.cache_resource
def download_and_load_model():
    # Download model if not exists
    if not os.path.exists(MODEL_PATH):
        try:
            with st.spinner('📥 Downloading model from Google Drive (first time only, ~30 seconds)...'):
                session = requests.Session()
                response = session.get(MODEL_URL, stream=True)
                
                # Save the file
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=32768):
                        if chunk:
                            f.write(chunk)
                
                st.success("✅ Model downloaded successfully!")
                
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            return None
    
    # Load model with PyTorch 2.6 fix
    try:
        model = EthiopianFoodClassifier(num_classes=11)
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=False))
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Class names for Ethiopian food
class_names = ['Beyaynetu', 'Chechebsa', 'Doro Wat', 'Firfir', 'Genfo',
               'Kikil', 'Kitfo', 'Shekla Tibs', 'Shiro Wat', 'Tihlo', 'Tire Siga']

# Image preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main():
    st.set_page_config(
        page_title="Ethiopian Food Classifier",
        page_icon="🌮",
        layout="wide"
    )
    
    # Header
    st.title("🌮 Ethiopian Food Classifier")
    st.markdown("**Upload a photo of Ethiopian food and I'll tell you what it is!**")
    
    # Load model
    model = download_and_load_model()
    
    if model is None:
        st.error("""
        **Troubleshooting tips:**
        - Make sure your Google Drive file is shared publicly
        - Check that the FILE_ID is correct in the code
        - The file should be named: `ethiopian_food_classifier_11_classes.pt`
        """)
        return
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Upload Food Photo")
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=["jpg", "jpeg", "png"],
            help="Upload a clear photo of Ethiopian food"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Your Food Photo", use_column_width=True)
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        if uploaded_file is not None:
            if st.button('🧠 Classify Food', type='primary', use_container_width=True):
                with st.spinner('Analyzing your food...'):
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
                        st.success("**🍽️ Prediction Results**")
                        
                        # Top prediction
                        top_confidence = top3_probs[0].item() * 100
                        top_food = class_names[top3_indices[0].item()]
                        
                        st.markdown(f"### 🥇 {top_food}")
                        st.markdown(f"**Confidence: {top_confidence:.1f}%**")
                        st.progress(int(top_confidence))
                        
                        # Show other possibilities
                        if len(top3_probs) > 1:
                            st.markdown("**Other possibilities:**")
                            for i in range(1, min(3, len(top3_probs))):
                                confidence = top3_probs[i].item() * 100
                                food_name = class_names[top3_indices[i].item()]
                                st.markdown(f"**{i+1}.** {food_name} - {confidence:.1f}%")
                            
                    except Exception as e:
                        st.error(f"❌ Error during classification: {str(e)}")
        
        else:
            st.info("👆 **Upload a food photo to get started!**")
            
            # Show supported foods
            st.markdown("---")
            st.subheader("🍴 Supported Ethiopian Foods:")
            for food in class_names:
                st.write(f"• {food}")
            
            st.markdown("---")
            st.info("💡 **Tip:** Use clear, well-lit photos for best results!")

if __name__ == "__main__":
    main()

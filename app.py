# app.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gdown
import os

# === CONFIG - REPLACE WITH YOUR GOOGLE DRIVE FILE ID ===
MODEL_URL = "https://drive.google.com/file/d/1z1D_CX9HBT5HVcQvf0bvaETXtr5jd4pF/view?usp=drive_link"  # ← REPLACE THIS!
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
        with st.spinner('📥 Downloading model (first time only)...'):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    
    # Load model
    model = EthiopianFoodClassifier(num_classes=11)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

# Class names
class_names = ['Beyaynetu', 'Chechebsa', 'Doro Wat', 'Firfir', 'Genfo',
               'Kikil', 'Kitfo', 'Shekla Tibs', 'Shiro Wat', 'Tihlo', 'Tire Siga']

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit app
def main():
    st.set_page_config(page_title="Ethiopian Food Classifier", page_icon="🌮")
    
    st.title("🌮 Ethiopian Food Classifier")
    st.write("Upload a photo of Ethiopian food and I'll tell you what it is!")
    
    # Load model
    try:
        model = download_and_load_model()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Please make sure the Google Drive file ID is correct in the code.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('🧠 Classify Food', type='primary'):
            with st.spinner('Analyzing...'):
                # Preprocess and predict
                image_tensor = transform(image).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
                # Get top 3 predictions
                top3_probs, top3_indices = torch.topk(probabilities, 3)
                
                # Display results
                st.success("🎯 **Prediction Results:**")
                
                for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
                    confidence = prob.item() * 100
                    food_name = class_names[idx.item()]
                    
                    if i == 0:
                        st.markdown(f"**🥇 {food_name}** - {confidence:.1f}% confidence")
                        st.progress(int(confidence))
                    elif i == 1:
                        st.markdown(f"**🥈 {food_name}** - {confidence:.1f}%")
                    else:
                        st.markdown(f"**🥉 {food_name}** - {confidence:.1f}%")

if __name__ == "__main__":
    main()
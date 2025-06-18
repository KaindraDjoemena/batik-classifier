import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import timm
from torchvision import transforms
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Batik Classifier",
    layout="wide" # Use wide layout for side-by-side view
)

# --- Model Definition ---
class BatikClassifier(nn.Module):
  def __init__(self, num_classes, dropout_rate=0.5):
    super(BatikClassifier, self).__init__()
    self.base_model = timm.create_model("efficientnet_b0", pretrained=False)
    self.features = nn.Sequential(*list(self.base_model.children())[:-1])
    enet_out_size = 1280
    self.classifier = nn.Sequential(
        nn.Linear(enet_out_size, 512),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(512, num_classes)
    )
    for param in self.features.parameters():
      param.requires_grad = False

  def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)
    output = self.classifier(x)
    return output

# --- Class Names ---
CLASS_NAMES = [
    'Asmat', 'Bali', 'Betawi', 'Boraspati', 'Celup', 'Cendrawasih', 'Ceplok',
    'Ciamis', 'Dayak', 'Gajah', 'Garutan', 'Gentongan', 'Insang', 'Jakarta',
    'Kawung', 'Keraton', 'Lontara', 'Lumbung', 'Mataketeran', 'Megamendung',
    'Pala', 'Parang', 'Pring', 'Rumah_Minang', 'Sekar', 'Sidoluhur',
    'Sidomukti', 'Tifa', 'Yogyakarta_Parang'
]

# --- Model and Transform Loading ---
@st.cache_resource
def load_model_and_transforms():
    model = BatikClassifier(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load('batik_classifier.pth', map_location=torch.device('cpu')))
    model.eval()

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    return model, transform

model, transform = load_model_and_transforms()

def predict(image: Image.Image):
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
    return probabilities.cpu().numpy().flatten()

# --- Streamlit App Interface ---
st.title("Indonesian Batik Pattern Detector")
st.markdown(
    "Upload an image and the AI will identify the Batik motif. "
    "The results will be shown as a bar chart displaying the model's confidence."
)

uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Your Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("Analyzing the threads of tradition..."):
            all_probabilities = predict(image)

            result_df = pd.DataFrame({
                'Batik Type': CLASS_NAMES,
                'Probability': all_probabilities
            }).sort_values(by='Probability', ascending=False)

            result_df.reset_index(drop=True, inplace=True)

            # top prediction
            st.write("### Prediction Result")
            top_prediction_class = result_df.loc[0, 'Batik Type']
            top_prediction_confidence = result_df.loc[0, 'Probability']
            st.success(f"**Top Match:** {top_prediction_class.replace('_', ' ')}")
            st.info(f"**Confidence:** {top_prediction_confidence:.2%}")

            # bar chart
            st.write("### Confidence Distribution")

            chart_df = result_df.copy()
            chart_df['Batik Type'] = chart_df['Batik Type'].str.replace('_', ' ')
            
            highlight_color = "#2ca02c"
            default_color = "#639fff"

            chart_df['Color'] = np.where(
                chart_df.index == 0, 
                highlight_color,
                default_color
            )

            st.bar_chart(
                chart_df,
                x='Batik Type',
                y='Probability',
                color='Color',
                use_container_width=True
            )
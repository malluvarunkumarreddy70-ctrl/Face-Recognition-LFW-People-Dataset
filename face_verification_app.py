import streamlit as st
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import os
# Import your model here
from model import GhostFaceNet 

@st.cache_resource
def load_resources():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "ghostfacenet_best_weights_only.pth")
    yunet_path = os.path.join(base_path, "face_detection_yunet_2023mar.onnx")
    
    num_classes = 62  
    model = GhostFaceNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))    
    model.eval()
    
    detector = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320)) 
    return model, detector


def get_embedding(img_array, model, detector):
    h, w = img_array.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(img_array)
    
    if faces is None: return None
    
    # Crop & Resize
    x, y, w_f, h_f = faces[0][:4].astype(int)
    face = img_array[max(0,y):y+h_f, max(0,x):x+w_f]
    face = cv2.resize(face, (112, 112))
    
    # Transform
    t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    tensor = t(face).unsqueeze(0)
    
    with torch.no_grad():
        embedding, _ = model(tensor)
    return embedding

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="Face Auth Demo")
st.title("🛡️ Secure Face Authentication")

model, detector = load_resources()

# Step 1: Upload Reference Image
st.sidebar.header("Step 1: Setup ID")
ref_file = st.sidebar.file_uploader("Upload ID Photo", type=['jpg', 'png'])

if ref_file:
    ref_img = np.array(Image.open(ref_file).convert("RGB"))
    st.sidebar.image(ref_img, caption="Reference ID", width=150)
    
    # Step 2: Live Verification
    st.header("Step 2: Verify Identity")
    live_file = st.camera_input("Scan your face now")

    if live_file:
        live_img = np.array(Image.open(live_file).convert("RGB"))
        
        # Get Embeddings
        emb_ref = get_embedding(ref_img, model, detector)
        emb_live = get_embedding(live_img, model, detector)
        
        if emb_ref is not None and emb_live is not None:
            # Calculate Cosine Similarity
            dist = F.cosine_similarity(emb_ref, emb_live).item()
            
            if dist > 0.6: # Threshold - adjust based on your LFW results
                st.success(f"✅ Authenticated! (Confidence: {dist:.2f})")
                st.balloons()
            else:
                st.error(f"❌ Access Denied. User does not match (Score: {dist:.2f})")
        else:
            st.warning("Could not detect a clear face in one of the images.")
else:
    st.info("Please upload a reference photo in the sidebar to begin.")
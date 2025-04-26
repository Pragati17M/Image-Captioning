# import streamlit as st
# from PIL import Image
# import torch
# from model import EncoderCNN, DecoderRNN
# from vocabulary import Vocabulary
# from nlp_utils import clean_sentence
# import torchvision.transforms as transforms
# import pyttsx3
# import os
# import cv2
# import numpy as np
# import pickle


# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load Vocabulary
# with open("vocab.pkl", "rb") as f:
#     vocab = pickle.load(f)

# # Set Hyperparameters (same as used during training)
# embed_size = 256
# hidden_size = 512
# vocab_size = len(vocab)

# # Load Models
# encoder = EncoderCNN(embed_size)
# decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
# encoder.load_state_dict(torch.load(os.path.join("models", "encoder-1.pkl"), map_location=device))
# decoder.load_state_dict(torch.load(os.path.join("models", "decoder-1.pkl"), map_location=device))
# encoder.eval().to(device)
# decoder.eval().to(device)

# # Image Transform
# transform_test = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
# ])

# # Caption Generator
# def generate_caption(image):
#     image_tensor = transform_test(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         features = encoder(image_tensor).unsqueeze(1)
#         output = decoder.sample(features)
#     caption = clean_sentence(output, vocab.idx2word)
#     return caption

# # Text-to-Speech
# def speak_text(text):
#     engine = pyttsx3.init()
#     engine.say(text)
#     engine.runAndWait()

# # Streamlit UI
# st.set_page_config(page_title="Image Captioning for Blind", layout="centered")
# st.title("🖼️ AI Image Captioning App for the Visually Impaired")
# st.write("Upload or capture an image to hear what it contains.")

# # Upload or Camera
# option = st.radio("Choose image input method:", ("Upload Image", "Use Webcam"))

# if option == "Upload Image":
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
#     if uploaded_file:
#         image = Image.open(uploaded_file).convert("RGB")
#         st.image(image, caption="Uploaded Image", use_column_width=True)
#         if st.button("Generate Caption"):
#             caption = generate_caption(image)
#             st.success("Caption: " + caption)
#             speak_text(caption)

# elif option == "Use Webcam":
#     capture = st.camera_input("Take a picture")
#     if capture:
#         image = Image.open(capture).convert("RGB")
#         st.image(image, caption="Captured Image", use_column_width=True)
#         if st.button("Generate Caption"):
#             caption = generate_caption(image)
#             st.success("Caption: " + caption)
#             speak_text(caption)








import streamlit as st
from PIL import Image
import torch
from model import EncoderCNN, DecoderRNN
from vocabulary import Vocabulary
from nlp_utils import clean_sentence
import torchvision.transforms as transforms
import pyttsx3
import os
import pickle

# ========== Session State for Audio ==========
if "welcome_spoken" not in st.session_state:
    st.session_state.welcome_spoken = False
if "upload_spoken" not in st.session_state:
    st.session_state.upload_spoken = False
if "webcam_spoken" not in st.session_state:
    st.session_state.webcam_spoken = False
if "image_uploaded" not in st.session_state:
    st.session_state.image_uploaded = False
if "image_captured" not in st.session_state:
    st.session_state.image_captured = False

# ========== TTS Function ==========
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# ========== Device & Model Load ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

embed_size = 256
hidden_size = 512
vocab_size = len(vocab)

encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
encoder.load_state_dict(torch.load(os.path.join("models", "encoder-1.pkl"), map_location=device))
decoder.load_state_dict(torch.load(os.path.join("models", "decoder-1.pkl"), map_location=device))
encoder.eval().to(device)
decoder.eval().to(device)

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def generate_caption(image):
    image_tensor = transform_test(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = encoder(image_tensor).unsqueeze(1)
        output = decoder.sample(features)
    caption = clean_sentence(output, vocab.idx2word)
    return caption

# ========== Page Config & Styling ==========
st.set_page_config(page_title="Accessible AI Image Captioning", layout="centered")

# Speak only once
if not st.session_state.welcome_spoken:
    speak_text("Welcome to the AI image captioning app for the visually impaired. Choose an option to upload or capture an image.")
    st.session_state.welcome_spoken = True

st.title("🧠 Accessible AI Image Captioning")
st.write("Helping visually impaired users understand images using AI.")

# ========== Image Input Choice ==========
option = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))

# ========== Upload Image ==========
if option == "Upload Image":
    if not st.session_state.upload_spoken:
        speak_text("You selected upload image. Now choose an image file.")
        st.session_state.upload_spoken = True

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if not st.session_state.image_uploaded:
            speak_text("Image uploaded. You can now generate a caption.")
            st.session_state.image_uploaded = True

        if st.button("🔍 Generate Caption"):
            speak_text("Generating caption.")
            caption = generate_caption(image)
            st.success("🗣️ Caption: " + caption)
            speak_text(caption)

# ========== Webcam Capture ==========
elif option == "Use Webcam":
    if not st.session_state.webcam_spoken:
        speak_text("You selected camera input. Please take a picture.")
        st.session_state.webcam_spoken = True

    capture = st.camera_input("Capture a photo")
    
    if capture:
        image = Image.open(capture).convert("RGB")
        st.image(image, caption="Captured Image", use_column_width=True)

        if not st.session_state.image_captured:
            speak_text("Image captured. You can now generate a caption.")
            st.session_state.image_captured = True

        if st.button("🔍 Generate Caption"):
            speak_text("Generating caption.")
            caption = generate_caption(image)
            st.success("🗣️ Caption: " + caption)
            speak_text(caption)





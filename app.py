import streamlit as st
import time
import textwrap
import json
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from detector import analyze_image_from_bytes
import os
from dotenv import load_dotenv

# ─── Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Text Detector | Authenticity Scanner",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Refreshing Light Purple Custom CSS ────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background-color: #f8f5fc; /* Very light purple/lavender background */
        font-family: 'Inter', sans-serif;
        color: #334155;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    .hero-container {
        text-align: center;
        padding: 3rem 1rem 2rem;
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4c1d95; /* Deep purple */
        line-height: 1.2;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: #64748b;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
    }
    
    .clean-divider {
        height: 1px;
        background-color: #e2e8f0;
        margin: 2rem 0;
        border: none;
    }

    .white-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    }
    
    .card-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #475569;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .stTextArea textarea {
        background: #f8fafc !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        color: #334155 !important;
        font-size: 1rem !important;
        padding: 1rem !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2) !important;
    }
    
    .stButton > button {
        background-color: #8b5cf6 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 500 !important;
        width: 100% !important;
        transition: background-color 0.2s !important;
    }
    
    .stButton > button:hover {
        background-color: #7c3aed !important;
    }
    
    .result-card {
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    
    .result-ai {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
    }
    
    .result-human {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
    }
    
    .result-icon { font-size: 2rem; margin-bottom: 0.5rem; }
    .result-label { font-size: 1.3rem; font-weight: 600; margin-bottom: 0.2rem; }
    .result-label-ai { color: #b91c1c; }
    .result-label-human { color: #15803d; }
    .result-sublabel { font-size: 0.9rem; color: #64748b; margin-bottom: 1rem; }
    
    .reasoning-box {
        background: rgba(255, 255, 255, 0.6);
        border-left: 3px solid;
        padding: 1rem;
        border-radius: 4px;
        margin-top: 1rem;
        font-size: 0.95rem;
        color: #334155;
        line-height: 1.5;
    }
    
    .reasoning-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .confidence-container { margin-top: 1rem; }
    .confidence-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem; }
    .confidence-label { font-size: 0.85rem; font-weight: 500; color: #475569; }
    .confidence-value { font-size: 1.2rem; font-weight: 700; }
    .confidence-value-ai { color: #b91c1c; }
    .confidence-value-human { color: #15803d; }
    
    .confidence-bar-bg { height: 6px; background: #e2e8f0; border-radius: 10px; overflow: hidden; }
    .confidence-bar { height: 100%; border-radius: 10px; transition: width 0.5s ease-out; }
    .bar-ai { background-color: #ef4444; }
    .bar-human { background-color: #22c55e; }
    
    .analyzing-box { 
        text-align: center; 
        padding: 2rem; 
        background: #ffffff; 
        border: 1px solid #e2e8f0; 
        border-radius: 12px; 
        margin-top: 1rem; 
    }
    .analyzing-text { color: #64748b; font-weight: 500; margin-top: 0.5rem; }
    
    .warning-box { 
        background: #fffbeb; 
        border: 1px solid #fde68a; 
        border-radius: 8px; 
        padding: 1rem; 
        color: #b45309; 
        font-size: 0.95rem; 
        margin-top: 1rem; 
    }
</style>
""", unsafe_allow_html=True)


# ─── Gemini Setup ─────────────────────────────────────────────────────
# This loads from your local .env file when testing on your computer
load_dotenv()

# This reads the environment variable (both locally and on Render)
API_KEY = os.getenv("GCP_API_KEY")

class DetectionResult(BaseModel):
    label: str = Field(description="Must be 'Fake' if AI-generated, or 'Real' if human-written")
    score: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief explanation of why this classification was chosen")

class ImageDetectionResult(BaseModel):
    label: str = Field(description="Must be 'AI Generated' if the image is AI-generated, or 'Real' if it's a real photograph")
    score: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Detailed explanation of why this classification was chosen, including specific visual artifacts or patterns observed")

SYSTEM_INSTRUCTION = """
You are an expert AI writing detector. Analyze the user's text and determine if it is likely written by a human or generated by an AI.
Rely on established signs of AI writing, such as:
1. High density of "AI vocabulary": words like "delve", "tapestry", "multifaceted", "testament", "orchestrate", "realm", "crucial", "underscores", "moreover", "navigating the landscape".
2. Formulaic structures: opening with broad historical/general statements, or ending with outline-like summaries (e.g. "In conclusion").
3. Superficial analysis: perfectly balanced paragraphs that lack genuine insight, specific detail, or strong editorial opinions.
4. Overuse of transitional adverbs and parallelisms (e.g. "Not just X, but also Y").

Provide your output strictly matching the requested JSON schema.
"""

IMAGE_SYSTEM_INSTRUCTION = """
You are an expert AI image detector. Analyze the provided image and determine if it is likely AI-generated or a real photograph.
Look for common signs of AI-generated images such as:
1. Unnatural textures or patterns in skin, hair, or clothing
2. Inconsistent lighting or shadows
3. Distorted or impossible geometry (extra fingers, melted objects, etc.)
4. Overly perfect or symmetrical compositions
5. Unusual artifacts or blurring in certain areas
6. Inconsistent depth of field
7. Strange reflections or missing environmental details
8. Text or logos that appear distorted or nonsensical

Provide your output strictly matching the requested JSON schema with detailed reasoning about specific visual elements you observed.
"""

def analyze_text(text):
    client = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite", 
        contents=text,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=DetectionResult,
            temperature=0.1
        )
    )
    return json.loads(response.text)

# ─── Sidebar configuration ────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    st.markdown("---")
    st.markdown("""
    **Detection criteria:**
    - Vocabulary analysis
    - Structural patterns
    - Logical progression
    - Visual artifacts (for images)
    - Texture and lighting analysis (for images)
    """)

# ─── Hero Header ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Content Authenticity Scanner</div>
    <div class="hero-subtitle">
        A simple and clean tool to detect AI-generated text and images.
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Input & Processing ───────────────────────────────────────────────
# Add tabs for text and image detection
tab1, tab2 = st.tabs(["Text Detection", "Image Detection"])

# Text Detection Tab
with tab1:
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("""<div class="white-card"><div class="card-label">Input Text</div></div>""", unsafe_allow_html=True)
        user_input = st.text_area("Input", height=300, placeholder="Paste the text you'd like to check here...", label_visibility="collapsed", key="text_input")
        
        scan_clicked = st.button("Check Content", key="text_check")

    with col2:
        if scan_clicked:
            if not user_input.strip():
                st.markdown("""<div class="warning-box">⚠️ Please enter some text to check.</div>""", unsafe_allow_html=True)
            else:
                analyzing_placeholder = st.empty()
                analyzing_placeholder.markdown("""
                <div class="analyzing-box">
                    <div style="font-size: 1.5rem;">⏳</div>
                    <div class="analyzing-text">Analyzing text...</div>
                </div>
                """, unsafe_allow_html=True)
                
                try:
                    result = analyze_text(user_input)
                    analyzing_placeholder.empty()
                    
                    label = result["label"]
                    score = round(result["score"] * 100, 2)
                    reasoning = result["reasoning"]
                    
                    is_ai = label == "Fake"
                    
                    if is_ai:
                        st.success("🤖 AI-Generated Content Detected")
                        st.write(f"Confidence: {score}%")
                        st.write(f"Analysis: {reasoning}")
                    else:
                        st.success("✔ Human-Written Content")
                        st.write(f"Authenticity Score: {score}%")
                        st.write(f"Analysis: {reasoning}")
                        
                except Exception as e:
                    analyzing_placeholder.empty()
                    st.error(f"API Error: {str(e)}\n\nMake sure your API key is valid.")
        else:
            st.markdown("""
            <div style="height: 100%; display: flex; align-items: center; justify-content: center; color: #94a3b8; font-style: italic; background: #ffffff; border: 1px dashed #cbd5e1; border-radius: 12px; margin-top: 1rem;">
                Results will appear here
            </div>
            """, unsafe_allow_html=True)

# Image Detection Tab
with tab2:
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("""<div class="white-card"><div class="card-label">Upload Image</div></div>""", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            # Display the uploaded image
            st.image(uploaded_file, caption="Uploaded Image", width=400)
        
        image_scan_clicked = st.button("Check Image", key="image_check")

    with col2:
        if image_scan_clicked:
            if uploaded_file is None:
                st.markdown("""<div class="warning-box">⚠️ Please upload an image to check.</div>""", unsafe_allow_html=True)
            else:
                analyzing_placeholder = st.empty()
                analyzing_placeholder.markdown("""
                <div class="analyzing-box">
                    <div style="font-size: 1.5rem;">⏳</div>
                    <div class="analyzing-text">Analyzing image...</div>
                </div>
                """, unsafe_allow_html=True)
                
                try:
                    # Get image data and mime type
                    image_bytes = uploaded_file.getvalue()
                    mime_type = uploaded_file.type
                    
                    # Analyze the image
                    result = analyze_image_from_bytes(image_bytes, mime_type)
                    analyzing_placeholder.empty()
                    
                    label = result["label"]
                    score = round(result["score"] * 100, 2)
                    reasoning = result["reasoning"]
                    
                    is_ai = label == "AI Generated"
                    
                    if is_ai:
                        st.success("🤖 AI-Generated Image Detected")
                        st.write(f"Confidence: {score}%")
                        st.write(f"Analysis: {reasoning}")
                    else:
                        st.success("✔ Real Photograph")
                        st.write(f"Authenticity Score: {score}%")
                        st.write(f"Analysis: {reasoning}")
                        
                except Exception as e:
                    analyzing_placeholder.empty()
                    st.error(f"API Error: {str(e)}\n\nMake sure your API key is valid.")
        else:
            st.markdown("""
            <div style="height: 100%; display: flex; align-items: center; justify-content: center; color: #94a3b8; font-style: italic; background: #ffffff; border: 1px dashed #cbd5e1; border-radius: 12px; margin-top: 1rem;">
                Results will appear here
            </div>
            """, unsafe_allow_html=True)
import streamlit as st
import tensorflow as tf
from preprocessing import InferencePreprocessor
import nltk
import base64
import speech_recognition as sr  
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components
import numpy as np
import re

st.set_page_config(layout="wide", page_title="Fake news Detection", initial_sidebar_state="collapsed")

# --- BACKEND LOADERS ---
@st.cache_resource
def load_model_and_prep(model_path):
    # Load Model
    try:
        model = tf.keras.models.load_model(model_path)
        model_name = model_path.split('/')[-1]  # Extract just the file name
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

    # Load Preprocessor
    try:
        preprocessor = InferencePreprocessor()
    except Exception as e:
        st.error(f"Error loading preprocessor: {e}")
        return None, None, None
        
    return model, model_name, preprocessor

# Update path if necessary
model, model_name, preprocessor = load_model_and_prep('model/best_cnn_rnn_model.keras')

def transcribe_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        placeholder = st.empty()
        placeholder.info("🎤 Listening... Speak now!")
        r.adjust_for_ambient_noise(source)
        try:
            audio_data = r.listen(source, timeout=5)
            placeholder.success("Processing audio...")
            text = r.recognize_google(audio_data)
            placeholder.empty()
            return text
        except sr.WaitTimeoutError:
            placeholder.warning("No speech detected.")
            return None
        except sr.UnknownValueError:
            placeholder.error("Could not understand audio.")
            return None
        except sr.RequestError as e:
            placeholder.error(f"API Error: {e}")
            return None
        except Exception as e:
            placeholder.error(f"Microphone error: {e}")
            return None
        
def lime_predict_proba(text_list):
    """
    Wraps the Keras model to accept a list of raw texts 
    and return (N, 2) probabilities for [Real, Fake].
    """
    # 1. Preprocess every text in the list
    cleaned_texts = [preprocessor.clean_text(t) for t in text_list]
    
    # 2. Convert to tensor AND Reshape
    input_tensor = tf.convert_to_tensor(cleaned_texts, dtype=tf.string)
    input_tensor = tf.reshape(input_tensor, (-1, 1))
    
    # 3. Predict 
    probs = model.predict(input_tensor, verbose=0)
    
    # 4. Convert to (N, 2) format: [Prob_Real, Prob_Fake]
    results = []
    for p in probs:
        prob_fake = p[0]
        prob_real = 1 - prob_fake
        results.append([prob_real, prob_fake])
        
    return np.array(results)
        

# --- CSS STYLING ---
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return ""

def inject_custom_css():
    bin_str = get_base64_of_bin_file('fakenews5.jpg') 
    
    st.markdown(f"""
    <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center top;
        }}
        .block-container {{
            background-color: transparent !important;
            padding-top: 2rem;
            max-width: 1200px;
        }}
        h1 {{
            color: white !important;
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 800;
            text-shadow: 2px 2px 4px #000000;
            margin-bottom: 0px;
        }}
        div[data-testid="stMarkdownContainer"] .subtitle {{
            color: #ddd;
            font-size: 20px;
            margin-bottom: 30px;
            text-shadow: 1px 1px 2px #000000;
        }}

        
        /* The White Card (Input) */
        [data-testid="stForm"] {{
            background-color: rgba(179, 204, 204, 0.95);
            border-radius: 12px;
            padding: 30px;
            border: none;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}

        /* The Analysis Card (LIME Results) - NEW ADDITION */
        .analysis-card {{
            background-color: rgba(40, 44, 52, 0.95);
            border-radius: 12px;
            padding: 25px;
            margin-top: 20px;
            border: 1px solid #444;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}

        .custom-label {{
            background-color: #3f51b5;
            color: white;
            padding: 4px 10px;
            font-size: 14px;
            font-weight: bold;
            border-radius: 4px;
            margin-bottom: 8px;
            display: inline-block;
        }}
        .stTextInput input, .stTextArea textarea {{
            background-color: #2b2c34 !important;
            color: white !important;
            border-radius: 6px;
            border: 1px solid #444;
        }}
        .stTextInput label, .stTextArea label {{
            display: none;
        }}
        div.stFormSubmitButton > button {{
            appearance: none;
            background-color: #FFFFFF !important;
            border-radius: 40em;
            border-style: none;
            box-shadow: #ADCFFF 0 -12px 6px inset;
            color: #000000 !important;
            cursor: pointer;
            display: inline-block;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 1.2rem;
            font-weight: 700;
            padding: 1rem 1.3rem;
            text-align: center;
            transition: all 0.15s ease;
            width: 100%;
        }}
        div.stFormSubmitButton > button:hover {{
            background-color: #07bd06 !important;  
            box-shadow: #057804 0 -6px 8px inset;
            transform: scale(1.125);
        }}
        div.stFormSubmitButton > button:active {{
            transform: scale(1.025);
        }}
        /* Word Feature Bars */
        .feature-bar-container {{
            margin-bottom: 12px;
            display: flex;
            align-items: center;
        }}
        .feature-bar {{
            height: 28px;
            border-radius: 6px;
            display: flex;
            align-items: center;
            padding: 0 10px;
            color: white;
            font-size: 0.9rem;
            font-weight: 600;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .streamlit-expanderHeader {{
            background-color: #1a1a1a !important;
            color: white !important;
            border: 1px solid #333;
        }}
        .streamlit-expanderContent {{
            background-color: #1a1a1a !important;
            color: #ccc !important;
            border: 1px solid #333;
            border-top: none;
        }}

    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- MAIN APP LAYOUT ---

# 1. Header Area
col_h1, col_h2 = st.columns([0.05, 0.95])
with col_h2:
    st.markdown("<h1>📰  FAKE NEWS DETECTION SYSTEM </h1>", unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Utilizing advanced deep learning models for news integrity assessment.</p>', unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if 'speech_input' not in st.session_state:
    st.session_state['speech_input'] = ""

# 2. Input Section
with st.form("analysis_form"):
    st.markdown('<h1 style="color:#333; margin-top:0;">Analyze Article</h1>', unsafe_allow_html=True)
    
    st.markdown('<h1 style="color:#333; font-size:25px;">Article Title</h1>', unsafe_allow_html=True)
    title = st.text_input("Article Title", placeholder="Paste the headline here...") 
    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True) 

    col_label, col_btn = st.columns([0.85, 0.15])
    with col_label:
        st.markdown('<h1 style="color:#333; font-size:25px;">Article Body Content</h1>', unsafe_allow_html=True)
    with col_btn:
        record_click = st.form_submit_button("🎤 Record")

    text = st.text_area(
        "Article Body Content", 
        value=st.session_state['speech_input'], 
        placeholder="Paste content or click Record...", 
        height=200
    ) 
    
    words = text.split()
    word_count = len(words)
    min_words = 20 

    if word_count < min_words and text != "":
        st.markdown(f"""
            <div style="
                padding: 10px;
                border-radius: 10px;
                background-color: #ffcccc; 
                color: #EE4B2B;
                border: 1px ;
                font-weight: bold;
                margin-bottom: 15px;
            ">
                ⚠️ Please enter at least {min_words} words. Current: {word_count}
            </div>
            """, unsafe_allow_html=True)

    submitted = st.form_submit_button("ANALYZE ARTICLE INTEGRITY")


# 3. Logic Handling
if record_click:
    transcribed_text = transcribe_speech()
    if transcribed_text:
        current_text = st.session_state['speech_input']
        if current_text:
            st.session_state['speech_input'] = current_text + " " + transcribed_text
        else:
            st.session_state['speech_input'] = transcribed_text
        st.rerun() 

if submitted:
    if not title.strip() or not text.strip():
        st.warning("⚠️ Please enter both a title and article body.")
    elif word_count < min_words:
        pass
    else:
        combined_text_check = f"{title} {text}"
        if not preprocessor.is_english(combined_text_check):
            st.error("⚠️ Unsupported Language Detected. This system currently supports English articles only.")
            st.stop()
        # Use a single spinner for the whole process
        with st.spinner("Analyzing patterns..."):
            try:
                combined_text = f"{title} {text}"
                cleaned_text = preprocessor.clean_text(combined_text)

                input_tensor = tf.convert_to_tensor([[cleaned_text]], dtype=tf.string)
                prediction_prob = model.predict(input_tensor, verbose=0)[0][0]
                
                is_fake = prediction_prob >= 0.5
                
                # --- RESULTS VARIABLES ---
                if cleaned_text is None:
                    st.error("The input text could not be processed. Please ensure it contains valid English content.")
                    st.stop()

                elif is_fake:
                    verdict_text = "POTENTIAL MISINFORMATION"
                    verdict_subtext = "Integrity Analysis detects significant anomalies."
                    verdict_bg = "linear-gradient(90deg, #D32F2F 0%, #FFCDD2 100%)"
                    icon_bg = "#B71C1C"
                    verdict_icon = "🛑"
                    confidence = prediction_prob
                    conf_label = "Confidence Score (Fake)"
                    bar_color = "#D32F2F"
                else:
                    verdict_text = "VERIFIED AUTHENTICITY"
                    verdict_subtext = "Content aligns with standard reporting patterns."
                    verdict_bg = "linear-gradient(90deg, #1565C0 0%, #BBDEFB 100%)"
                    icon_bg = "#0D47A1"
                    verdict_icon = "🛡️"
                    confidence = 1 - prediction_prob
                    conf_label = "Confidence Score (Real)"
                    bar_color = "#1565C0"

                # --- DISPLAY MAIN VERDICT ---
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background: {verdict_bg}; padding: 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); color: white; display: flex; align-items: center; margin-bottom: 25px;">
                    <div style="background-color: {icon_bg}; width: 60px; height: 60px; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-size: 30px; margin-right: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
                        {verdict_icon}
                    </div>
                    <div>
                        <h2 style="margin: 0; color: white; font-weight: 800; font-size: 24px; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{verdict_text}</h2>
                        <p style="margin: 5px 0 0 0; color: rgba(255,255,255,0.9); font-size: 14px;">{verdict_subtext}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Metrics
                col_res1, col_res2 = st.columns([1, 4])
                with col_res1:
                    st.markdown(f"""
                    <div style="text-align: center; background: rgba(255,255,255,0.9); padding: 10px; border-radius: 10px;">
                        <div style="color: #555; font-size: 12px; font-weight: bold; text-transform: uppercase;">{conf_label}</div>
                        <div style="color: {bar_color}; font-size: 36px; font-weight: 900;">{confidence:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_res2:
                    st.write("") 
                    st.write("") 
                    st.markdown(f"""
                    <div style="background-color: rgba(255,255,255,0.5); border-radius: 20px; height: 25px; width: 100%; overflow: hidden; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);">
                        <div style="width: {confidence*100}%; background-color: {bar_color}; height: 100%; border-radius: 20px; transition: width 1s ease-in-out;"></div>
                    </div>
                    """, unsafe_allow_html=True)

                # Diagnostics
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("System Diagnostics"):
                    st.markdown(f"""
                    <ul style="color: #ccc;">
                        <li><b>Raw Probability:</b> <code>{prediction_prob:.6f}</code></li>
                        <li><b>Model Used:</b> <code style="color: #4fc3f7">{model_name}</code></li>
                    </ul>
                    """, unsafe_allow_html=True)

                # --- LIME EXPLAINER SECTION ---
                with st.container():
                    st.markdown("""
                    <div style="
                        background-color: rgba(30, 30, 30, 0.85);
                        padding: 25px;
                        border-radius: 12px;
                        border: 1px solid #444;
                        box-shadow: 0 10px 25px rgba(0,0,0,0.4);
                        margin-top: 30px;
                    ">
                        <h3 style="color:white;">🔍 Content Analysis: Highlighted Text</h3>
                    """, unsafe_allow_html=True)

                    with st.spinner("Highlighting important words..."):
                        explainer = LimeTextExplainer(class_names=["Real", "Fake"])
                        exp = explainer.explain_instance(
                            cleaned_text,
                            lime_predict_proba,
                            num_features=10,
                            num_samples=500
                        )

                        features = exp.as_list()
                        max_score = max(abs(score) for _, score in features)

                        highlighted_text = combined_text 

                        for word, score in features:
                            if len(word.strip()) < 2:
                                continue

                            intensity = abs(score) / max_score
                            opacity = 0.15 + intensity * 0.55  # 0.15 → 0.70

                            if score > 0:
                                color = f"rgba(220,38,38,{opacity})"   # FAKE
                            else:
                                color = f"rgba(14,165,233,{opacity})"  # REAL

                            highlighted_text = re.sub(
                                rf"\b({re.escape(word)})\b",
                                rf"<span style='background:{color}; padding:2px 4px; border-radius:4px; font-weight:600;'>\1</span>",
                                highlighted_text,
                                flags=re.IGNORECASE
                            )

                        st.markdown(
                            f"""
                            <div style="
                                background: rgba(255,255,255,0.97);
                                padding: 22px;
                                border-radius: 12px;
                                max-height: 380px;
                                overflow-y: auto;
                                font-size: 0.95rem;
                                line-height: 1.65;
                                color: #000;
                            ">
                                {highlighted_text}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.markdown("</div>", unsafe_allow_html=True)



                with st.container():
                    st.markdown("""
                    <div style="
                        background-color: rgba(30, 30, 30, 0.85);
                        padding: 25px;
                        border-radius: 12px;
                        border: 1px solid #444;
                        box-shadow: 0 10px 25px rgba(0,0,0,0.4);
                        margin-top: 30px;
                    ">
                        <h3 style="color:white;">Top Influential Words</h3>
                    """, unsafe_allow_html=True)
                    
                    features = exp.as_list()
                    fake_words = sorted([f for f in features if f[1] > 0], key=lambda x: x[1], reverse=True)
                    real_words = sorted([f for f in features if f[1] < 0], key=lambda x: x[1])
                    max_chars = 15  # maximum characters before truncating
                    # Find maximum absolute score among all words (both real and fake)
                    all_scores = [abs(score) for _, score in features]
                    max_score = max(all_scores, default=1)

                    col1, col2 = st.columns(2)

                    # REAL evidence
                    with col1:
                        st.markdown(f"""
                        <div style="background: linear-gradient(90deg, #1565C0 0%, #BBDEFB 100%); padding: 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); color: white; display: flex; align-items: center; margin-bottom: 25px;">
                            <div style="background-color:  #0D47A1; width: 60px; height: 60px; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-size: 30px; margin-right: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
                                🛡️
                            </div>
                            <div>
                                <h2 style="margin: 0; color: white; font-weight: 800; font-size: 24px; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">Evidence for REAL</h2>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if not real_words:
                            st.caption("No significant words found pointing to 'Real'.")
                        
                        for word, score in real_words:  
                            display_word = word if len(word) <= max_chars else word[:max_chars] + "..."
                            width_pct = min(abs(score) / max_score * 100, 100)
                            
                            st.markdown(f"""
                            <div class="feature-bar-container" style="margin-bottom:5px;">
                                <div class="feature-bar" style="
                                    width: {width_pct}%; 
                                    min-width: {max_chars*8}px;  /* approximate width for 20 characters */
                                    background: linear-gradient(90deg, #0ea5e9, #38bdf8); 
                                    padding: 5px 10px; 
                                    border-radius: 5px;
                                    color: #FFFFFF;
                                    font-size: 0.8rem;
                                    white-space: nowrap;
                                    overflow: hidden;
                                    text-overflow: ellipsis;
                                ">
                                    {display_word}
                                </div>
                                <span style="margin-left: 10px; color: #FFFFFF; font-size: 0.8rem;">{abs(score):.4f}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    # FAKE evidence
                    with col2:
                        st.markdown(f"""
                        <div style="background: linear-gradient(90deg, #D32F2F 0%, #FFCDD2 100%); padding: 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); color: white; display: flex; align-items: center; margin-bottom: 25px;">
                            <div style="background-color: #B71C1C; width: 60px; height: 60px; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-size: 30px; margin-right: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
                                🚨
                            </div>
                            <div>
                                <h2 style="margin: 0; color: white; font-weight: 800; font-size: 24px; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">Words indicating FAKE</h2>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if not fake_words:
                            st.caption("No significant words found pointing to 'Fake'.")

                        for word, score in fake_words:  
                            display_word = word if len(word) <= max_chars else word[:max_chars] + "..."
                            width_pct = min(abs(score) / max_score * 100, 100)
                            
                            st.markdown(f"""
                            <div class="feature-bar-container" style="margin-bottom:5px;">
                                <div class="feature-bar" style="
                                    width: {width_pct}%; 
                                    min-width: {max_chars*8}px;  /* approximate width for 20 characters */
                                    background: linear-gradient(90deg, #dc2626, #f87171); 
                                    padding: 5px 10px; 
                                    border-radius: 5px;
                                    color: #FFFFFF;
                                    font-size: 0.8rem;
                                    white-space: nowrap;
                                    overflow: hidden;
                                    text-overflow: ellipsis;
                                ">
                                    {display_word}
                                </div>
                                <span style="margin-left: 10px; color: #FFFFFF; font-size: 0.8rem;">{score:.4f}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
    
            except Exception as e:
                st.error(f"Analysis Failed: {e}")
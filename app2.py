import streamlit as st
import tensorflow as tf
from preprocessing import InferencePreprocessor
import nltk
import base64


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

model, model_name, preprocessor = load_model_and_prep('model/best_cnn_rnn_model.keras')


# --- CSS STYLING (THE DESIGN LOGIC) ---
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode() # css understand
    except FileNotFoundError:
        return ""

def inject_custom_css():
    bin_str = get_base64_of_bin_file('fakenews1.jpg') 
    
    st.markdown(f"""
    <style>
        /* 1. Global Background Image */
        .stApp {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center top;
        }}
        
        /* 2. Make the main container transparent so we can control sections individually */
        .block-container {{
            background-color: transparent !important;
            padding-top: 2rem;
            max-width: 1200px;
        }}

        /* 3. Header Styling */
        h1 {{
            color: white !important;
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 800;
            text-shadow: 2px 2px 4px #000000;
            margin-bottom: 0px;
        }}
        .subtitle {{
            color: #ddd;
            font-size: 16px;
            margin-bottom: 30px;
            text-shadow: 1px 1px 2px #000000;
        }}

        /* 4. THE WHITE CARD (Input Section) */
        [data-testid="stForm"] {{
            background-color: rgba(179, 204, 204, 0.95); /* Light grey/white, slight transparency */
            border-radius: 12px;
            padding: 30px;
            border: none;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}

        /* 5. Custom Blue Labels for Inputs */
        .custom-label {{
            background-color: #3f51b5; /* Indigo/Blue */
            color: white;
            padding: 4px 10px;
            font-size: 14px;
            font-weight: bold;
            border-radius: 4px;
            margin-bottom: 8px;
            display: inline-block;
        }}

        /* 6. Input Fields (Dark Mode Style) */
        .stTextInput input, .stTextArea textarea {{
            background-color: #2b2c34 !important; /* Dark Grey/Black */
            color: white !important;
            border-radius: 6px;
            border: 1px solid #444;
        }}
        /* Hide default Streamlit labels (article title,text) since alrd make custom ones  */
        .stTextInput label, .stTextArea label {{
            display: none;
        }}

        /* 7. Fancy Button Style (Button-73 adapted for Streamlit FORM submit) */
        div.stFormSubmitButton > button {{
            appearance: none;
            background-color: #FFFFFF !important;
            border-radius: 40em;
            border-style: none;
            box-shadow: #ADCFFF 0 -12px 6px inset;
            box-sizing: border-box;

            color: #000000 !important;
            cursor: pointer;

            display: inline-block;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 1.2rem;
            font-weight: 700;
            letter-spacing: -0.24px;

            margin: 0;
            outline: none;
            padding: 1rem 1.3rem;

            text-align: center;
            text-decoration: none;

            transition: all 0.15s ease;
            user-select: none;
            -webkit-user-select: none;
            touch-action: manipulation;

            width: 100%;
        }}

        /* Hover effect */
        div.stFormSubmitButton > button:hover {{
            background-color: #07bd06 !important;  
            box-shadow: #057804 0 -6px 8px inset;  /* slightly lighter shadow for depth */
            transform: scale(1.125);
        }}

        /* Active (pressed) */
        div.stFormSubmitButton > button:active {{
            transform: scale(1.025);
        }}

        /* Responsive */
        @media (min-width: 768px) {{
            div.stFormSubmitButton > button {{
                font-size: 1.5rem;
                padding: 0.75rem 2rem;
            }}
        }}

        /* Dark Expander for Diagnostics */
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



# 2. Input Section (The White Card)
with st.form("analysis_form"):
    st.markdown('<h3 style="color:#333; margin-top:0;">Article Submission</h3>', unsafe_allow_html=True)
    
    # Custom Labels + Inputs
    st.markdown('<div class="custom-label">Article Title</div>', unsafe_allow_html=True)
    title = st.text_input("Article Title", placeholder="Paste the headline here...") # Label hidden via CSS

    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True) # Spacer

    st.markdown('<div class="custom-label">Article Body Content</div>', unsafe_allow_html=True)
    text = st.text_area("Article Body Content", placeholder="Paste the content here...", height=200) # Label hidden via CSS
    
    # Submit Button (Centered by CSS width=100%)
    submitted = st.form_submit_button("ANALYZE ARTICLE INTEGRITY")


# 3. Results Section (Transparent Background)
if submitted:
    if not title.strip() or not text.strip():
        st.warning("⚠️ Please enter both a title and article body.")
    else:
        with st.spinner("Analyzing patterns..."):
            # Backend Processing
            try:
                combined_text = f"{title} {text}"
                cleaned_text = preprocessor.clean_text(combined_text)
                input_tensor = tf.convert_to_tensor([[cleaned_text]], dtype=tf.string)
                prediction_prob = model.predict(input_tensor, verbose=0)[0][0]
                
                is_fake = prediction_prob >= 0.5
                
                # --- RESULTS UI ---
                st.markdown("<br>", unsafe_allow_html=True)
                
                # 1. Define Styles based on prediction
                if is_fake:
                    verdict_text = "POTENTIAL MISINFORMATION"
                    verdict_subtext = "Integrity Analysis detects significant anomalies."
                    # Red Gradient
                    verdict_bg = "linear-gradient(90deg, #D32F2F 0%, #FFCDD2 100%)"
                    icon_bg = "#B71C1C"
                    verdict_icon = "🛑"
                    confidence = prediction_prob
                    conf_label = "Confidence Score (Fake)"
                    bar_color = "#D32F2F"
                
                else:
                    verdict_text = "VERIFIED AUTHENTICITY"
                    verdict_subtext = "Content aligns with standard reporting patterns."
                    # Blue Gradient
                    verdict_bg = "linear-gradient(90deg, #1565C0 0%, #BBDEFB 100%)"
                    icon_bg = "#0D47A1"
                    verdict_icon = "🛡️"
                    confidence = 1 - prediction_prob
                    conf_label = "Confidence Score (Real)"
                    bar_color = "#1565C0"

                # 2. Render the "Verdict Card"
                # NOTICE: The HTML below is aligned to the LEFT to prevent it from becoming a code block
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

                # 3. Metrics & Progress Bar (Updated Design)
                col_res1, col_res2 = st.columns([1, 4])
                
                with col_res1:
                    # Big Number
                    st.markdown(f"""
                    <div style="text-align: center; background: rgba(255,255,255,0.9); padding: 10px; border-radius: 10px;">
                        <div style="color: #555; font-size: 12px; font-weight: bold; text-transform: uppercase;">{conf_label}</div>
                        <div style="color: {bar_color}; font-size: 36px; font-weight: 900;">{confidence:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_res2:
                    st.write("") # Spacer to align
                    st.write("") 
                    # Progress Bar
                    st.markdown(f"""
                    <div style="background-color: rgba(255,255,255,0.5); border-radius: 20px; height: 25px; width: 100%; overflow: hidden; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);">
                        <div style="width: {confidence*100}%; background-color: {bar_color}; height: 100%; border-radius: 20px; transition: width 1s ease-in-out;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                        <span style="font-size: 12px; color: #fff; background: rgba(0,0,0,0.5); padding: 2px 6px; border-radius: 4px;">Low Confidence</span>
                        <span style="font-size: 12px; color: #fff; background: rgba(0,0,0,0.5); padding: 2px 6px; border-radius: 4px;">High Confidence</span>
                    </div>
                    """, unsafe_allow_html=True)

                # System Diagnostics (Expander)
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("System Diagnostics"):
                    st.markdown(f"""
                    <ul style="color: #ccc;">
                        <li><b>Raw Probability:</b> <code>{prediction_prob:.6f}</code></li>
                        <li><b>Model Used:</b> <code style="color: #4fc3f7">{model_name}</code></li>
                    </ul>
                    <p style="color: #888; font-style: italic; font-size: 12px; margin-left: 20px;">(This prediction is based on structural, lexical, and semantic features extracted from the text.)</p>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Analysis Failed: {e}")
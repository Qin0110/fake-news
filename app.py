import streamlit as st
import tensorflow as tf
from preprocessing import InferencePreprocessor


@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model/best_cnn_rnn_model.keras')

@st.cache_resource
def load_preprocessor():
    return InferencePreprocessor()

model = load_model()
preprocessor = load_preprocessor()

# UI
st.title("📰 Fake News Detection System")
st.markdown("Enter a news article below to detect if it is **Real** or **Fake**.")

title = st.text_input("Article Title", placeholder="Paste the headline here...")
text = st.text_area("Article Body", placeholder="Paste the content here...", height=200)


# Prediction
if st.button("Analyze News"):
    if not title.strip() or not text.strip():
        st.warning("Please provide both a title and the body text.")
    else:
        with st.spinner("Analyzing patterns..."):
            try:
                combined_text = f"{title} {text}"
                cleaned_text = preprocessor.clean_text(combined_text)

                # TensorFlow string tensor 
                input_tensor = tf.convert_to_tensor([[cleaned_text]], dtype=tf.string)

                prediction_prob = model.predict(input_tensor, verbose=0)[0][0]
                is_fake = prediction_prob >= 0.5
                
                st.divider()

                if is_fake:
                    st.error("🚨 **Prediction: FAKE NEWS**")
                    bar_color = "#ff4b4b" 
                    confidence = prediction_prob
                else:
                    st.success("✅ **Prediction: REAL NEWS**")
                    bar_color = "#21c354" 
                    confidence = 1 - prediction_prob

                st.info(f"Confidence Score: {confidence:.2%}")

                # Bar colooorrrr
                st.markdown(f"""
                <div style="background-color: #e6e9ef; border-radius: 5px; height: 16px; width: 100%;">
                    <div style="background-color: {bar_color}; width: {confidence * 100}%; height: 100%; border-radius: 5px; transition: width 2s;">
                    </div>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

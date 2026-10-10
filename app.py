"""Streamlit front-end for the CNN + RNN fake-news detector.

Run with:  streamlit run app.py
"""

import base64
import html
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from lime.lime_text import LimeTextExplainer

from preprocessing import InferencePreprocessor

# Optional: PyAudio needs native portaudio bindings that are not everywhere.
try:
    import speech_recognition as sr
    SPEECH_IMPORT_ERROR = None
except Exception as _exc:
    sr = None
    SPEECH_IMPORT_ERROR = str(_exc)


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = APP_DIR / "model" / "best_cnn_rnn_model.keras"
MODEL_SEARCH_DIRS = ("model", "LinuxEnv")
BACKGROUND_IMAGE = APP_DIR / "fakenews5.jpg"

MIN_WORDS = 20                 # shortest article we will score
LANG_CHECK_MIN_CHARS = 60      # below this, langdetect is too unreliable to block on
MAX_HISTORY = 10

st.set_page_config(
    layout="wide",
    page_title="Fake News Detection",
    page_icon="📰",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner="Loading detection model…")
def load_resources(model_path):
    """Returns (model, preprocessor, errors).

    Renders nothing: cached functions skip their output on a cache hit.
    """
    errors = []

    model = None
    if not os.path.exists(model_path):
        errors.append(
            f"Model file not found: `{model_path}`. Train a model first "
            "(`python cnn+rnn.py`) or check the path in the sidebar."
        )
    else:
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as exc:
            errors.append(f"Could not load the model from `{model_path}` — {exc}")

    preprocessor = None
    try:
        preprocessor = InferencePreprocessor()
    except Exception as exc:
        errors.append(
            f"Could not initialise the text preprocessor — {exc}. "
            "This usually means the NLTK corpora could not be downloaded; "
            "run `python verify_setup.py` for a detailed report."
        )

    return model, preprocessor, errors


@st.cache_data(show_spinner=False)
def discover_models():
    """Every .keras checkpoint shipped with the project."""
    found = []
    for folder in MODEL_SEARCH_DIRS:
        directory = APP_DIR / folder
        if directory.is_dir():
            found.extend(sorted(str(p) for p in directory.glob("*.keras")))
    if str(DEFAULT_MODEL) in found:
        found.remove(str(DEFAULT_MODEL))
        found.insert(0, str(DEFAULT_MODEL))
    return found


@st.cache_data(show_spinner=False)
def get_background_data_uri(path):
    """Data URI for the page background, or '' when unavailable."""
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return ""
    suffix = Path(path).suffix.lower()
    mime = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
    }.get(suffix, "image/jpeg")
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def predict_proba_fake(model, cleaned_texts, batch_size=64):
    """P(fake) for already-cleaned documents, as a 1-D array."""
    tensor = tf.constant(cleaned_texts, dtype=tf.string)
    tensor = tf.reshape(tensor, (-1, 1))
    probs = model.predict(tensor, verbose=0, batch_size=batch_size)
    return np.asarray(probs, dtype=np.float64).reshape(-1)


def make_lime_predictor(model):
    """Callable returning (N, 2) probabilities as [Real, Fake].

    LIME perturbs the already-cleaned string, so do NOT re-clean here --
    re-running NLTK on every sample was the main source of latency.
    """
    def predictor(text_list):
        texts = [t if isinstance(t, str) else str(t) for t in text_list]
        if not texts:
            return np.empty((0, 2))
        prob_fake = predict_proba_fake(model, texts)
        return np.column_stack([1.0 - prob_fake, prob_fake])
    return predictor


_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _score_for_word(word, scores, preprocessor):
    """Look up a surface word against the lemmatised LIME features."""
    lowered = word.lower()
    if lowered in scores:
        return scores[lowered]
    for variant in preprocessor.lemma_variants(lowered):
        if variant in scores:
            return scores[variant]
    return None


def build_highlighted_html(raw_text, scores, preprocessor):
    """Escape raw_text and tint influential words.

    Walks the text once, so a span inserted for one word cannot be re-matched
    by another -- a chained re.sub corrupts the markup as soon as a feature is
    a CSS keyword like "background". Escaping stops article text injecting HTML.
    """
    if not raw_text:
        return "<em>No text to display.</em>"

    if not scores:
        return html.escape(raw_text).replace("\n", "<br>")

    max_abs = max(abs(v) for v in scores.values()) or 1.0

    parts = []
    cursor = 0
    for match in _WORD_RE.finditer(raw_text):
        parts.append(html.escape(raw_text[cursor:match.start()]))
        word = match.group(0)
        score = _score_for_word(word, scores, preprocessor)

        if score is None or len(word) < 2:
            parts.append(html.escape(word))
        else:
            opacity = 0.18 + (abs(score) / max_abs) * 0.55
            colour = (f"rgba(220,38,38,{opacity:.2f})" if score > 0
                      else f"rgba(14,165,233,{opacity:.2f})")
            parts.append(
                f"<span style=\"background:{colour}; padding:2px 4px; "
                f"border-radius:4px; font-weight:600;\" "
                f"title=\"{'fake' if score > 0 else 'real'} {score:+.4f}\">"
                f"{html.escape(word)}</span>"
            )
        cursor = match.end()

    parts.append(html.escape(raw_text[cursor:]))
    return "".join(parts).replace("\n", "<br>")


def render_feature_bars(features, positive):
    """Horizontal bars for the words pushing toward one class."""
    if not features:
        st.caption(
            "No significant words found pointing to "
            f"'{'Fake' if positive else 'Real'}'."
        )
        return

    max_abs = max(abs(score) for _, score in features) or 1.0
    gradient = ("linear-gradient(90deg, #dc2626, #f87171)" if positive
                else "linear-gradient(90deg, #0ea5e9, #38bdf8)")
    max_chars = 15

    rows = []
    for word, score in features:
        display_word = word if len(word) <= max_chars else word[:max_chars] + "…"
        width_pct = min(abs(score) / max_abs * 100, 100)
        rows.append(f"""
        <div class="feature-bar-container">
            <div class="feature-bar" style="width:{width_pct:.1f}%;
                 min-width:{max_chars * 8}px; background:{gradient};">
                {html.escape(display_word)}
            </div>
            <span class="feature-score">{score:+.4f}</span>
        </div>
        """)
    st.markdown("".join(rows), unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def microphone_available():
    """Whether a usable capture device exists, without raising."""
    if sr is None:
        return False
    try:
        return len(sr.Microphone.list_microphone_names()) > 0
    except Exception:
        return False


def transcribe_speech():
    """Record from the default microphone and return the transcript, or None."""
    if sr is None:
        st.error(f"Speech recognition is unavailable: {SPEECH_IMPORT_ERROR}")
        return None

    recognizer = sr.Recognizer()
    placeholder = st.empty()
    try:
        # sr.Microphone() raises when PyAudio/the device is missing, so it
        # must sit inside the try.
        with sr.Microphone() as source:
            placeholder.info("🎤 Listening… speak now.")
            recognizer.adjust_for_ambient_noise(source, duration=0.6)
            audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=30)
        placeholder.info("Transcribing audio…")
        text = recognizer.recognize_google(audio_data)
        placeholder.empty()
        return text
    except sr.WaitTimeoutError:
        placeholder.warning("No speech detected — please try again.")
    except sr.UnknownValueError:
        placeholder.error("Could not understand the audio.")
    except sr.RequestError as exc:
        placeholder.error(f"Speech API error: {exc}")
    except OSError as exc:
        placeholder.error(f"No usable microphone was found ({exc}).")
    except Exception as exc:
        placeholder.error(f"Microphone error: {exc}")
    return None


def build_markdown_report(result):
    """Markdown summary of one analysis."""
    lines = [
        "# Fake News Detection — Analysis Report",
        "",
        f"- **Generated:** {result['timestamp']}",
        f"- **Model:** {result['model_name']}",
        f"- **Verdict:** {result['verdict_text']}",
        f"- **{result['conf_label']}:** {result['confidence']:.2%}",
        f"- **Raw P(fake):** {result['probability']:.6f}",
        f"- **Decision threshold:** {result['threshold']:.2f}",
        f"- **Inference time:** {result['latency_ms']:.0f} ms",
        "",
        "## Article",
        "",
        f"**Title:** {result['title']}",
        "",
        result["body"],
        "",
        "## Most influential words",
        "",
        "| Word | Contribution | Points toward |",
        "| --- | ---: | --- |",
    ]
    for word, score in result["features"]:
        lines.append(f"| {word} | {score:+.4f} | {'Fake' if score > 0 else 'Real'} |")

    if not result["features"]:
        lines.append("| _none_ | — | — |")

    lines += [
        "",
        "---",
        "",
        "_Automated assistance only. A model prediction is not a fact-check; "
        "always corroborate with primary sources._",
    ]
    return "\n".join(lines)


def inject_custom_css():
    # str, not Path: st.cache_data hashes its arguments and str is always hashable.
    background = get_background_data_uri(str(BACKGROUND_IMAGE))
    background_rule = (
        f'background-image: url("{background}");' if background
        else "background: linear-gradient(160deg, #0f2027, #203a43, #2c5364);"
    )

    st.markdown(f"""
    <style>
        .stApp {{
            {background_rule}
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

        /* The input card */
        [data-testid="stForm"] {{
            background-color: rgba(179, 204, 204, 0.95);
            border-radius: 12px;
            padding: 30px;
            border: none;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}

        /* Dark panel used for every results section */
        .analysis-card {{
            background-color: rgba(30, 30, 30, 0.88);
            border-radius: 12px;
            padding: 25px 25px 10px 25px;
            margin-top: 30px;
            border: 1px solid #444;
            box-shadow: 0 10px 25px rgba(0,0,0,0.4);
        }}
        .analysis-card h3 {{
            color: #ffffff;
            margin-top: 0;
        }}
        .analysis-card p.hint {{
            color: #b0b0b0;
            font-size: 0.85rem;
            margin: 4px 0 0 0;
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
        div.stFormSubmitButton > button:hover:enabled {{
            background-color: #07bd06 !important;
            box-shadow: #057804 0 -6px 8px inset;
            transform: scale(1.05);
        }}
        div.stFormSubmitButton > button:active:enabled {{
            transform: scale(1.02);
        }}

        /* Word contribution bars */
        .feature-bar-container {{
            margin-bottom: 6px;
            display: flex;
            align-items: center;
        }}
        .feature-bar {{
            height: 28px;
            border-radius: 6px;
            display: flex;
            align-items: center;
            padding: 0 10px;
            color: #ffffff;
            font-size: 0.85rem;
            font-weight: 600;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            box-shadow: 0 2px 4px rgba(0,0,0,0.15);
        }}
        .feature-score {{
            margin-left: 10px;
            color: #ffffff;
            font-size: 0.8rem;
            font-variant-numeric: tabular-nums;
        }}

        .highlight-body {{
            background: rgba(255,255,255,0.97);
            padding: 22px;
            border-radius: 12px;
            max-height: 380px;
            overflow-y: auto;
            font-size: 0.95rem;
            line-height: 1.8;
            color: #111;
        }}

        .legend {{
            display: flex;
            gap: 18px;
            align-items: center;
            margin: 0 0 14px 0;
            color: #ddd;
            font-size: 0.85rem;
        }}
        .legend .swatch {{
            display: inline-block;
            width: 14px;
            height: 14px;
            border-radius: 3px;
            margin-right: 6px;
            vertical-align: -2px;
        }}
    </style>
    """, unsafe_allow_html=True)


def card_open(title, hint=None):
    hint_html = f'<p class="hint">{html.escape(hint)}</p>' if hint else ""
    st.markdown(
        f'<div class="analysis-card"><h3>{title}</h3>{hint_html}</div>',
        unsafe_allow_html=True,
    )


SAMPLE_ARTICLES = {
    "Wire-service style": {
        "title": "Regional transport authority approves budget for rail upgrade",
        "text": (
            "The regional transport authority voted on Tuesday to approve a "
            "multi-year budget covering signalling upgrades and platform "
            "extensions across the commuter rail network. Officials said the "
            "programme would be delivered in phases, with the first works "
            "beginning next spring and the final stage scheduled for 2029. "
            "According to the published planning document, the authority expects "
            "journey times on the busiest corridor to fall by roughly four "
            "minutes once the signalling work is complete. A spokesperson said "
            "the board had reviewed three independent cost assessments before "
            "the vote, and that quarterly progress reports would be published "
            "on the authority's website. Two board members abstained, citing "
            "concerns about contingency funding for the later phases."
        ),
    },
    "Sensational style": {
        "title": "SHOCKING: Scientists ADMIT everything you were told was a lie",
        "text": (
            "In a stunning development that the mainstream media absolutely "
            "refuses to cover, insiders have finally admitted what millions of "
            "people already suspected. Anonymous sources say the truth has been "
            "deliberately buried for decades, and the evidence is absolutely "
            "undeniable. Experts everywhere are reportedly stunned and cannot "
            "explain how this was allowed to continue for so long. Officials "
            "declined to comment, which critics say proves the cover-up goes all "
            "the way to the top. Share this before it gets taken down, because "
            "they do not want you to see it. Wake up and do your own research "
            "now, because nobody else is going to tell you what is really "
            "happening behind closed doors."
        ),
    },
}


def init_state():
    defaults = {
        "text_buffer": "",
        "title_buffer": "",
        "input_version": 0,     # bumped to force the widgets to re-read the buffers
        "result": None,
        "history": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def load_sample(name):
    sample = SAMPLE_ARTICLES[name]
    st.session_state.title_buffer = sample["title"]
    st.session_state.text_buffer = sample["text"]
    st.session_state.input_version += 1
    st.session_state.result = None


def clear_inputs():
    st.session_state.title_buffer = ""
    st.session_state.text_buffer = ""
    st.session_state.input_version += 1
    st.session_state.result = None


def render_results(result, preprocessor):
    if result["is_fake"]:
        verdict_bg = "linear-gradient(90deg, #D32F2F 0%, #FFCDD2 100%)"
        icon_bg, verdict_icon, bar_color = "#B71C1C", "🛑", "#D32F2F"
    else:
        verdict_bg = "linear-gradient(90deg, #1565C0 0%, #BBDEFB 100%)"
        icon_bg, verdict_icon, bar_color = "#0D47A1", "🛡️", "#1565C0"

    confidence = result["confidence"]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background: {verdict_bg}; padding: 20px; border-radius: 12px;
         box-shadow: 0 4px 15px rgba(0,0,0,0.2); color: white; display: flex;
         align-items: center; margin-bottom: 25px;">
        <div style="background-color: {icon_bg}; width: 60px; height: 60px;
             border-radius: 50%; display: flex; justify-content: center;
             align-items: center; font-size: 30px; margin-right: 20px;
             box-shadow: 0 2px 5px rgba(0,0,0,0.2);">{verdict_icon}</div>
        <div>
            <h2 style="margin: 0; color: white; font-weight: 800; font-size: 24px;
                text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{result['verdict_text']}</h2>
            <p style="margin: 5px 0 0 0; color: rgba(255,255,255,0.9);
               font-size: 14px;">{result['verdict_subtext']}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_res1, col_res2 = st.columns([1, 4])
    with col_res1:
        st.markdown(f"""
        <div style="text-align: center; background: rgba(255,255,255,0.9);
             padding: 10px; border-radius: 10px;">
            <div style="color: #555; font-size: 12px; font-weight: bold;
                 text-transform: uppercase;">{result['conf_label']}</div>
            <div style="color: {bar_color}; font-size: 36px;
                 font-weight: 900;">{confidence:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_res2:
        st.write("")
        st.write("")
        st.markdown(f"""
        <div style="background-color: rgba(255,255,255,0.5); border-radius: 20px;
             height: 25px; width: 100%; overflow: hidden;
             box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);">
            <div style="width: {confidence * 100:.2f}%; background-color: {bar_color};
                 height: 100%; border-radius: 20px;
                 transition: width 1s ease-in-out;"></div>
        </div>
        """, unsafe_allow_html=True)

    # Diagnostics
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("System diagnostics"):
        diag_left, diag_right = st.columns(2)
        with diag_left:
            st.markdown(f"""
            - **Raw P(fake):** `{result['probability']:.6f}`
            - **Decision threshold:** `{result['threshold']:.2f}`
            - **Model:** `{result['model_name']}`
            """)
        with diag_right:
            st.markdown(f"""
            - **Words submitted:** `{result['word_count']}`
            - **Tokens after cleaning:** `{result['token_count']}`
            - **Inference time:** `{result['latency_ms']:.0f} ms`
            """)
        if result["token_count"] < 10:
            st.warning(
                "Very few tokens survived cleaning, so this prediction rests on "
                "little evidence. Treat it with caution."
            )

    features = result["features"]

    # Highlighted article
    if result["explanation_available"]:
        card_open(
            "🔍 Content analysis: highlighted text",
            "Shading shows how strongly each word pushed the prediction. "
            "Hover a word to see its exact contribution.",
        )
        st.markdown("""
        <div class="legend">
            <span><span class="swatch" style="background:rgba(220,38,38,0.65)"></span>
                  pushes toward Fake</span>
            <span><span class="swatch" style="background:rgba(14,165,233,0.65)"></span>
                  pushes toward Real</span>
        </div>
        """, unsafe_allow_html=True)

        highlighted = build_highlighted_html(
            result["combined_text"], result["score_map"], preprocessor
        )
        st.markdown(
            f'<div class="highlight-body">{highlighted}</div>',
            unsafe_allow_html=True,
        )

        # Contribution bars
        card_open("Top influential words")

        fake_words = sorted([f for f in features if f[1] > 0],
                            key=lambda x: x[1], reverse=True)
        real_words = sorted([f for f in features if f[1] < 0], key=lambda x: x[1])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(90deg, #1565C0 0%, #BBDEFB 100%);
                 padding: 14px 20px; border-radius: 12px; color: white;
                 display: flex; align-items: center; margin-bottom: 18px;">
                <div style="background-color: #0D47A1; width: 48px; height: 48px;
                     border-radius: 50%; display: flex; justify-content: center;
                     align-items: center; font-size: 24px; margin-right: 16px;">🛡️</div>
                <h2 style="margin: 0; color: white; font-weight: 800;
                    font-size: 21px;">Evidence for REAL</h2>
            </div>
            """, unsafe_allow_html=True)
            render_feature_bars(real_words, positive=False)

        with col2:
            st.markdown("""
            <div style="background: linear-gradient(90deg, #D32F2F 0%, #FFCDD2 100%);
                 padding: 14px 20px; border-radius: 12px; color: white;
                 display: flex; align-items: center; margin-bottom: 18px;">
                <div style="background-color: #B71C1C; width: 48px; height: 48px;
                     border-radius: 50%; display: flex; justify-content: center;
                     align-items: center; font-size: 24px; margin-right: 16px;">🚨</div>
                <h2 style="margin: 0; color: white; font-weight: 800;
                    font-size: 21px;">Evidence for FAKE</h2>
            </div>
            """, unsafe_allow_html=True)
            render_feature_bars(fake_words, positive=True)
    else:
        st.info(
            "Word-level explanation unavailable: "
            + result.get("explanation_error", "not enough distinct tokens.")
        )

    # Export
    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        "⬇️ Download analysis report (Markdown)",
        data=build_markdown_report(result),
        file_name=f"fake_news_report_{result['file_stamp']}.md",
        mime="text/markdown",
    )

    st.caption(
        "⚖️ This tool offers automated assistance, not a verdict of fact. "
        "Model predictions can be wrong — always corroborate with primary sources."
    )


def render_history():
    history = st.session_state.history
    if not history:
        return
    with st.expander(f"🕓 Session history ({len(history)})"):
        for entry in reversed(history):
            icon = "🛑" if entry["is_fake"] else "🛡️"
            st.markdown(
                f"{icon} **{entry['verdict_text']}** — {entry['confidence']:.1%} · "
                f"`{entry['timestamp']}` · {html.escape(entry['title'][:80])}"
            )


inject_custom_css()
init_state()

col_h1, col_h2 = st.columns([0.05, 0.95])
with col_h2:
    st.markdown("<h1>📰  FAKE NEWS DETECTION SYSTEM </h1>", unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Utilizing advanced deep learning models for '
        'news integrity assessment.</p>',
        unsafe_allow_html=True,
    )

# Sidebar controls
available_models = discover_models()

with st.sidebar:
    st.header("⚙️ Settings")

    if available_models:
        model_path = st.selectbox(
            "Model checkpoint",
            available_models,
            format_func=lambda p: os.path.relpath(p, APP_DIR),
        )
    else:
        model_path = str(DEFAULT_MODEL)
        st.warning("No .keras checkpoints found.")

    threshold = st.slider(
        "Decision threshold", 0.05, 0.95, 0.50, 0.01,
        help="P(fake) at or above this value is reported as misinformation. "
             "Lower it to catch more fakes at the cost of false alarms.",
    )

    st.divider()
    st.subheader("Explanation (LIME)")
    explain_enabled = st.checkbox("Explain the prediction", value=True)
    num_features = st.slider("Words to highlight", 5, 25, 10, 1,
                             disabled=not explain_enabled)
    num_samples = st.slider("LIME samples", 200, 2000, 500, 100,
                            disabled=not explain_enabled,
                            help="More samples give a steadier explanation "
                                 "but take longer.")

    st.divider()
    strict_language = st.checkbox(
        "Enforce English-only input", value=True,
        help="The model was trained on English news. Uncheck to analyse "
             "anyway when language detection misfires.",
    )

model, preprocessor, load_errors = load_resources(model_path)
if load_errors:
    for message in load_errors:
        st.error(message)
    st.stop()

model_name = os.path.basename(model_path)

# Quick actions
action_cols = st.columns([0.22, 0.22, 0.15, 0.41])
with action_cols[0]:
    if st.button("📄 Load neutral sample"):
        load_sample("Wire-service style")
        st.rerun()
with action_cols[1]:
    if st.button("🔥 Load sensational sample"):
        load_sample("Sensational style")
        st.rerun()
with action_cols[2]:
    if st.button("🧹 Clear"):
        clear_inputs()
        st.rerun()

# Input form
version = st.session_state.input_version
mic_ok = microphone_available()

with st.form("analysis_form"):
    st.markdown('<h1 style="color:#333; margin-top:0;">Analyze Article</h1>',
                unsafe_allow_html=True)

    st.markdown('<h1 style="color:#333; font-size:25px;">Article Title</h1>',
                unsafe_allow_html=True)
    title = st.text_input(
        "Article Title",
        value=st.session_state.title_buffer,
        placeholder="Paste the headline here...",
        key=f"title_input_{version}",
    )
    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)

    st.markdown('<h1 style="color:#333; font-size:25px;">Article Body Content</h1>',
                unsafe_allow_html=True)
    text = st.text_area(
        "Article Body Content",
        value=st.session_state.text_buffer,
        placeholder="Paste content or click Record...",
        height=200,
        key=f"body_input_{version}",
    )

    word_count = len(text.split())
    if 0 < word_count < MIN_WORDS:
        st.markdown(f"""
            <div style="padding: 10px; border-radius: 10px;
                 background-color: #ffcccc; color: #EE4B2B; font-weight: bold;
                 margin-bottom: 15px;">
                ⚠️ Please enter at least {MIN_WORDS} words. Current: {word_count}
            </div>
            """, unsafe_allow_html=True)

    col_submit, col_record = st.columns([0.85, 0.15])
    with col_submit:
        submitted = st.form_submit_button("ANALYZE ARTICLE INTEGRITY")
    with col_record:
        record_click = st.form_submit_button(
            "🎤 Record",
            type="secondary",
            disabled=not mic_ok,
            help=None if mic_ok else
            "No microphone detected (PyAudio may be missing on this machine).",
        )

# Speech capture
if record_click:
    # Persist typed text so the recording appends instead of replacing.
    st.session_state.title_buffer = title
    transcribed_text = transcribe_speech()
    if transcribed_text:
        current = text.strip()
        st.session_state.text_buffer = (
            f"{current} {transcribed_text}".strip() if current else transcribed_text
        )
        st.session_state.input_version += 1
        st.rerun()
    else:
        st.session_state.text_buffer = text

# Analysis
if submitted:
    st.session_state.title_buffer = title
    st.session_state.text_buffer = text

    combined_text = f"{title.strip()} {text.strip()}".strip()
    blocking_error = None

    if not title.strip() or not text.strip():
        blocking_error = "⚠️ Please enter both a title and article body."
    elif word_count < MIN_WORDS:
        blocking_error = (
            f"⚠️ Please enter at least {MIN_WORDS} words "
            f"(currently {word_count})."
        )
    elif strict_language and not preprocessor.is_english(
            combined_text, min_chars=LANG_CHECK_MIN_CHARS):
        blocking_error = (
            "⚠️ Unsupported language detected. This system supports English "
            "articles only — untick *Enforce English-only input* in the "
            "sidebar to analyse anyway."
        )

    if blocking_error:
        st.warning(blocking_error)
        st.session_state.result = None
    else:
        try:
            with st.spinner("Analyzing patterns…"):
                cleaned_text = preprocessor.clean_text(combined_text)
                tokens = cleaned_text.split()

                if not tokens:
                    raise ValueError(
                        "The input text could not be processed — nothing "
                        "remained after cleaning. Please ensure it contains "
                        "valid English content."
                    )

                started = time.perf_counter()
                probability = float(predict_proba_fake(model, [cleaned_text])[0])
                latency_ms = (time.perf_counter() - started) * 1000.0

                is_fake = probability >= threshold
                if is_fake:
                    verdict_text = "POTENTIAL MISINFORMATION"
                    verdict_subtext = "Integrity analysis detects significant anomalies."
                    confidence = probability
                    conf_label = "Confidence Score (Fake)"
                else:
                    verdict_text = "VERIFIED AUTHENTICITY"
                    verdict_subtext = "Content aligns with standard reporting patterns."
                    confidence = 1.0 - probability
                    conf_label = "Confidence Score (Real)"

            features = []
            explanation_available = False
            explanation_error = ""

            if explain_enabled:
                if len(set(tokens)) < 2:
                    explanation_error = "the article has too few distinct words."
                else:
                    with st.spinner("Highlighting important words…"):
                        try:
                            explainer = LimeTextExplainer(
                                class_names=["Real", "Fake"], random_state=42
                            )
                            explanation = explainer.explain_instance(
                                cleaned_text,
                                make_lime_predictor(model),
                                num_features=num_features,
                                num_samples=num_samples,
                                labels=(1,),
                            )
                            features = explanation.as_list(label=1)
                            explanation_available = True
                        except Exception as exc:
                            explanation_error = f"LIME failed — {exc}"
            else:
                explanation_error = "explanations are switched off in the sidebar."

            now = datetime.now()
            st.session_state.result = {
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                "file_stamp": now.strftime("%Y%m%d_%H%M%S"),
                "title": title.strip(),
                "body": text.strip(),
                "combined_text": combined_text,
                "cleaned_text": cleaned_text,
                "word_count": word_count,
                "token_count": len(tokens),
                "probability": probability,
                "threshold": threshold,
                "is_fake": bool(is_fake),
                "confidence": float(confidence),
                "conf_label": conf_label,
                "verdict_text": verdict_text,
                "verdict_subtext": verdict_subtext,
                "model_name": model_name,
                "latency_ms": latency_ms,
                "features": features,
                "score_map": {word.lower(): score for word, score in features},
                "explanation_available": explanation_available,
                "explanation_error": explanation_error,
            }

            st.session_state.history.append({
                "timestamp": st.session_state.result["timestamp"],
                "title": title.strip() or "(untitled)",
                "verdict_text": verdict_text,
                "confidence": float(confidence),
                "is_fake": bool(is_fake),
            })
            del st.session_state.history[:-MAX_HISTORY]

        except Exception as exc:
            st.session_state.result = None
            st.error(f"Analysis failed: {exc}")

# Outside the submit branch, so a sidebar change (which reruns) keeps results.
if st.session_state.result:
    render_results(st.session_state.result, preprocessor)

render_history()

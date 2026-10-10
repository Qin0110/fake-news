"""End-to-end health check for the fake-news detection project.

Run this before launching the app -- it verifies, in order:

  1. Python version
  2. Required third-party packages
  3. NLTK corpora (downloading anything missing)
  4. The text-cleaning pipeline
  5. The saved Keras model, including one real prediction
  6. Optional extras (speech input, dataset splits)

Usage:
    python verify_setup.py
    python verify_setup.py --model model/best_cnn_rnn_model.keras

Exit code is 0 when everything required for `streamlit run app.py` works,
and 1 otherwise.
"""

import argparse
import importlib
import os
import sys

PASS = "[ OK ]"
FAIL = "[FAIL]"
WARN = "[WARN]"

# (import name, pip name)
REQUIRED_PACKAGES = [
    ("streamlit", "streamlit"),
    ("tensorflow", "tensorflow"),
    ("keras", "keras"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("joblib", "joblib"),
    ("nltk", "nltk"),
    ("contractions", "contractions"),
    ("ftfy", "ftfy"),
    ("langdetect", "langdetect"),
    ("lime", "lime"),
    ("matplotlib", "matplotlib"),
]

# Not needed by app.py: speech is optional, seaborn/keras-tuner are training-only.
OPTIONAL_PACKAGES = [
    ("speech_recognition", "SpeechRecognition"),
    ("pyaudio", "PyAudio"),
    ("seaborn", "seaborn"),
    ("keras_tuner", "keras-tuner"),
]

SAMPLE_TEXT = (
    "Regional transport authority approves budget for rail upgrade. "
    "The authority voted on Tuesday to approve a multi-year budget covering "
    "signalling upgrades and platform extensions across the commuter network."
)


class Reporter:
    def __init__(self):
        self.failures = []
        self.warnings = []

    def section(self, title):
        print(f"\n{title}\n{'-' * len(title)}")

    def ok(self, message):
        print(f"{PASS} {message}")

    def warn(self, message):
        print(f"{WARN} {message}")
        self.warnings.append(message)

    def fail(self, message):
        print(f"{FAIL} {message}")
        self.failures.append(message)


def check_python(report):
    report.section("1. Python")
    version = sys.version_info
    label = f"Python {version.major}.{version.minor}.{version.micro}"
    if version < (3, 9):
        report.fail(f"{label} is too old; this project targets Python 3.10-3.12.")
    elif version >= (3, 13):
        report.warn(f"{label} is newer than the pinned TensorFlow supports "
                    "(3.10-3.12 recommended).")
    else:
        report.ok(label)


def check_packages(report):
    report.section("2. Packages")
    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "unknown")
            report.ok(f"{pip_name} ({version})")
        except Exception as exc:
            report.fail(f"{pip_name} is missing or broken - {exc}")

    for import_name, pip_name in OPTIONAL_PACKAGES:
        try:
            importlib.import_module(import_name)
            report.ok(f"{pip_name} (optional)")
        except Exception:
            report.warn(f"{pip_name} unavailable - optional feature disabled.")


def check_nltk(report):
    report.section("3. NLTK corpora")
    try:
        from preprocessing import ensure_nltk_data, _resource_present
    except Exception as exc:
        report.fail(f"Could not import preprocessing.py - {exc}")
        return

    try:
        unavailable = ensure_nltk_data(quiet=True, force=True)
    except Exception as exc:
        report.fail(f"NLTK download step raised - {exc}")
        return

    tagger_ok = False
    for path in ("taggers/averaged_perceptron_tagger_eng",
                 "taggers/averaged_perceptron_tagger"):
        if _resource_present(path):
            tagger_ok = True
            report.ok(f"POS tagger present ({path.split('/')[-1]})")
            break

    if not tagger_ok:
        report.warn("No POS tagger available - lemmatisation falls back to "
                    "noun-only, which slightly changes token output.")

    if unavailable:
        report.warn("Could not download: " + ", ".join(unavailable))
    else:
        report.ok("All corpora available.")


def check_preprocessing(report):
    report.section("4. Text cleaning")
    try:
        from preprocessing import InferencePreprocessor
        preprocessor = InferencePreprocessor()
    except Exception as exc:
        report.fail(f"InferencePreprocessor could not be created - {exc}")
        return None

    try:
        cleaned = preprocessor.clean_text(SAMPLE_TEXT)
    except Exception as exc:
        report.fail(f"clean_text() raised - {exc}")
        return None

    if not cleaned.strip():
        report.fail("clean_text() returned an empty string for valid English input.")
        return None

    report.ok(f"clean_text() -> {len(cleaned.split())} tokens: "
              f"{' '.join(cleaned.split()[:8])} ...")

    try:
        if preprocessor.is_english(SAMPLE_TEXT):
            report.ok("Language detection identifies the sample as English.")
        else:
            report.warn("Language detection did not identify the sample as English.")
    except Exception as exc:
        report.warn(f"is_english() raised - {exc}")

    # Guard the exact edge cases that used to crash the app.
    for label, value in (("empty string", ""), ("whitespace", "   "),
                         ("punctuation only", "!!! ??? ..."), ("None", None)):
        try:
            preprocessor.clean_text(value)
            report.ok(f"Handles {label} input without raising.")
        except Exception as exc:
            report.fail(f"clean_text({label}) raised - {exc}")

    return preprocessor


def check_model(report, model_path, preprocessor):
    report.section("5. Model")
    if not os.path.exists(model_path):
        report.fail(f"Model file not found: {model_path}")
        return

    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    report.ok(f"Found {model_path} ({size_mb:.1f} MB)")

    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
    except Exception as exc:
        report.fail(f"Model failed to load - {exc}")
        return

    report.ok(f"Loaded - {model.count_params():,} parameters, "
              f"{len(model.layers)} layers.")

    if preprocessor is None:
        report.warn("Skipping the prediction check (preprocessor unavailable).")
        return

    try:
        import numpy as np
        cleaned = preprocessor.clean_text(SAMPLE_TEXT)
        tensor = tf.reshape(tf.constant([cleaned], dtype=tf.string), (-1, 1))
        probability = float(np.asarray(model.predict(tensor, verbose=0)).reshape(-1)[0])
    except Exception as exc:
        report.fail(f"Prediction failed - {exc}")
        return

    if not 0.0 <= probability <= 1.0:
        report.fail(f"Model returned {probability}, which is not a probability. "
                    "The final layer should be a sigmoid.")
        return

    verdict = "FAKE" if probability >= 0.5 else "REAL"
    report.ok(f"Prediction works - P(fake)={probability:.4f} -> {verdict}")

    # A batch call is what LIME does; make sure it is wired correctly too.
    try:
        batch = tf.reshape(tf.constant([cleaned, cleaned, ""], dtype=tf.string), (-1, 1))
        out = np.asarray(model.predict(batch, verbose=0)).reshape(-1)
        if out.shape[0] != 3:
            report.fail(f"Batch prediction returned {out.shape[0]} rows, expected 3.")
        else:
            report.ok("Batch prediction works (required by the LIME explainer).")
    except Exception as exc:
        report.fail(f"Batch prediction failed - {exc}")


def check_extras(report):
    report.section("6. Optional assets")

    for split in ("0.2_split_data.joblib", "0.3_split_data.joblib"):
        if os.path.exists(split):
            report.ok(f"{split} present (needed only for training/evaluation).")
        else:
            report.warn(f"{split} missing - training scripts will not run.")

    if os.path.exists("fakenews5.jpg"):
        report.ok("Background image present.")
    else:
        report.warn("fakenews5.jpg missing - the app falls back to a gradient.")

    try:
        import speech_recognition as sr
        count = len(sr.Microphone.list_microphone_names())
        if count:
            report.ok(f"{count} microphone device(s) detected.")
        else:
            report.warn("No microphone detected - the Record button is disabled.")
    except Exception:
        report.warn("Microphone check skipped (SpeechRecognition/PyAudio missing).")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default=os.path.join('model', 'best_cnn_rnn_model.keras'))
    args = parser.parse_args()

    # Resolve paths against this file so the check works from any directory.
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    print("=" * 64)
    print(" Fake News Detection - setup verification")
    print("=" * 64)

    report = Reporter()
    check_python(report)
    check_packages(report)
    check_nltk(report)
    preprocessor = check_preprocessing(report)
    check_model(report, args.model, preprocessor)
    check_extras(report)

    print("\n" + "=" * 64)
    if report.failures:
        print(f"{FAIL} {len(report.failures)} blocking problem(s):")
        for message in report.failures:
            print(f"       - {message}")
        print("\nFix the above, then re-run this script.")
        return 1

    if report.warnings:
        print(f"{PASS} Ready to run, with {len(report.warnings)} warning(s):")
        for message in report.warnings:
            print(f"       - {message}")
    else:
        print(f"{PASS} Everything checks out.")

    print("\nStart the app with:  streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

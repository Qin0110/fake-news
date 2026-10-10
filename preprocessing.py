"""Text cleaning shared by training (Preprocessing) and serving (InferencePreprocessor).

Both use one pipeline so train and inference can never drift apart.
"""

import os
import re

import nltk
import contractions
import ftfy
from langdetect import detect, LangDetectException, DetectorFactory
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

DetectorFactory.seed = 42  # langdetect is non-deterministic otherwise

# NLTK >= 3.9 renamed the tagger to ..._eng; older releases only have the old
# name. Request both, tolerate whichever is missing.
_NLTK_RESOURCES = (
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
)

_NLTK_READY = False


def _resource_present(lookup_path):
    # wordnet/omw-1.4 stay zipped: the bare path raises even when usable, so
    # checking it alone would re-download them on every start.
    for candidate in (lookup_path, f"{lookup_path}.zip"):
        try:
            nltk.data.find(candidate)
            return True
        except LookupError:
            continue
        except Exception:
            continue
    return False


def ensure_nltk_data(quiet=True, force=False):
    """Download missing corpora once per process; returns ids that failed."""
    global _NLTK_READY
    if _NLTK_READY and not force:
        return []

    unavailable = []
    for lookup_path, package_id in _NLTK_RESOURCES:
        if _resource_present(lookup_path):
            continue
        try:
            if not nltk.download(package_id, quiet=quiet):
                unavailable.append(package_id)
        except Exception:
            unavailable.append(package_id)

    _NLTK_READY = True
    return unavailable


# Kept despite being stop words: they carry the polarity of a claim.
NEGATION_WORDS = frozenset({
    'no', 'not', 'nor', 'neither', 'never', 'none', 'nobody', 'nowhere', 'nothing',
    "don't", "aren't", "isn't", "wasn't", "weren't",
    "won't", "wouldn't", "didn't", "doesn't",
    "hasn't", "haven't", "hadn't", "shouldn't", "can't", "couldn't", "mustn't",
})

_URL_RE = re.compile(r'https?://\S+|www\.\S+')
_HTML_TAG_RE = re.compile(r'<.*?>')
_FALLBACK_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


class _TextCleaner:
    """How raw article text becomes tokens."""

    def __init__(self):
        ensure_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english')) - NEGATION_WORDS
        self._lemma_cache = {}

    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def fix_encoding_errors(self, text):
        try:
            return ftfy.fix_text(text)
        except Exception:
            return text

    def is_english(self, text, min_chars=0):
        """min_chars gives short inputs the benefit of the doubt; 0 is strict."""
        text = str(text)
        if len(text.strip()) < min_chars:
            return True
        try:
            return detect(text) == 'en'
        except LangDetectException:
            return False
        except Exception:
            return False

    def lemma_variants(self, word):
        """Lemmas across all four POS. Lets the UI map 'run' back onto 'running'."""
        key = word.lower()
        cached = self._lemma_cache.get(key)
        if cached is None:
            cached = {key}
            for pos in (wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV):
                try:
                    cached.add(self.lemmatizer.lemmatize(key, pos=pos))
                except Exception:
                    pass
            self._lemma_cache[key] = cached
        return cached

    def _tokenize(self, text):
        try:
            return word_tokenize(text)
        except LookupError:
            return _FALLBACK_TOKEN_RE.findall(text)  # punkt missing

    def _pos_tag(self, tokens):
        try:
            return nltk.pos_tag(tokens)
        except LookupError:
            return [(token, 'NN') for token in tokens]  # tagger missing

    def clean_tokens(self, text):
        """Raw text -> cleaned lemmatised tokens.

        Do not reorder: the deployed model was trained on exactly this sequence.
        """
        text = str(text) if text is not None else ""
        text = text.strip()
        if not text:
            return []

        text = self.fix_encoding_errors(text)

        try:
            text = contractions.fix(text)
        except Exception:
            pass

        text = text.lower()
        text = _URL_RE.sub('', text)
        text = _HTML_TAG_RE.sub('', text)

        tokens = self._tokenize(text)
        pos_tags = self._pos_tag(tokens)

        processed_tokens = []
        for word, tag in pos_tags:
            if word.isalpha() and word.isascii() and word not in self.stop_words:
                wn_tag = self.get_wordnet_pos(tag)
                processed_tokens.append(self.lemmatizer.lemmatize(word, pos=wn_tag))
        return processed_tokens


class Preprocessing(_TextCleaner):
    """Builds the model-ready dataset from the raw fake_news.csv corpus."""

    REQUIRED_COLUMNS = ('title', 'text', 'label')

    def __init__(self, csv_path="fake_news.csv"):
        # Imported here, not at module level, so the app does not load pandas.
        import pandas as pd

        super().__init__()

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Could not find '{csv_path}'. The raw corpus is git-ignored; "
                "place fake_news.csv next to preprocessing.py (columns: "
                "id, title, author, text, label) before rebuilding the splits."
            )

        self.df = pd.read_csv(csv_path)

        missing = [c for c in self.REQUIRED_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"'{csv_path}' is missing required column(s): {', '.join(missing)}"
            )

        self.df.dropna(subset=list(self.REQUIRED_COLUMNS), inplace=True)
        self.df.drop_duplicates(inplace=True)
        self.df.drop(['author', 'id'], axis=1, inplace=True, errors='ignore')

        initial_rows = len(self.df)
        print(f"Total rows before language filtering: {initial_rows}")

        self.df = self.df[self.df['text'].apply(self.is_english)]

        final_rows = len(self.df)
        print(f"Total rows after language filtering: {final_rows}")
        print(f"Removed {initial_rows - final_rows} non-English rows.")

        if final_rows == 0:
            raise ValueError("No English rows survived filtering -- check the corpus.")

        self.df['combined_text'] = self.df['title'].astype(str) + " " + self.df['text'].astype(str)

        print("Cleaning documents (this takes a few minutes)...")
        self.df['processed_text_list'] = self.df['combined_text'].apply(self.clean_tokens)
        self.df['clean_text_string'] = self.df['processed_text_list'].apply(' '.join)

        empty = (self.df['clean_text_string'].str.strip() == "").sum()
        if empty:
            print(f"Dropping {empty} document(s) that cleaned down to an empty string.")
            self.df = self.df[self.df['clean_text_string'].str.strip() != ""]

        self.text_data = self.df['clean_text_string'].values

        self.X = self.text_data
        self.y = self.df['label'].astype(int).values

        print(f"Dataset ready: {len(self.X)} documents, "
              f"{int(self.y.sum())} fake / {int((self.y == 0).sum())} real.")

    def clean_text(self, text):
        """Returns a list of tokens."""
        return self.clean_tokens(text)

    def split_data(self, test_size, random_state=42):
        """Stratified split. test_size is halved into validation and test,
        so 0.2 gives 80/10/10."""
        from sklearn.model_selection import train_test_split

        X_train, X_test_validate, y_train, y_test_validate = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y
        )

        X_test, X_validate, y_test, y_validate = train_test_split(
            X_test_validate,
            y_test_validate,
            test_size=0.5,
            random_state=random_state,
            stratify=y_test_validate
        )
        return X_train, X_validate, X_test, y_train, y_validate, y_test

    def dump_variable(self, test_size, out_dir="."):
        """Writes <out_dir>/<test_size>_split_data.joblib."""
        import joblib

        X_train, X_validate, X_test, y_train, y_validate, y_test = self.split_data(
            test_size=test_size
        )
        data_package = (X_train, X_validate, X_test, y_train, y_validate, y_test)

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{test_size}_split_data.joblib')
        joblib.dump(data_package, out_path)

        print(f"Saved {out_path} -- train={len(X_train)}, "
              f"validate={len(X_validate)}, test={len(X_test)}")
        return out_path


class InferencePreprocessor(_TextCleaner):
    """Serving-time cleaner used by the Streamlit app."""

    def clean_text(self, text):
        """Returns a string, which is what the TextVectorization layer expects."""
        return " ".join(self.clean_tokens(text))


if __name__ == "__main__":
    preprocessor = Preprocessing()
    preprocessor.dump_variable(test_size=0.2)
    preprocessor.dump_variable(test_size=0.3)

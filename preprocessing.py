import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

import nltk
import contractions
import ftfy
from langdetect import detect, LangDetectException, DetectorFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

DetectorFactory.seed = 42

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('average_perceptron_tagger_eng')
nltk.download('omw-1.4')

class Preprocessing:
    def __init__(self):
        self.df = pd.read_csv("fake_news.csv")
        self.df.dropna(inplace=True)
        self.df.drop_duplicates(inplace=True)
        self.df.drop(['author', 'id'], axis=1, inplace=True)
        
        initial_rows = len(self.df)
        print(f"Total rows before language filtering: {initial_rows}")
        
        self.df = self.df[self.df['text'].apply(self.is_english)]
        
        final_rows = len(self.df)
        print(f"Total rows after language filtering: {final_rows}")
        print(f"Removed {initial_rows - final_rows} non-English rows.")

        self.lemmatizer = WordNetLemmatizer()
        
        negation_list = {
            'no', 'not', 'nor', 'neither', 'never', 'none', 'nobody', 'nowhere', 'nothing',
            "don't", "aren't", "isn't", "wasn't", "weren't",
            "won't", "wouldn't", "didn't", "doesn't",
            "hasn't", "haven't", "hadn't", "shouldn't", "can't", "couldn't", "mustn't"
        }
        self.stop_words = set(stopwords.words('english')) - negation_list
        
        self.df['combined_text'] = self.df['title'] + " " + self.df['text']
        
        self.df['processed_text_list'] = self.df['combined_text'].apply(self.clean_text)
        self.df['clean_text_string'] = self.df['processed_text_list'].apply(lambda x: ' '.join(x))
        
        self.text_data = self.df['clean_text_string'].values
        
        #region X and y variables here
        self.X = self.text_data
        self.y = self.df['label'].values

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
        return ftfy.fix_text(text)

    def is_english(self, text):

        try:
            return detect(str(text)) == 'en'
        except LangDetectException:
            return False

    def clean_text(self, text):
        text = str(text) if text is not None else ""
        
        # If text is empty Or whitespace, return empty list (Ensure input is a string and not a number or NaN)
        if not text.strip():
            return []

        text = self.fix_encoding_errors(text)

        try:
            text = contractions.fix(text)
        except IndexError:
            pass # Keep text as is if library fails (validation)
        
        text = str(text).lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        
        tokens = word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        processed_tokens = []
        
        for word, tag in pos_tags:
            if word.isalpha() and word.isascii() and  word not in self.stop_words:
                wn_tag = self.get_wordnet_pos(tag)
                lemma = self.lemmatizer.lemmatize(word.lower(), pos=wn_tag)
                processed_tokens.append(lemma)
        
        return processed_tokens
    
    def split_data(self, test_size, random_state=42):
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
    
    def dump_variable(self, test_size):
        X_train, X_validate, X_test, y_train, y_validate, y_test = self.split_data(test_size=test_size)
        data_package = (X_train, X_validate, X_test, y_train, y_validate, y_test)
        joblib.dump(data_package, f'{test_size}_split_data.joblib')



class InferencePreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

        negation_list = {
            'no', 'not', 'nor', 'neither', 'never', 'none', 'nobody', 'nowhere', 'nothing',
            "don't", "aren't", "isn't", "wasn't", "weren't",
            "won't", "wouldn't", "didn't", "doesn't",
            "hasn't", "haven't", "hadn't", "shouldn't", "can't", "couldn't", "mustn't"
        }
        self.stop_words = set(stopwords.words('english')) - negation_list

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

    def is_english(self, text):
        try:
            return detect(str(text)) == 'en'
        except LangDetectException:
            return False
        
    def fix_encoding_errors(self, text):
        return ftfy.fix_text(text)
    
    def clean_text(self, text):
        text = str(text).strip()
        
        if not text:
            return ""

        text = self.fix_encoding_errors(text)

        try:
            text = contractions.fix(text)
        except IndexError:
            pass # Keep text as is if library fails (validation)

        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        
        tokens = word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        processed_tokens = []

        for word, tag in pos_tags:
            if word.isalpha() and word.isascii() and word not in self.stop_words:
                wn_tag = self.get_wordnet_pos(tag)
                lemma = self.lemmatizer.lemmatize(word.lower(), pos=wn_tag)
                processed_tokens.append(lemma)
        return " ".join(processed_tokens)

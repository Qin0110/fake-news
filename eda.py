import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer


X_train, X_validate, X_test, y_train, y_validate, y_test = joblib.load('0.3_split_data.joblib')

X_all = np.concatenate((X_train, X_validate, X_test))
y_all = np.concatenate((y_train, y_validate, y_test))


df = pd.DataFrame({
    'text': X_all,
    'label': y_all
})

####################  CLEANED 
####################    OR                (Uncomment either one to use)
#################### UNCLEANED 

# df = pd.read_csv("fake_news.csv")

#  # Drop rows with missing text so itt doesn't crash
# df.dropna(subset=['text', 'label'], inplace=True)

def plot_distribution(df):
    counts = df['label'].value_counts() 
    counts.plot(kind='bar')

    plt.xlabel('label')
    plt.ylabel('Count')
    plt.title('Distribution of Categories')
    plt.tight_layout()
    plt.show()



def plot_top_ngrams(corpus, n=2, title="Top N-grams", n_top=20):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    
    bag_of_words = vec.transform(corpus)
    
    sum_of_words = bag_of_words.sum(axis=0) 
    
    # (Word, Frequency)
    words_freq = [(word, sum_of_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:n_top]
    
    df_ngram = pd.DataFrame(words_freq, columns=['Ngram', 'Count'])
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Count', y='Ngram', data=df_ngram, palette='viridis', hue='Ngram', legend=False)
    plt.title(title)
    plt.xlabel('Frequency')
    plt.ylabel('Phrase')
    plt.tight_layout()
    plt.savefig(f'{n}gram_analysis_{title.replace(" ", "_")}.png')
    plt.show()

if __name__ == "__main__":
    print("Plotting Class Distribution...")
    plot_distribution(df)

    real_text = df[df['label'] == 0]['text']
    fake_text = df[df['label'] == 1]['text']

    print("Generating Real News Bigrams...")
    plot_top_ngrams(real_text, n=2, title="Top 20 Bigrams - REAL News")
    
    print("Generating Fake News Bigrams...")
    plot_top_ngrams(fake_text, n=2, title="Top 20 Bigrams - FAKE News")
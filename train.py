import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_sentiment_pipeline():
    base_path = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_path, "data", "imdb_reviews_sample.csv")
    df = pd.read_csv(csv_path)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    X = vectorizer.fit_transform(df['review'])
    y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    
    model = LogisticRegression()
    model.fit(X, y)
    return vectorizer, model

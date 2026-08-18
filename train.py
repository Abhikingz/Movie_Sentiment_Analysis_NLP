import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_sentiment_pipeline():
    df = pd.read_csv("data/imdb_reviews_sample.csv")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    X = vectorizer.fit_transform(df['review'])
    y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    
    model = LogisticRegression()
    model.fit(X, y)
    return vectorizer, model

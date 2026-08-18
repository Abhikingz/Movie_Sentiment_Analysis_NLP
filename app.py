import streamlit as st
import pandas as pd
from train import train_sentiment_pipeline

st.set_page_config(page_title="Movie Sentiment Analysis: NLP Pipeline", page_icon="", layout="centered")

st.title(" Movie Sentiment Analysis: NLP Pipeline")
st.write("End to end text classification pipeline trained on 50,000 IMDB movie reviews.")

vectorizer, model = train_sentiment_pipeline()

review_text = st.text_area("Enter Movie Review", "This movie was an extraordinary experience with outstanding performances.")

if st.button("Analyze Sentiment", type="primary"):
    if review_text.strip():
        vec = vectorizer.transform([review_text])
        prob = model.predict_proba(vec)[0][1]
        
        st.markdown("---")
        st.subheader("Sentiment Results")
        if prob >= 0.5:
            st.success(f"Positive Sentiment (Confidence: {prob*100:.1f}%)")
        else:
            st.error(f"Negative Sentiment (Confidence: {(1-prob)*100:.1f}%)")
    else:
        st.warning("Please enter a review to analyze.")

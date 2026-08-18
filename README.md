# Movie Sentiment Analysis: NLP Pipeline

An end to end natural language processing pipeline for movie review sentiment classification trained on 50,000 IMDB reviews. The architecture benchmarks classical ML classifiers alongside a BiLSTM neural network, achieving 89% accuracy.

## Project Documentation & Technical Report

* **Download Technical PDF Report**: [Technical_Report_Movie_Sentiment_Analysis_NLP.pdf](Technical_Report_Movie_Sentiment_Analysis_NLP.pdf)
* **Primary Dataset**: [Kaggle IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* **Local Sample Data**: Included in `data/imdb_reviews_sample.csv`

## Key Features

* Complete text preprocessing pipeline including tokenization, lemmatization, and stop word removal
* TF IDF and Word2Vec feature extraction techniques
* BiLSTM neural network trained in PyTorch
* Streamlit web interface for real time review sentiment scoring

## Quickstart Guide

```bash
pip install -r requirements.txt
streamlit run app.py
```

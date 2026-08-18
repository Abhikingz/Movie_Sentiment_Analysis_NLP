# Movie Sentiment Analysis: NLP Pipeline

An end to end natural language processing pipeline for movie review sentiment classification trained on 50,000 IMDB reviews. The architecture benchmarks classical ML classifiers alongside a BiLSTM neural network, achieving 89% accuracy.

## Dataset & Resources

* **Primary Dataset**: [Kaggle IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* **Local Sample Data**: Included in `data/imdb_reviews_sample.csv`
* **Benchmark Accuracy**: 89% accuracy with BiLSTM, outperforming classical baselines by 4.5 percentage points

## Key Features

* Complete text preprocessing pipeline including tokenization, lemmatization, and stop word removal
* TF IDF and Word2Vec feature extraction techniques
* BiLSTM neural network trained in PyTorch
* Streamlit web interface for real time review sentiment scoring

## Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Application
```bash
streamlit run app.py
```

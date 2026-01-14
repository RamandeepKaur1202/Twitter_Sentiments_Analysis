# Twitter Sentiment Analysis using Machine Learning

## Project Overview

This project focuses on performing **sentiment analysis on Twitter data** using **Natural Language Processing (NLP)** and **Machine Learning** techniques. The goal is to classify tweets as **Positive** or **Negative** based on their textual content.

Due to the informal and noisy nature of Twitter data, effective preprocessing and feature extraction techniques were applied to achieve reliable sentiment classification.

> Dataset used: Sentiment140 (not uploaded to GitHub due to large size).



## Objectives

- Preprocess raw Twitter text data
- Convert text into numerical features using TF-IDF
- Train and evaluate machine learning models
- Compare model performance and select the best model
- Predict sentiment for unseen tweets



## Models Used

- **Logistic Regression (Final Model)**
- **Linear Support Vector Machine (SVM)**



## Technologies & Libraries

- Python
- Pandas, NumPy
- NLTK
- Scikit-learn
- TF-IDF Vectorizer



## Text Preprocessing Steps

- Convert text to lowercase
- Remove URLs, mentions, and special characters
- Remove stopwords
- Lemmatization
- Removal of unused columns


##  Feature Extraction

- **TF-IDF (Term Frequency–Inverse Document Frequency)**
- Unigrams and bigrams
- Feature size control to reduce noise



## Model Performance

| Model               | Accuracy   | F1-Score |
| ------------------- | ---------- | -------- |
| Logistic Regression | **78.83%** | **0.79** |
| Linear SVM          | 78.72%     | 0.79     |



## Best Model Selection

**Logistic Regression** was selected as the final model due to:

- Slightly higher accuracy
- Balanced precision and recall
- Faster training time
- Better interpretability



## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix



## Final Result

The final Logistic Regression model achieved an accuracy of **78.83%**, demonstrating strong generalization performance on unseen Twitter data. The model effectively classifies tweet sentiment without bias toward any particular class.



## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/RamandeepKaur1202/Twitter_Sentiments_Analysis.git
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Twitter_Sentiments_Analysis.ipynb
   ```
3. Run all cells sequentially



## Future Scope

- Sarcasm detection
- Real-time sentiment analysis using Twitter API
- Deep learning models (LSTM, BERT)
- Aspect-based sentiment analysis



##  Author

**Ramandeep Kaur**\
B.Tech Computer Science Engineering





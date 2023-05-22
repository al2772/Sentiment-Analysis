# Sentiment-Analysis

Uses the Natural Language Toolkit (NLTK) library for text processing and scikit-learn library for machine learning. It assumes you already have a labeled dataset in a CSV file where each row contains a text sample and its corresponding sentiment label.

- The code starts by importing the necessary libraries and modules, including pandas for data handling, NLTK for text processing, and scikit-learn for machine learning.
- The data is loaded from a CSV file into a pandas DataFrame, separating the text samples (text_data) and the corresponding sentiment labels (labels).
- The data is split into training and testing sets using the train_test_split function from scikit-learn.
- It uses the TfidfVectorizer from scikit-learn to convert the text data into TF-IDF features for both the training and testing sets.
- The SVM (Support Vector Machine) model is chosen and trained on the training data using the TF-IDF features.
- The trained model is used to make predictions on the testing set, and the accuracy of the model is calculated.
- Finally, the code prompts the user to enter a sentence for sentiment analysis. The sentence is transformed into TF-IDF features using the same vectorizer, and the model predicts the sentiment of the input.

Make sure to replace 'sentiment_data.csv' with the path to your labeled dataset file. The sentiment labels in the dataset should be encoded as 0 (negative), 1 (neutral), and 2 (positive).

Remember to install the required libraries (NLTK and scikit-learn) using pip install nltk scikit-learn before running the code.

This code provides a clean and readable implementation of the sentiment analysis project, covering data preprocessing, feature extraction, model training, evaluation, and sentiment analysis on user input.

A sample of 'sentiment_data.csv' is given.

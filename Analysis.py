import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn import metrics

# Step 1: Data loading and preprocessing
data = pd.read_csv('sentiment_data.csv')
text_data = data['text']
labels = data['sentiment']

# Step 2: Splitting the data into training and testing sets
text_train, text_test, labels_train, labels_test = train_test_split(text_data, labels, test_size=0.2, random_state=42)

# Step 3: Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
features_train = vectorizer.fit_transform(text_train)
features_test = vectorizer.transform(text_test)

# Step 4: Model training (Support Vector Machine)
model = svm.SVC(kernel='linear')
model.fit(features_train, labels_train)

# Step 5: Model evaluation
predictions = model.predict(features_test)
accuracy = metrics.accuracy_score(labels_test, predictions)
print("Accuracy:", accuracy)

# Step 6: Sentiment analysis on user input
user_input = input("Enter a sentence for sentiment analysis: ")
user_input_features = vectorizer.transform([user_input])
user_input_sentiment = model.predict(user_input_features)

if user_input_sentiment[0] == 0:
    print("Negative sentiment.")
elif user_input_sentiment[0] == 1:
    print("Neutral sentiment.")
else:
    print("Positive sentiment.")

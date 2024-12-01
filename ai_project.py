import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree
import numpy as np
import pandas as pd

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Example labeled dataset for training (replace with real datasets like LIAR)
data = {
    "text": [
        "Breaking news: Aliens have landed on Earth and are negotiating with world leaders!",
        "Scientists discover a new species of frog in the Amazon rainforest.",
        "Shocking: Vaccines cause more harm than good, experts claim.",
        "Local hero saves child from burning building.",
    ],
    "label": [1, 0, 1, 0],  # 1 = Fake, 0 = Real
}
df = pd.DataFrame(data)

# Feature Extraction Functions
def extract_features(sentence):
    features = {}

    # Tokenization
    tokens = nltk.word_tokenize(sentence)
    features["num_tokens"] = len(tokens)

    # Sentiment Analysis
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(sentence)
    features["sentiment_compound"] = sentiment["compound"]
    features["sentiment_neg"] = sentiment["neg"]
    features["sentiment_pos"] = sentiment["pos"]
    features["sentiment_neu"] = sentiment["neu"]

    # Named Entity Recognition
    named_entities = ne_chunk(pos_tag(tokens))
    entities = [chunk for chunk in named_entities if isinstance(chunk, Tree)]
    features["num_named_entities"] = len(entities)

    # Readability Metrics
    features["avg_word_length"] = np.mean([len(word) for word in tokens if word.isalpha()])
    features["sentence_length"] = len(sentence)

    # Exaggeration Detection
    exaggeration_keywords = ["breaking", "shocking", "amazing", "unbelievable"]
    features["exaggeration_count"] = sum(1 for word in tokens if word.lower() in exaggeration_keywords)

    return features

# Create Feature Matrix and Labels
features = pd.DataFrame([extract_features(text) for text in df["text"]])
labels = df["label"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train Classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Evaluate Model
y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Real-Time Prediction
def predict_fake_news(sentence):
    input_features = pd.DataFrame([extract_features(sentence)])
    prediction = classifier.predict(input_features)[0]
    probability = classifier.predict_proba(input_features)[0][1]  # Probability of being fake
    return prediction, probability

# Input Sentence
input_sentence = input("Enter a sentence to analyze: ")

# Predict and Output Result
prediction, probability = predict_fake_news(input_sentence)
print("\nPrediction:", "Fake" if prediction == 1 else "Real")
print("Fake News Probability:", probability)

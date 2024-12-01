import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

# Example labeled dataset for training (replace with real datasets like LIAR or FakeNewsNet)
data = {
    "text": [
        # Fake News
        "Breaking news: Aliens have landed on Earth and are negotiating with world leaders!",
        "Shocking: Vaccines cause more harm than good, experts claim.",
        "Unbelievable: Man builds time machine and travels to the year 3023.",
        "Scientists confirm that eating only chocolate leads to a longer life!",
        "Astrologer predicts the end of the world next week!",
        "Exclusive: Dinosaurs are being cloned on a secret island in the Pacific.",
        "Experts reveal that the moon landing was faked in a Hollywood studio.",
        "Study shows that drinking soda daily increases intelligence by 50%.",
        "Breaking: A new element discovered on Mars can grant eternal youth.",
        "Mystery: Thousands of UFOs spotted hovering over major cities worldwide.",

        # Real News
        "Scientists discover a new species of frog in the Amazon rainforest.",
        "Local hero saves child from burning building.",
        "NASA launches a new telescope to study distant galaxies.",
        "United Nations announces new climate change initiative to reduce emissions.",
        "Researchers find evidence of water on the surface of Mars.",
        "WHO declares a new strategy to combat global pandemics.",
        "A local bakery donates 10,000 loaves of bread to food banks.",
        "Economists discuss the impact of inflation on global markets.",
        "The president addresses the nation regarding recent policy changes.",
        "Archaeologists uncover an ancient city buried beneath the Sahara Desert.",
    ],
    "label": [
        # Labels for Fake News
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        
        # Labels for Real News
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],  # 1 = Fake, 0 = Real
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

    # Named Entity Recognition with Entity Type Analysis
    named_entities = ne_chunk(pos_tag(tokens))
    entity_types = [chunk.label() for chunk in named_entities if isinstance(chunk, Tree)]
    features["num_named_entities"] = len(entity_types)
    features["num_organizations"] = entity_types.count("ORGANIZATION")
    features["num_locations"] = entity_types.count("GPE")  # Geopolitical entities
    features["num_persons"] = entity_types.count("PERSON")

    # Readability Metrics
    features["avg_word_length"] = np.mean([len(word) for word in tokens if word.isalpha()])
    features["sentence_length"] = len(sentence)

    # Exaggeration Detection with Weighted Keywords
    exaggeration_keywords = ["breaking", "shocking", "amazing", "unbelievable", "exclusive", "never seen before", "all diseases"]
    features["exaggeration_count"] = sum(1 for word in tokens if word.lower() in exaggeration_keywords)

    # Keyword Density for Real News (e.g., common real news terms)
    real_news_keywords = ["discovery", "research", "study", "initiative", "policy", "announcement"]
    features["real_keywords_count"] = sum(1 for word in tokens if word.lower() in real_news_keywords)

    return features

# Create Feature Matrix and Labels
features = pd.DataFrame([extract_features(text) for text in df["text"]])
labels = df["label"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train Classifier with Hyperparameter Tuning
classifier = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
classifier.fit(X_train, y_train)

# Evaluate Model
y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Real-Time Prediction Function
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

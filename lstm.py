import nltk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, Attention
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Ensure NLTK resources are downloaded
nltk.download('punkt')

# Expanded Dataset (Example)
data = {
    "text": [
        # Fake News
        "Breaking news: Aliens have landed on Earth and are negotiating with world leaders!",
        "Shocking: Vaccines cause more harm than good, experts claim.",
        "Unbelievable: Man builds time machine and travels to the year 3023.",
        "Scientists confirm that eating only chocolate leads to a longer life!",
        "Astrologer predicts the end of the world next week!",
        "Dinosaurs are being cloned on a secret island in the Pacific.",
        "Experts reveal that the moon landing was faked in a Hollywood studio.",
        "Study shows that drinking soda daily increases intelligence by 50%.",
        # Real News
        "NASA launches a new telescope to study distant galaxies.",
        "Researchers find evidence of water on the surface of Mars.",
        "United Nations announces a global initiative to combat climate change.",
        "Archaeologists uncover an ancient city buried beneath the Sahara Desert.",
        "WHO declares new guidelines to combat global pandemics.",
        "Researchers discover a new species of bird in the Amazon rainforest.",
        "Local bakery donates 10,000 loaves of bread to food banks.",
    ],
    "label": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # 1 = Fake, 0 = Real
}

df = pd.DataFrame(data)

# Preprocessing
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    return " ".join(tokens)

df["text"] = df["text"].apply(preprocess_text)

# Tokenization and Padding
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(df["text"])
vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(df["text"])
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df["label"], test_size=0.2, random_state=42)

# Build the Enhanced LSTM Model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.5),
    LSTM(64),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=16,
    verbose=2
)

# Evaluate the Model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Real-Time Prediction
def predict_fake_news_lstm(text):
    preprocessed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding="post")
    prediction = model.predict(padded_sequence)[0][0]
    label = "Fake" if prediction > 0.5 else "Real"
    return label, prediction

# Test Real-Time Predictions
while True:
    input_text = input("Enter a sentence to classify (or type 'exit' to quit): ")
    if input_text.lower() == "exit":
        break
    label, prob = predict_fake_news_lstm(input_text)
    print(f"Prediction: {label} (Probability of being Fake: {prob:.2f})")

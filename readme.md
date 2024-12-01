# AI Final Project: Automated Fake News Detector for Social Media

* Group 16
* Ilayda Koca

## **Introduction**

### **Motivation**

The rise of misinformation on social media presents an intriguing and challenging problem for algorithm design. Fake news not only spreads faster than truthful content but also exploits linguistic and contextual nuances, making it a compelling subject for advanced AI applications. This project is motivated by the opportunity to explore and implement state-of-the-art algorithms to address a real-world problem within a controlled academic framework.

Building an Automated Fake News Detector provides a platform to apply and integrate a variety of machine learning and natural language processing (NLP) techniques. It challenges us to design and evaluate algorithms capable of distinguishing between fake and real news with high accuracy while maintaining transparency and interpretability.

This project also provides a valuable learning opportunity to:

- Experiment with feature engineering, decision tree-based models, recurrent neural networks (RNN/LSTM), and transformer-based architectures.
- Evaluate algorithms using established datasets such as LIAR and FakeNewsNet.
- Analyze and compare performance metrics to determine the most effective approach.
By addressing the technical and analytical aspects of this problem, this project allows us to deepen our understanding of AI methodologies while contributing a meaningful solution to an increasingly relevant societal issue.

### **Contributions**

For this project, I introduced multiple AI approaches to achieve accurate and transparent misinformation detection. The key contributions include:

1. **Natural Language Processing (NLP)**: Extracts linguistic features from posts using tokenization, sentiment analysis, and Named Entity Recognition (NER).
2. **Decision Tree Models**: Implements Random Forest classifiers to identify patterns and classify text based on extracted features.
3. **Neural Networks (RNN/LSTM)**: Captures contextual dependencies and temporal patterns within the text for nuanced fake news detection.
4. **Transformer-based Models (e.g., BERT/RoBERTa)**: Provides sophisticated context and semantic understanding for detecting subtle misinformation.

## **Solution Formulation**

### **AI Techniques Implemented**

This project uses a combination of AI techniques to tackle the problem of detecting fake news on social media. Each method contributes to different aspects of the solution, helping the system analyze text content, extract meaningful features, and make accurate predictions. Below, we describe the techniques we implemented, their roles in solving the problem, and why they were chosen.

---

### **1. Natural Language Processing (NLP)**

#### **Steps:**
1. **Tokenization**:
   - The text is split into individual words or tokens to analyze linguistic features like word counts and keyword frequencies.
   - Example:
     ```
     Input: "Breaking news: Aliens have landed on Earth!"
     Tokens: ["Breaking", "news", ":", "Aliens", "have", "landed", "on", "Earth", "!"]
     ```
   - **Role**: Helps identify structural patterns, such as text complexity, and locate keywords.

2. **Sentiment Analysis**:
   - We calculate the emotional tone of the text using scores for positive, negative, and neutral sentiment, as well as an overall "compound" score.
   - Fake news often exaggerates emotions, so strong sentiment is flagged as a potential indicator.
   - Example:
     ```
     Input: "Shocking discovery changes everything!"
     Sentiment Scores: {'neg': 0.2, 'neu': 0.4, 'pos': 0.4, 'compound': 0.7}
     ```
   - **Role**: Identifies exaggerated or emotionally charged language often seen in fake news.

3. **Named Entity Recognition (NER)**:
   - Proper nouns (e.g., organizations, locations, people) are extracted to assess credibility.
   - Example:
     ```
     Input: "NASA discovers water on Mars."
     Output: [('NASA', 'ORGANIZATION'), ('Mars', 'GPE')]
     ```
   - **Role**: Fake news often lacks references to credible entities or uses irrelevant ones, which NER helps identify.

#### **Why NLP?**
Natural Language Processing (NLP) is a fundamental part of this project because it enables the system to extract meaningful insights from raw text data, which is inherently unstructured and complex. Social media posts are highly variable in language style, length, and tone, and NLP provides the tools to handle these variations systematically.
It enables the system to:
- Extract patterns from text data.
- Represent unstructured information in a structured format.
- Provide interpretable insights for both users and machine learning models. 

---

### **2. Decision Tree Models (Random Forest)**

#### **Steps Implemented**
The Random Forest Classifier is used as a core algorithm to classify social media posts as either **fake** or **real** based on features extracted through Natural Language Processing (NLP). The steps for implementing this technique are as follows:

1. **Feature Extraction**:
   - NLP techniques are applied to transform raw text into structured features such as:
     - Sentiment scores (positive, negative, compound).
     - Number of tokens, named entities (e.g., persons, organizations).
     - Exaggeration keyword counts (e.g., "breaking," "shocking").
     - Real news keyword counts (e.g., "discovery," "research").
   - These features capture linguistic and structural characteristics that differentiate fake news from real news.

2. **Train-Test Split**:
   - The extracted features are divided into **training** and **test datasets** (80% for training, 20% for testing) to ensure the model is evaluated on unseen data.

3. **Training the Random Forest Classifier**:
   - The Random Forest Classifier creates multiple decision trees, each trained on a random subset of features and data.
   - Each decision tree makes a prediction, and the final classification is determined by majority voting across all trees.
   - Example Rule Learned by a Tree:
     ```
     IF sentiment_compound > 0.7 AND num_named_entities = 0 THEN classify as Fake.
     ELSE classify as Real.
     ```
   - Hyperparameters used:
     - **n_estimators=200**: Builds 200 decision trees.
     - **max_depth=10**: Limits tree depth to prevent overfitting.

4. **Evaluation**:
   - The model’s performance is evaluated on the test dataset using the following metrics:
     - **Accuracy**: Proportion of correctly classified posts.
     - **Precision**: Ability to identify fake news accurately.
     - **Recall**: Ability to detect all fake news instances.
     - **F1 Score**: Harmonic mean of precision and recall.
   - A confusion matrix is also generated to analyze false positives and false negatives.

5. **Real-Time Prediction**:
   - For a new input sentence, the same feature extraction process is applied.
   - The Random Forest model predicts:
     - **Binary Output**: `1 (Fake)` or `0 (Real)`.
     - **Probability**: Confidence level of the prediction (e.g., 85% likely to be fake).

---

#### **How It Solves the Problem**
- **Addresses High Variability**:
  - The ensemble nature of Random Forest handles the variability in linguistic features across different posts by averaging predictions from multiple trees.
- **Reduces Overfitting**:
  - By limiting tree depth (`max_depth=10`) and using random feature subsets, the model generalizes well to unseen data.
- **Handles Imbalanced Features**:
  - Random Forest is robust to features with varying scales and distributions, such as the number of tokens versus sentiment scores.

---

#### **Pseudo Code**
```python
# Train a Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Feature extraction and split
features = pd.DataFrame([extract_features(text) for text in df["text"]])
labels = df["label"]
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train the model
classifier = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
classifier.fit(X_train, y_train)

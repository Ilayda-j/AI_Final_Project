# AI Final Project: Automated Fake News Detector for Social Media

## Introduction

## Algorithms

### Algorithm 1: NLP High-Level Explanation of the Algorithm

This algorithm decides whether a sentence is **fake** or **real** by analyzing specific linguistic and structural features in the text, processing those features into numerical values, and using a machine learning classifier to make predictions based on the processed data.

---

#### Key Steps in the Algorithm

##### 1. Feature Extraction
The algorithm breaks down each sentence into quantifiable attributes, or "features," which provide insights into the text's characteristics. These features include:
- **Tokenization**: Counts the number of words in the sentence.
- **Sentiment Analysis**: Measures the emotional tone (positive, negative, neutral, or compound sentiment).
- **Named Entity Recognition (NER)**:
  - Detects proper nouns (e.g., organizations, locations, people).
  - Counts how many of these entities appear and categorizes them into types (e.g., "PERSON," "GPE" for geopolitical entities).
- **Readability Metrics**:
  - Average word length.
  - Sentence length.
- **Exaggeration Detection**:
  - Looks for keywords often used in sensational fake news (e.g., "breaking," "shocking," "unbelievable").
- **Real News Keywords**:
  - Counts words associated with credible or factual news, like "study," "initiative," "policy," or "discovery."

---

##### 2. Training the Machine Learning Model
- The extracted features are stored in a **feature matrix**, and each sentence is labeled as **1 (Fake)** or **0 (Real)**.
- The dataset is split into **training** and **test sets**.
- A **Random Forest Classifier** is trained on the features of the training set to learn patterns that distinguish fake news from real news.
  - Random Forest is an ensemble model that builds multiple decision trees and combines their outputs for robust predictions.
  - It uses hyperparameters like `n_estimators=200` (number of trees) and `max_depth=10` (limits tree depth to avoid overfitting).

---

##### 3. Prediction
When a new sentence is input:
1. The sentence is processed using the same feature extraction steps as the training data.
2. The feature values are passed to the trained Random Forest model.
3. The model predicts:
   - **Binary Output**: `1 (Fake)` or `0 (Real)`.
   - **Probability**: Confidence level for the prediction, e.g., 80% likelihood of being fake.

---

#### How the Algorithm Differentiates Between Fake and Real News
##### Fake News:
- Higher **exaggeration keyword counts**.
- Fewer or irrelevant named entities (e.g., lack of organizations or locations).
- Strong emotional sentiment (positive or negative extremes).
- Simplistic language (shorter average word lengths or sentences).

##### Real News:
- Relevant named entities (e.g., "NASA," "United Nations").
- Keywords indicating credible reporting (e.g., "study," "policy").
- Balanced sentiment (neutral or compound sentiment scores close to zero).
- Complex and factual language.

---

#### Example Workflow
##### Input Sentence:
*"Breaking news: Dinosaurs have been cloned in a secret lab!"*

##### Feature Extraction:
- Tokens: `["Breaking", "news", "Dinosaurs", "have", "been", "cloned", "in", "a", "secret", "lab", "!"]`
  - Number of Tokens: 11
- Sentiment Compound: 0.7 (high positive sentiment)
- Named Entities: 0 (no valid persons, organizations, or locations)
- Exaggeration Count: 2 (`"Breaking"`, `"secret"`)
- Real Keywords Count: 0

##### Prediction:
- Features suggest sensationalism, lack of credible keywords, and missing named entities.
- **Output**: **Prediction: Fake**, **Probability: 0.85**

---

#### Strengths and Limitations

##### Strengths:
- Combines linguistic and semantic features to analyze text.
- Machine learning model generalizes patterns from training data.

##### Limitations:
- Relies on a predefined keyword list for exaggeration and real news, which may not cover all cases.
- Limited understanding of context (e.g., satire or ambiguous claims).
- Depends on the quality and diversity of the training dataset.


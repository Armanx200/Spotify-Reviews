import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load and preprocess dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['score'] = pd.to_numeric(data['score'], errors='coerce')
    
    # Create binary labels for sentiment classification
    data['sentiment'] = data['score'].apply(lambda x: 'positive' if x >= 4 else 'negative')

    # Drop missing content and null scores
    data = data.dropna(subset=['content', 'score'])
    
    return data[['content', 'sentiment']]

# Text preprocessing and model training
def train_model(data):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data['content'], data['sentiment'], test_size=0.2, random_state=42)
    
    # Convert text data to numerical features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    
    # Test the model and print accuracy
    y_pred = model.predict(X_test_tfidf)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    
    # Save the trained model and vectorizer
    joblib.dump(model, 'sentiment_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

data = load_data('Dataset.csv') 
train_model(data)

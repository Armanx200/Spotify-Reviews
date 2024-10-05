import joblib

# Load the trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Function to predict sentiment
def predict_sentiment(review):
    review_tfidf = vectorizer.transform([review])  # Transform the review using the vectorizer
    prediction = model.predict(review_tfidf)[0]  # Predict the sentiment
    return prediction

review = input("Enter a review: ")
sentiment = predict_sentiment(review)
print(f"The review is: {sentiment}")

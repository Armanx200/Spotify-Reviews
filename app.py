from predict_model import predict_sentiment


while True:
    print("\n--- Review Sentiment Classifier ---")
    review = input("Enter a review (or type 'exit' to quit): ")

    if review.lower() == 'exit':
        print("Goodbye!")
        break

    sentiment = predict_sentiment(review)
    print(f"The review is: {sentiment}\n")


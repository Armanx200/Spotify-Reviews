# 🎉 Sentiment Classifier App 💬

Welcome to the **Sentiment Classifier App**! This app takes in a review and predicts whether it's **positive** or **negative** using machine learning. 🚀

## 📂 Dataset

We used a dataset of **84,166 reviews** to train our model. The reviews are classified as **positive** or **negative** based on their content and score.

---

## 🔍 Features
- 🧠 **Trained on Logistic Regression** – Achieving **88.15% accuracy**!
- 🏋️ **TF-IDF Vectorizer** – Converts text to numerical features.
- ⚡ **Real-time Sentiment Prediction** – Input your review and get instant feedback.

---

## 🛠️ How It Works
### 1. **Train the Model**
```bash
python train_model.py
```
- Loads the dataset, preprocesses reviews, trains the classifier, and saves the model.

### 2. **Use the Model**
```bash
python app.py
```
- Launch the interactive app where you can input reviews and receive predictions.

---

## 🚀 Getting Started

### 🧩 Prerequisites
- Python 3.x
- Pandas, Scikit-Learn, Joblib

### 🔧 Installation
1. Clone the repo:
    ```bash
    git clone https://github.com/your-username/sentiment-classifier.git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Train the model:
    ```bash
    python train_model.py
    ```
4. Start the app:
    ```bash
    python app.py
    ```

---

## ⚙️ File Structure
```bash
.
├── train_model.py      # Model training script
├── predict_model.py    # Prediction module
├── app.py              # User interface script
├── reviews.csv         # Dataset file (84,166 reviews)
├── sentiment_model.pkl # Trained model
├── vectorizer.pkl      # TF-IDF vectorizer
└── README.md           # This file 😎
```

---

## 🤔 How to Use
1. Run the app with:
    ```bash
    python app.py
    ```
2. Enter a review, and the app will predict whether it's **positive** or **negative**!

```bash
Enter a review: "I love this app, it’s amazing!"
The review is: positive 👍
```

---

## 🧠 Model Performance
Achieved **88.15% accuracy** on the dataset with Logistic Regression!

---

## 🌟 Contributing
Feel free to fork this repo and contribute! We welcome improvements and new features.

---

## 📧 Contact
For any inquiries, reach out at:
- GitHub: [Arman Kianian](https://github.com/Armanx200)
- Email: Kianianarman1@gmail.com

---

Give this repo a ⭐ if you find it helpful! 😊

---

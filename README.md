# 🌐 Senticore: Sentiment Analysis Tool

Senticore is a sentiment analysis application that combines **machine learning** with a simple **Streamlit frontend**.  
It allows users to input text and instantly get predictions of sentiment (positive/negative/neutral).

---

## 📂 Project Structure

```plaintext
Senticore/
├── app.py
├── notebooks/
│   └── Senticore(1).ipynb
├── requirements.txt
├── senticore_model.pkl
├── senticore_vectorizer.pkl
├── .gitignore
└── README.md



---

## 🚀 Getting Started

## 1️⃣ Clone the Repository
```bash
git clone https://github.com/NeelaveniJonnada/Senticore.git
cd Senticore

## Create Virtual Environment (optional but recommended)
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate # On Mac/Linux


---
## 3️⃣ Install Dependencies
pip install -r requirements.txt


---

## 4️⃣ Run the App
streamlit run app.py

---

##🛠 Requirements

Main libraries:

streamlit

scikit-learn

joblib

nltk

pandas

numpy

(see requirements.txt for full list)
--

## 📊 Training

The notebook notebooks/Senticore.ipynb includes:

Data preprocessing (tokenization, stopwords, vectorization)

Model training (SVM, Logistic Regression, etc.)

Model evaluation (accuracy, confusion matrix)

Saving the trained model/vectorizer
---
## 🌟 Features

User-friendly Streamlit interface

Real-time sentiment prediction

Re-trainable with custom datasets

Extendable to other ML algorithms (Naive Bayes, Random Forest, etc.)

---

##📌 Future Improvements

Support for deep learning models

Better visualization of results

Multi-language sentiment analysis

Deployment on Streamlit Cloud / Hugging Face Spaces
---

##👩‍💻 Author
```plaintext
Neelaveni Jonnada
B.Tech Information Technology
Vishnu Institute of Technology

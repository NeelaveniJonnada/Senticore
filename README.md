# 🌐 Senticore: Sentiment Analysis Tool

Senticore is a sentiment analysis application that combines **machine learning** with a simple **Streamlit frontend**.  
It allows users to input text and instantly get predictions of sentiment (positive/negative/neutral).

---

## 📌 Features
- User-Friendly Interface
- Real-Time Sentiment Prediction
- Text Preprocessing Automation
- Multi-Model Support (during development)
- Visualization of Results
- ModelPersistence
- Extendable Architecture
- Export Functionality

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


````
---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/NeelaveniJonnada/Senticore.git
cd Senticore
````
### 2. Create Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate    # For Linux/macOS
venv\Scripts\activate       # For Windows

````
## 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt

````

## 4️⃣ Run the App
```bash
streamlit run app.py
```
Click on Local URL: http://localhost:8501

---

###  Output

<img width="1904" height="920" alt="Screenshot 2025-09-11 161909" src="https://github.com/user-attachments/assets/b79894f1-cac0-482c-b51a-35ee5a4e5cac" />

<img width="1915" height="927" alt="Screenshot 2025-09-11 162007" src="https://github.com/user-attachments/assets/5ae53307-28a2-409a-8b89-91eff99d2488" />

<img width="1899" height="914" alt="Screenshot 2025-09-11 162035" src="https://github.com/user-attachments/assets/d3b80fce-7f9e-4318-8618-bb127ff74532" />

<img width="1911" height="923" alt="Screenshot 2025-09-11 162146" src="https://github.com/user-attachments/assets/a4c913c9-a638-4773-9a78-7cabd63bae9e" />

## 📊 Training

The notebook notebooks/Senticore.ipynb includes:

Data preprocessing (tokenization, stopwords, vectorization)

Model training (SVM, Logistic Regression, etc.)

Model evaluation (accuracy, confusion matrix)

Saving the trained model/vectorizer


---

## 💡 Future Enhancements

* Support for deep learning models
* Better visualization of results
* Multi-language sentiment analysis

---

## 👩‍💻 Author

**Jonnada Neelaveni**
Vishnu Institute of Technology
[LinkedIn](https://www.linkedin.com/in/neelaveni-jonnada-901ba02ab/) 

---

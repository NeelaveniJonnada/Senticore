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

Home page
<img width="1904" height="920" alt="Screenshot 2025-09-11 161909" src="https://github.com/user-attachments/assets/b79894f1-cac0-482c-b51a-35ee5a4e5cac" />

Sign-Up page
<img width="1915" height="927" alt="Screenshot 2025-09-11 162007" src="https://github.com/user-attachments/assets/5ae53307-28a2-409a-8b89-91eff99d2488" />

Sign-In page
<img width="1899" height="914" alt="Screenshot 2025-09-11 162035" src="https://github.com/user-attachments/assets/d3b80fce-7f9e-4318-8618-bb127ff74532" />

Dashboard after logging-in
<img width="1911" height="923" alt="Screenshot 2025-09-11 162146" src="https://github.com/user-attachments/assets/a4c913c9-a638-4773-9a78-7cabd63bae9e" />

About page
<img width="1909" height="927" alt="Screenshot 2025-09-11 162235" src="https://github.com/user-attachments/assets/ae3529e0-1553-44d6-9234-e5847341d0f8" />

Qucik Guide page
<img width="1908" height="931" alt="Screenshot 2025-09-11 162305" src="https://github.com/user-attachments/assets/e01aff93-9a23-4358-a521-5ff1675427a2" />

Try it out page
<img width="1907" height="930" alt="Screenshot 2025-09-11 162445" src="https://github.com/user-attachments/assets/46492c7d-3308-403a-b100-9a3bf72f79ef" />


<img width="1919" height="942" alt="Screenshot 2025-09-11 162517" src="https://github.com/user-attachments/assets/ef9bb357-471d-421b-9bb9-2874b7b71acc" />

Download Result
<img width="1011" height="869" alt="Screenshot 2025-09-11 162542" src="https://github.com/user-attachments/assets/ef5e44c6-6003-4bb1-b362-0410a8bf543a" />


<img width="1906" height="923" alt="Screenshot 2025-09-11 162712" src="https://github.com/user-attachments/assets/720ac047-bb75-4d13-9d45-f2b4676b117f" />

<img width="1906" height="922" alt="Screenshot 2025-09-11 162802" src="https://github.com/user-attachments/assets/7ef290d3-fe31-4b17-9f0b-40eb99a2868f" />

History page
<img width="1905" height="925" alt="Screenshot 2025-09-11 162935" src="https://github.com/user-attachments/assets/23e36861-e1c2-49dc-b2f2-5cfd23d83405" />


<img width="1901" height="926" alt="Screenshot 2025-09-11 162956" src="https://github.com/user-attachments/assets/c9602292-352e-475d-bc1f-3b19d66bf817" />


<img width="1894" height="913" alt="Screenshot 2025-09-11 163017" src="https://github.com/user-attachments/assets/e9ef8566-e2a0-42a9-bcc9-a6d8cd58090d" />

Download History
<img width="932" height="849" alt="Screenshot 2025-09-11 163318" src="https://github.com/user-attachments/assets/ff19a7d9-51ff-4f75-9905-9d75da29a563" />

Profile page
<img width="1895" height="812" alt="Screenshot 2025-09-11 163454" src="https://github.com/user-attachments/assets/42ba08ea-77d0-4b7e-9a84-aa69988ea993" />


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

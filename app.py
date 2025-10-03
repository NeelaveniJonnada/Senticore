
import streamlit as st
import joblib
import pandas as pd
import re
from datetime import datetime
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.pdfgen import canvas



# ---------------------------
# Load model & vectorizer
# ---------------------------
model = joblib.load("senticore_model.pkl")
vectorizer = joblib.load("senticore_vectorizer.pkl")

# ---------------------------
# Page config & theme
# ---------------------------
st.set_page_config(page_title="Senticore • Sentiment Dashboard", page_icon="🧠", layout="wide")

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def set_theme():
    if st.session_state.theme == "dark":
        st.markdown("""
        <style>
        body, .stApp { background: #0f172a; color:white; }
        h1,h2,h3,h4,h5,h6 { color:#00ffd5; font-weight:700; }
        .label-bold { font-weight:700; font-size:1.05rem; }
        .stTextInput>div>div>input, .stTextArea>div>div>textarea { background-color:#0f172a; color:white; border:2px solid #00ffd5; border-radius:10px; padding:10px; }
        .stButton button { background: linear-gradient(45deg,#00ffd5,#10b981); color:black; font-weight:700; border-radius:10px; padding:10px 18px; }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        body, .stApp { background: #f9fafb; color:black; }
        h1,h2,h3,h4,h5,h6 { color:#1e3a8a; font-weight:700; }
        .label-bold { font-weight:700; font-size:1.05rem; }
        .stTextInput>div>div>input, .stTextArea>div>div>textarea { background-color:#ffffff; color:black; border:2px solid #4b5563; border-radius:10px; padding:10px; }
        .stButton button { background: linear-gradient(45deg,#3b82f6,#06b6d4); color:white; font-weight:700; border-radius:10px; padding:10px 18px; }
        </style>
        """, unsafe_allow_html=True)

set_theme()


# ✅ Custom style for Sign In & Sign Up form buttons
st.markdown("""
<style>
/* Target only form submit buttons */
.stForm button {
    background: orange;
    color: white;
    font-weight: 700;
    border-radius: 10px;
    padding: 10px 18px;
    border: none;
}

/* Hover effect */
.stForm button:hover {
    background: royalblue;
    color: black;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------
# Session-state defaults
# ---------------------------
if "users" not in st.session_state:
    st.session_state.users = {}
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "history" not in st.session_state:
    st.session_state.history = []
if "page" not in st.session_state:
    st.session_state.page = "Home"

# used for logout confirmation in the sidebar
if "logout_confirm" not in st.session_state:
    st.session_state.logout_confirm = False


def generate_result_pdf(text, result):
    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    c.setFont("Helvetica", 14)
    c.drawString(100, 750, "Senticore - Sentiment Analysis Result")
    c.setFont("Helvetica", 12)
    c.drawString(100, 700, f"Text: {text}")
    c.drawString(100, 670, f"Prediction: {result}")
    c.drawString(100, 640, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def generate_history_pdf(history):
    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 780, "Senticore - Sentiment Analysis History")
    y = 740
    c.setFont("Helvetica", 12)
    for entry in history:
        textline = f"{entry['time']} | {entry['text'][:80]} -> {entry['prediction']}"
        c.drawString(100, y, textline)
        y -= 20
        if y < 50:
            c.showPage()
            y = 780
    c.save()
    buffer.seek(0)
    return buffer

# ---------------------------
# Helper functions
# ---------------------------


def predict_sentiment(text: str):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]

    # ✅ Get real probabilities from RandomForest (or any classifier with predict_proba)
    probs = model.predict_proba(X)[0]

    # ✅ Map probabilities to class labels (lowercased for consistency)
    prob_dict = {
        cls.lower(): round(float(p * 100), 2)
        for cls, p in zip(model.classes_, probs)
    }

    return pred, prob_dict



def emoji_to_emotion(text):
    mapping = {"🙂":"joy", "😊":"joy", "😃":"joy", "😢":"sadness", "😭":"sadness", "😡":"anger", "😠":"anger", "😱":"fear", "😲":"surprise", "🤢":"disgust", "😐":"neutral"}
    for e, emo in mapping.items():
        if e in text:
            return emo
    if re.search(r'(:-\)|:\)|:D)', text): return "joy"
    if re.search(r'(:-\(|:\()', text): return "sadness"
    if "/s" in text.lower(): return "sarcasm"
    return None

def detect_emotion(text):
    emo = emoji_to_emotion(text)
    if emo: return "surprise" if emo == "sarcasm" else emo
    emotion_keywords = {"joy":["love","happy", "great","amazing"],"anger":["hate","angry","furious"],"sadness":["sad","terrible","worst"],"surprise":["wow","shocked"],"fear":["scared","worried"],"disgust":["disgusting","gross","nasty"]}
    txt = text.lower()
    for emo, kws in emotion_keywords.items():
        if any(k in txt for k in kws): return emo
    return "neutral"

def aspect_based_analysis(text):
    text_lower = text.lower()
    aspect_keywords = {"Camera":["camera","photo"],"Battery":["battery","charge"],"Screen":["screen","display"],"Performance":["speed","lag","performance"],"Design":["design","look"],"Price":["price","cost"],"Audio":["audio","sound","speaker"]}
    positive = {"good","great","excellent","love","nice","fast","bright"}
    negative = {"bad","terrible","worst","slow","dim","crash","broken"}
    aspects = {}
    for aspect, kws in aspect_keywords.items():
        if any(k in text_lower for k in kws):
            pos = sum(1 for p in positive if p in text_lower)
            neg = sum(1 for n in negative if n in text_lower)
            if pos > neg: aspects[aspect] = "positive"
            elif neg > pos: aspects[aspect] = "negative"
            else: aspects[aspect] = "neutral"
    if not aspects:
        general, _ = predict_sentiment(text)
        aspects["General"] = general
    return aspects


def sarcasm_detector(text):
    txt = text.lower()

    # Rule 1: explicit sarcasm cues
    if any(phrase in txt for phrase in ["/s", "yeah right", "as if", "sure thing", "just perfect", "oh wow"]):
        return True

    # Rule 2: "I love/like how/that ..." + negative context
    if re.search(r'\bi (love|like) (how|that)\b', txt) and re.search(r'\b(crash|fail|bad|worst|broken|useless)\b', txt):
        return True

    # Rule 3: positive + negative word mix
    if re.search(r'\b(good|great|love|amazing)\b', txt) and re.search(r'\b(bad|worst|hate|awful|terrible|disaster)\b', txt):
        return True

    return False


def explain_keywords(text, top_k=3):
    X = vectorizer.transform([text])
    arr = X.toarray()[0]
    if arr.sum() == 0:
        return re.findall(r'\w+', text)[:top_k]
    idxs = arr.argsort()[::-1]
    names = vectorizer.get_feature_names_out()
    kws = [names[i] for i in idxs if arr[i] > 0]
    clean_kws = [kw.replace("word__", "").replace("char__", "") for kw in kws]
    return clean_kws[:top_k]

def chatbot_response(sentiment):
    s = sentiment.lower().strip()   # normalize
    if s == "negative":
        return "💡 I'm sorry to hear that. Stay strong!"
    elif s == "positive":
        return "🎉 That's awesome! Keep going!"
    else:
        return "🙂 Got it! Thanks for sharing."
# ---------------------------
# Pages
# ---------------------------
def home_page():
    st.title("🏠 Welcome to Senticore")
    
    st.markdown("""
    **Senticore** 🧠 — **Analyze text. Detect sentiment. Uncover emotions.**  
    Quick insights, smart analysis, and actionable understanding — all in one tool.
    """, unsafe_allow_html=True)

    if st.session_state.get("logged_in", False):
        # Logged-in user → show Try it Out
        if st.button("🚀 Try it Out"):
            st.session_state.page = "Try it Out"
            st.rerun()
    else:
        # Not logged in → show Sign In / Sign Up
        # Not logged in → center Sign In / Sign Up
        col1, col2, col3 = st.columns([1, 2, 1])  # left, middle, right

        with col2:  # middle column
            c1, c2 = st.columns(2)  # split middle into two
            with c1:
                if st.button("🔑 Sign In"):
                    st.session_state.page = "Sign In"
            with c2:
                if st.button("🆕 Sign Up"):
                    st.session_state.page = "Sign Up"




def sign_in_page():
    st.markdown(
    """
    <style>
    /* Form container */
    div.stForm {
        background-color: #FFEBCD;  /* dark blue */
        padding: 20px;
        border-radius: 15px;
    }

    /* Input fields */
    div.stTextInput>div>div>input, div.stTextArea>div>div>textarea {
        background-color: #0f172a !important;  /* input box bg */
        color: white !important;
        border: 2px solid #00ffd5;
        border-radius: 10px;
        padding: 10px;
    }

    /* Button inside form */
    div.stButton>button {
        background: linear-gradient(45deg,#00ffd5,#10b981);
        color: black;
        font-weight: 700;
        border-radius: 10px;
        padding: 10px 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
    st.title("🔐 Sign In")
    st.write("Enter your credentials to sign in.")
    st.write("Please Sign in to access the application.")
    with st.form("signin_form"):
        user = st.text_input("Username", key="signin_user")
        pw = st.text_input("Password", type="password", key="signin_pass")
      
        submit = st.form_submit_button("Sign In")
    if submit:
        if not user or not pw:
            st.error("Please enter both username and password.")
        elif user in st.session_state.users and st.session_state.users[user] == pw:
            st.session_state.logged_in = True
            st.session_state.username = user
            st.success(f"✅ Logged in as {user}")
            # navigate to Try it Out (or Home) after sign-in
            st.session_state.page = "Home"
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

    if st.button("⬅ Back to Home"):
        st.session_state.page = "Home"
        st.rerun()


def sign_up_page():
    st.markdown(
    """
    <style>
    /* Form container */
    div.stForm {
        background-color: #FFEBCD;  /* dark blue */
        padding: 20px;
        border-radius: 15px;
    }

    /* Input fields */
    div.stTextInput>div>div>input, div.stTextArea>div>div>textarea {
        background-color: #0f172a !important;  /* input box bg */
        color: white !important;
        border: 2px solid #00ffd5;
        border-radius: 10px;
        padding: 10px;
    }

    /* Button inside form */
    div.stButton>button {
        background: linear-gradient(45deg,#00ffd5,#10b981);
        color: black;
        font-weight: 700;
        border-radius: 10px;
        padding: 10px 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
    st.title("📝 Sign Up")
    st.write("Create a new account.")
    with st.form("signup_form"):
        new_user = st.text_input("New Username", key="signup_new_user")
        new_pw = st.text_input("New Password", type="password", key="signup_new_pass")
        
        create = st.form_submit_button("Create Account")
    if create:
        if not new_user or not new_pw:
            st.error("Please enter a username and password.")
        elif new_user in st.session_state.users:
            st.error("Username already exists. Choose a different one.")
        else:
            st.session_state.users[new_user] = new_pw
            st.success("✅ Account created — please sign in.")
            st.session_state.page = "Sign In"
            st.rerun()

    if st.button("⬅ Back to Home"):
        st.session_state.page = "Home"
        st.rerun()



def profile_page():
    st.title("👤 Profile")
    if st.session_state.username:
        st.write(f"Hello, **{st.session_state.username}**!")
    else:
        st.write("You are not logged in. Please Sign In.")
        if st.button("⬅ Back to Home"):
            st.session_state.page = "Home"
            st.rerun()

def about_page():
    st.title("📖 About Senticore")
    
    st.markdown("""
Senticore is an advanced **🧠 emotion-aware** sentiment analysis platform designed to provide deep insights into textual data. Using **💻 state-of-the-art machine learning models** and **📚 natural language processing techniques**, Senticore can classify text into **✅ positive, ⚪ neutral, or ❌ negative sentiments**, detect **😊 emotions**, identify **😏 sarcasm**, and highlight **🔑 key aspects** mentioned in the text.

**✨ Key features include:**

- **🎯 Accurate Sentiment Prediction** – Analyze text for **✅ positive, ⚪ neutral, or ❌ negative sentiment** with confidence scores.
- **😊 Emotion Detection** – Detect underlying emotions such as **joy 😄, anger 😡, sadness 😢, fear 😱, disgust 🤢, and surprise 😲**, even from emojis or emoticons.
- **📌 Aspect-Based Analysis** – Identify the sentiment for different **aspects or features** mentioned in the text (e.g., **📷 camera, 🔋 battery, 💻 performance**).
- **😏 Sarcasm Detection** – Recognize **sarcastic comments** for better context understanding.
- **🔑 Explainability** – Highlight the most important **keywords** influencing sentiment predictions.
- **🤖 Chatbot Assistance** – Provides friendly **suggestions** or advice based on the detected sentiment.
- **📜 History & Reporting** – Save, visualize, and **export analysis history** in CSV or PDF formats.

Senticore empowers **users 👥, businesses 🏢, and researchers 🔬** to gain actionable insights from text data, improving **decision-making ✅**, enhancing **customer experience 💡**, and understanding **public opinion 🌐**.
""", unsafe_allow_html=True)


def how_to_use_page():
    st.title("ℹ️ Quick Guide")
    
    st.markdown("""
Welcome to **Senticore**! Here's how to get the most out of your sentiment analysis journey:  
""")
    
    # Step 1
    st.markdown("### 1️⃣ Go to **Try it Out**")
    st.markdown("Click on the **⚡ Try it Out** page from the sidebar to start analyzing text.")
    
    # Step 2
    st.markdown("### 2️⃣ Enter your **Text**")
    st.markdown("Type or paste your text in the input area. **Emojis 😄😢😡 are supported** and help improve emotion detection.")
    
    # Step 3
    st.markdown("### 3️⃣ Analyze & Explore")
    st.markdown("""
- View the **predicted sentiment**: ✅ Positive, ⚪ Neutral, ❌ Negative  
- Detect **emotions** like joy 😊, sadness 😢, anger 😡, and more  
- Check **aspect-based analysis** to see sentiment for different features (e.g., camera 📷, battery 🔋)  
- See **sarcasm detection** results  
- Get **keywords and chatbot suggestions 🤖** for insights
""")
    
    # Step 4
    st.markdown("### 4️⃣ Check Your **History**")
    st.markdown("All your analyzed text is saved automatically. View it on the **📜 History** page and **export it** as CSV or PDF for future reference.")
    
    # Bonus Tip
    st.markdown("💡 **Tip:** The more descriptive your text, the better Senticore can analyze sentiment and emotion!")
    
    # Optional: Add some visual spacing
    st.markdown("---")
    st.markdown("Enjoy exploring your text insights with **Senticore! 🧠**")

def try_it_out_page():
    st.title("⚡ Try it Out")
    # styled label we control
    st.markdown(
    '<div style="color:#B8860B; font-weight:600; font-size:16px; margin-bottom:6px;">💬 Enter your text here</div>',
    unsafe_allow_html=True
)

# actual textarea with the built-in label hidden
    text = st.text_area(
    "",                        # empty because we show our own label above
    height=100,
    key="tryit_input",
    label_visibility="collapsed",
    placeholder="Type or paste text here..."
)

    if st.button("🔮 Predict Sentiment"):
        if not text.strip():
            st.warning("⚠ Please enter some text.")
            return

        # ✅ Get prediction and fixed confidence scores
        sentiment, probs = predict_sentiment(text)
        emo = emoji_to_emotion(text) or detect_emotion(text)
        aspects = aspect_based_analysis(text)
        is_sarcastic = sarcasm_detector(text)
        keywords = explain_keywords(text)

        # 🎯 Show sentiment result
        st.markdown(f"### 🏷️ Prediction: **{sentiment.title()}**")

        # 🎯 Show confidence scores (fixed distributions)
        st.markdown("### 📊 Confidence Scores:")
        st.write(f"**Positive:** {probs.get('positive',0)}%")
        st.write(f"**Neutral:** {probs.get('neutral',0)}%")
        st.write(f"**Negative:** {probs.get('negative',0)}%")

        # 🎯 Plot bar chart with highlighted predicted sentiment
        # ✅ Sort probabilities (highest first)
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        classes = [cls.capitalize() for cls, _ in sorted_probs]
        values = [val for _, val in sorted_probs]

# ✅ Highlight predicted sentiment
        colors = []
        for cls in classes:
            if cls.lower() == sentiment.lower():
                if cls == "Positive":
                    colors.append("#10b981")  # green
                elif cls == "Negative":
                    colors.append("red")  # red
                else:
                    colors.append("blue")  # blue for neutral
            else:
                colors.append("#d1d5db")  # grey for others

        fig = go.Figure(go.Bar(
            x=values,
            y=classes,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{v}%" for v in values],
            textposition="auto"
        ))
        fig.update_layout(title="Sentiment Confidence (%)", xaxis=dict(range=[0, 100]))
        st.plotly_chart(fig, use_container_width=True)

        # 🎯 Emotion & aspects
        st.markdown(f"*Emotion:* {emo.title()}")
        st.markdown("*Aspect-based summary:*")
        for a, v in aspects.items():
            st.write(f"- *{a}:* {v.title()}")
        
        # Sarcasm Section
        st.subheader("🎭 Sarcasm Detection")
        if is_sarcastic:
            st.warning("⚠️ This text may contain **sarcasm**.")
        else:
            st.success("✅ No sarcasm detected.")
        st.markdown(f"*Keywords:* {', '.join(keywords)}")

        # 🎯 Chatbot response
       # 🎯 Chatbot response (custom styled box)
        st.subheader("🤖 Chatbot Says:")

        st.markdown(f"""
<div style="
    background-color:#FFFAF0;
    padding:15px;
    border-radius:10px;
    color:black;
    font-weight:600;
    font-size:1rem;
">
    {chatbot_response(sentiment)}
</div>
""", unsafe_allow_html=True)

        

        # 🎯 Save to history
        st.session_state.history.append({
            "text": text,
            "prediction": sentiment,
            "emotion": emo,
            "aspects": aspects,
            "sarcasm": is_sarcastic,
            "keywords": ", ".join(keywords),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
       

        # Download result
        st.subheader("📤 Share / Download")
        pdf_buf = generate_result_pdf(text, sentiment)
        st.markdown("""
    <style>
    div.stDownloadButton > button {
        background-color: #FF8C00;   /* Green */
        color: white;                /* Text color */
        border-radius: 8px;          /* Rounded corners */
        padding: 0.6em 1em;
        font-weight: bold;
    }
    div.stDownloadButton > button:hover {
        background-color: #45a049;   /* Darker green on hover */
        color: white;
    }
    </style>
""", unsafe_allow_html=True)
        st.download_button("Download Result as PDF", data=pdf_buf, file_name="senticore_result.pdf", mime="application/pdf")


def history_page():
    st.title("📜 Your History")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)

        # Pie chart
        st.subheader("📊 Sentiment Distribution")
        counts = df["prediction"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
        st.pyplot(fig)

        # Export
        csv = df.to_csv(index=False).encode("utf-8")
        st.markdown("""
    <style>
    div.stDownloadButton > button {
        background-color: #FF8C00;   /* Green */
        color: white;                /* Text color */
        border-radius: 8px;          /* Rounded corners */
        padding: 0.6em 1em;
        font-weight: bold;
    }
    div.stDownloadButton > button:hover {
        background-color: #45a049;   /* Darker green on hover */
        color: white;
    }
    </style>
""", unsafe_allow_html=True)
        st.download_button("Download History CSV", data=csv, file_name="history.csv", mime="text/csv")
        pdf_buf = generate_history_pdf(st.session_state.history)
        st.markdown("""
    <style>
    div.stDownloadButton > button {
        background-color: #FF8C00;   /* Green */
        color: white;                /* Text color */
        border-radius: 8px;          /* Rounded corners */
        padding: 0.6em 1em;
        font-weight: bold;
    }
    div.stDownloadButton > button:hover {
        background-color: #45a049;   /* Darker green on hover */
        color: white;
    }
    </style>
""", unsafe_allow_html=True)
        st.download_button("Download History PDF", data=pdf_buf, file_name="history.pdf", mime="application/pdf")
    else:
        st.info("No history yet. Try analyzing some text first!")


# ---------------------------
# Sidebar Navigation & Auth
# ---------------------------
def sidebar():
    # Theme toggle
    if st.sidebar.button("🌞 / 🌙 Toggle Theme"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

    st.sidebar.title("🧭 Navigation")

    # Navigation buttons (buttons only, no radio)
    if st.sidebar.button("🏠 Home"):
        st.session_state.page = "Home"
        st.rerun()
    if st.sidebar.button("ℹ️ About"):
        st.session_state.page = "About"
        st.rerun()
    if st.sidebar.button("❓ Quick Guide"):
        st.session_state.page = "How to Use"
        st.rerun()

    # Try it Out requires login — if not logged in, redirect to Sign In
    if st.sidebar.button("⚡ Try it Out"):
        if st.session_state.logged_in:
            st.session_state.page = "Try it Out"
        else:
            # redirect to sign in page when not logged
            st.session_state.page = "Sign In"
            # optional quick hint in sidebar before navigating
            st.sidebar.warning("Please sign in to access 'Try it Out'.")
        st.rerun()

    # History requires login as well
    if st.sidebar.button("📜 History"):
        if st.session_state.logged_in:
            st.session_state.page = "History"
        else:
            st.session_state.page = "Sign In"
            st.sidebar.warning("Please sign in to view History.")
        st.rerun()

    st.sidebar.markdown("---")

    # Show Profile & Logout only when logged in
    if st.session_state.logged_in:
        if st.sidebar.button("👤 Profile"):
            st.session_state.page = "Profile"
            st.rerun()

        # Logout flow with confirmation
        if not st.session_state.get("logout_confirm", False):
            if st.sidebar.button("🚪 Logout"):
                st.session_state.logout_confirm = True
                st.rerun()
        else:
            st.sidebar.warning("Are you sure you want to log out?")
            col_yes, col_no = st.sidebar.columns(2)
            if col_yes.button("Yes"):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.session_state.page = "Home"
                st.session_state.logout_confirm = False
                st.rerun()
            if col_no.button("No"):
                st.session_state.logout_confirm = False
                st.rerun()
    else:
        # When not logged in, a small hint to use Home page auth buttons
        st.sidebar.info("Sign In / Sign Up available on Home page")

    st.sidebar.markdown("---")



# ---------------------------
# Main
# ---------------------------
def main():
    sidebar()
    page = st.session_state.get("page", "Home")

    if page == "Home":
        home_page()

    elif page == "About":
        about_page()

    elif page == "How to Use":
        how_to_use_page()

    elif page == "Try it Out":
        if st.session_state.logged_in:
        # ✅ Only show analyzer when logged in
          try_it_out_page()
        else:
        # 🚫 Locked page for guests
          st.title("⚡ Try it Out")
          # Placeholder for auto-hide message
          
          st.markdown("### 🔒 Please **Sign In / Sign Up** to try the application out.")
          st.markdown("You need to log in to analyze text using Senticore.")

        # Back to Home button (centered)
          _, col, _ = st.columns([1,2,1])
          with col:
            if st.button("⬅ Back to Home"):
                st.session_state.page = "Home"
                st.rerun()

    elif page == "History":
        # enforce login
        if st.session_state.logged_in:
            history_page()
        else:
            st.warning("⚠ Please sign in to view History.")
            if st.button("🔑 Sign In to continue"):
                st.session_state.page = "Sign In"
                st.rerun()

    elif page == "Profile":
        profile_page()

    elif page == "Sign In":
        sign_in_page()

    elif page == "Sign Up":
        sign_up_page()



if __name__ == "__main__":
    main()

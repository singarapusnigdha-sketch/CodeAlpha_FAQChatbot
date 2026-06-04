import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string

# Download NLTK data (first time only)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# ─────────────────────────────────────────
# FAQ DATA (Edit this to add your own FAQs)
# ─────────────────────────────────────────
faq_data = [
    {"question": "What is CodeAlpha?", "answer": "CodeAlpha is a leading software development company focused on innovation and emerging technologies."},
    {"question": "How do I apply for an internship?", "answer": "You can apply for an internship through our official website at www.codealpha.tech."},
    {"question": "What is the duration of the internship?", "answer": "The internship typically lasts for 1 month, depending on the program you join."},
    {"question": "Will I get a certificate after the internship?", "answer": "Yes! You will receive a QR-verified completion certificate upon successfully finishing the internship tasks."},
    {"question": "What tasks are assigned during the AI internship?", "answer": "AI interns work on tasks like Language Translation Tool, FAQ Chatbot, Music Generation, and Object Detection."},
    {"question": "How many tasks do I need to complete?", "answer": "You need to complete at least 2 or 3 out of the 4 assigned tasks to be eligible for the certificate."},
    {"question": "Where do I submit my completed tasks?", "answer": "You must submit your completed tasks through the submission form shared in your respective WhatsApp group."},
    {"question": "Do I need to upload code to GitHub?", "answer": "Yes, upload your complete source code to GitHub in a repository named CodeAlpha_ProjectName."},
    {"question": "What perks do I get from this internship?", "answer": "You get an offer letter, completion certificate, unique ID certificate, letter of recommendation (based on performance), and placement support."},
    {"question": "How can I contact CodeAlpha?", "answer": "You can contact CodeAlpha via email at services@codealpha.tech or WhatsApp at +91 9336576683."},
    {"question": "Is there any stipend for the internship?", "answer": "Stipend details are shared during the onboarding process. Please check with the CodeAlpha team."},
    {"question": "What is Python?", "answer": "Python is a high-level, easy-to-learn programming language widely used in AI, data science, and web development."},
    {"question": "What is machine learning?", "answer": "Machine learning is a branch of AI where computers learn from data to make predictions or decisions without being explicitly programmed."},
    {"question": "What is NLP?", "answer": "NLP stands for Natural Language Processing. It is a field of AI that helps computers understand and process human language."},
]

# ─────────────────────────────────────────
# TEXT PREPROCESSING
# ─────────────────────────────────────────
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# Preprocess all FAQ questions
questions = [faq["question"] for faq in faq_data]
answers = [faq["answer"] for faq in faq_data]
processed_questions = [preprocess(q) for q in questions]

# ─────────────────────────────────────────
# FIND BEST ANSWER USING COSINE SIMILARITY
# ─────────────────────────────────────────
def get_best_answer(user_query, threshold=0.2):
    processed_query = preprocess(user_query)
    vectorizer = TfidfVectorizer()
    all_texts = processed_questions + [processed_query]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    query_vec = tfidf_matrix[-1]
    faq_vecs = tfidf_matrix[:-1]
    
    similarities = cosine_similarity(query_vec, faq_vecs).flatten()
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]
    
    if best_score < threshold:
        return "Sorry, I couldn't find a relevant answer. Please try rephrasing your question.", best_score
    
    return answers[best_idx], best_score

# ─────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────
st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 FAQ Chatbot")
st.markdown("Ask me anything! I'll find the best answer from the FAQs.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["user"])
    with st.chat_message("assistant"):
        st.write(chat["bot"])

# Input
user_input = st.chat_input("Type your question here...")

if user_input:
    # Get answer
    answer, score = get_best_answer(user_input)
    
    # Show in chat
    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("assistant"):
        st.write(answer)
        st.caption(f"Confidence: {score:.0%}")
    
    # Save to history
    st.session_state.chat_history.append({"user": user_input, "bot": answer})

# Sidebar - Show available FAQs
with st.sidebar:
    st.header("📋 Available FAQs")
    st.markdown("Here are some questions you can ask:")
    for q in questions:
        st.markdown(f"• {q}")

# Footer
st.markdown("---")
st.markdown("Built with `NLTK` + `Scikit-learn` + `Streamlit` 🐍")

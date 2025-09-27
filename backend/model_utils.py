# model.py

import re
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
import torch

# -----------------------------
# Stopwords et phrases à filtrer
stops = stopwords.words("english")
FILTER_PHRASES = [
    "strong interest", "i have a solid understanding", "mindset focused",
    "i am committed", "continuous learning", "delivering impactful",
    "please contact", "phone number", "email", "gmail", "linkedin"
]

# -----------------------------
# Fonctions de nettoyage
def filter_phrases(text):
    for phrase in FILTER_PHRASES:
        text = re.sub(re.escape(phrase), '', text, flags=re.I)
    return text

def remove_consecutive_duplicates(text):
    return re.sub(r'\b(\w+)( \1\b)+', r'\1', text, flags=re.IGNORECASE)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\b[\w.-]+?@\w+?\.\w+?\b', ' ', text)
    text = re.sub(r'\b\d{7,15}\b', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [w for w in words if w not in stops and len(w) > 2]
    text = ' '.join(words)
    text = remove_consecutive_duplicates(text)
    return text

# -----------------------------
# Extraction années et lieux
def extract_years_and_places(text):
    years = list(set(re.findall(r'\b(20[2-3][0-9])\b', text)))
    places = []
    for place in ['sfax', 'tunis', 'paris', 'lyon', 'france', 'tunisia', 'germany']:
        if place in text.lower():
            places.append(place)
    return years, places

# -----------------------------
# Extraction compétences, expérience, mots-clés
def extract_sections(text):
    text = text.lower()
    text = re.sub(r'\b[\w.-]+?@\w+?\.\w+?\b', ' ', text)
    text = re.sub(r'\+?\d[\d\s\-]{8,}', ' ', text)
    text = re.sub(r'\b(?:tunisia|france|germany|sfax|paris|tunis|project|job|jobs)\b', ' ', text)

    lines = list(set([line.strip() for line in text.split('\n') if line.strip()]))

    tech_keywords = [line for line in lines if any(kw in line for kw in [
        'tools', 'technologies', 'stack', 'skills', 'frameworks', 'languages'])]

    skill_words = []
    for line in tech_keywords:
        skill_words.extend(re.findall(r'\b[a-zA-Z0-9#\+\-\.]{3,}\b', line))

    blacklist = {'developed', 'tools', 'technologies', 'project', 'system'}
    final_skills = sorted(set([w for w in skill_words if w not in blacklist]))

    exp_lines = [line for line in lines if 'developed' in line or 'internship' in line]
    experience = "\n- " + "\n- ".join(dict.fromkeys(exp_lines)) if exp_lines else "No experience found."

    vectorizer = CountVectorizer(stop_words='english', max_features=30)
    X = vectorizer.fit_transform([text])
    words = vectorizer.get_feature_names_out()
    keywords = ', '.join(sorted(set(w for w in words if w not in blacklist)))

    return ', '.join(final_skills), experience.strip(), keywords

# -----------------------------
# Chargement du modèle SBERT
def load_sbert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
def match_cv_to_jobs(cv_text, job_texts_clean, model, job_embeddings, top_k=5):
    import numpy as np
    from sentence_transformers import util

    # Embedding du CV
    cv_embedding = model.encode([cv_text], convert_to_tensor=True)

    # Similarité cosinus
    cos_scores = util.cos_sim(cv_embedding, job_embeddings)[0]

    # Top k
    top_results = np.argpartition(-cos_scores.cpu().numpy(), range(top_k))[:top_k]
    scores = cos_scores[top_results].cpu().numpy().tolist()

    # Conversion explicite des indices en int natif
    top_results = [int(i) for i in top_results]

    return list(zip(scores, top_results))



# 🔍 Matching CV – Offres d’Emploi
 ## Introduction

Ce projet propose un système qui permet à un candidat d’envoyer son CV et de recevoir automatiquement les **5 offres d’emploi les plus pertinentes**, grâce au modèle **Sentence‑BERT (SBERT)**. Le but est de faciliter la recherche d’emploi et d’améliorer la précision du matching.


##  Fonctionnement

1. Le candidat envoie son CV (PDF, texte ou image).
2. Le texte est extrait et prétraité (nettoyage, normalisation, tokenisation…).
3. SBERT génère des embeddings pour le CV et les offres.
4. Le système calcule la similarité cosinus et sélectionne les 5 meilleures offres.
5. Le frontend affiche les résultats de manière simple et claire.

---

##  Technologies utilisées

* **Python**, **FastAPI**, **PyTorch**, **SentenceTransformers**
* **Pandas**, **NumPy**, **NLTK**, **SpaCy**
* **Matplotlib** (visualisation)
* **PyMuPDF / PyTesseract** (OCR)
* **React**, **HTML**, **CSS3** (frontend)

---

## 🚀 Installation

```bash
pip install -r requirements.txt
```

### Lancer le backend

```bash
uvicorn main:app --reload
```

### Lancer le frontend

```bash
npm install
npm start
```

---


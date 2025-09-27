from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import fitz, pytesseract
from PIL import Image
from io import BytesIO
import numpy as np
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from model_utils import clean_text, extract_sections, extract_years_and_places, load_sbert_model, match_cv_to_jobs

#from model_utils import extract_sections, extract_years_and_places, clean_text
import sqlite3
import hashlib
import re
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Autorise l'accès depuis frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="../templates")



# Load data for CV matching
df_jobs = pd.read_csv(
    "job_cleaned.csv",
    usecols=["sector", "job_title_clean", "job_description_clean", "location", "salary"]
)

df_jobs = df_jobs.rename(columns={
    "sector": "domaine",
    "job_title_clean": "job_title",
    "job_description_clean": "job_description"
})
model = load_sbert_model()
model.eval()

job_texts_clean = [clean_text(desc) for desc in df_jobs["job_description"]]
#df_jobs = df_jobs[~df_jobs["job_title"].str.contains("Monster", case=False, na=False)]

job_embeddings = np.load("job_embeddings.npy")
#model = SentenceTransformer("all-MiniLM-L6-v2")

# --- SQLITE & AUTH UTILS ---

def get_db():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn

def create_users_table():
    conn = get_db()
    conn.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()

create_users_table()
def remove_consecutive_duplicates(text):
    # Supprime les mots répétés plusieurs fois de suite
    return re.sub(r'\b(\w+)( \1\b)+', r'\1', text, flags=re.IGNORECASE)

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(email: str, password: str) -> bool:
    conn = get_db()
    cur = conn.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cur.fetchone()
    conn.close()
    return user and user["password"] == hash_password(password)

def user_exists(email: str) -> bool:
    conn = get_db()
    cur = conn.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cur.fetchone()
    conn.close()
    return user is not None

def create_user(email: str, password: str) -> bool:
    try:
        conn = get_db()
        conn.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, hash_password(password)))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def extract_title_from_description(desc: str) -> str:
    # Extrait la première phrase terminée par '.', '!' ou '?'
    match = re.search(r'([^.?!]+[.?!])', desc)
    if match:
        return match.group(1).strip()
    # Sinon, on prend les 50 premiers caractères avec "..."
    return desc[:50].strip() + "..."
# --- ROUTES AUTHENTIFICATION ---
@app.get("/recherche", response_class=HTMLResponse)
def recherche_get(request: Request):
    domaines = sorted(df_jobs["domaine"].dropna().unique())
    return templates.TemplateResponse("recherche.html", {
        "request": request,
        "results": [],
        "keyword": "",
        "selected_domaine": "",
        "domaines": domaines
    })

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

@app.post("/login", response_class=HTMLResponse)
async def login_post(request: Request, email: str = Form(...), password: str = Form(...)):
    if verify_user(email, password):
        return RedirectResponse(url="/analyse", status_code=302)
    else:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Email ou mot de passe incorrect"})

@app.get("/signup", response_class=HTMLResponse)
def signup_get(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request, "error": None})

@app.post("/recherche", response_class=HTMLResponse)
async def recherche_post(
    request: Request,
    keyword: str = Form(""),
    domaine: str = Form("")
):
    filtered = df_jobs.copy()

    if keyword:
        filtered = filtered[
            filtered["job_title"].str.contains(keyword, case=False, na=False) |
            filtered["job_description"].str.contains(keyword, case=False, na=False)
        ]

    if domaine:
        filtered = filtered[filtered["domaine"] == domaine]

    results = filtered.to_dict(orient="records")
    domaines = sorted(df_jobs["domaine"].dropna().unique())

    return templates.TemplateResponse("recherche.html", {
        "request": request,
        "results": results,
        "keyword": keyword,
        "selected_domaine": domaine,
        "domaines": domaines
    })

@app.post("/signup", response_class=HTMLResponse)
async def signup_post(request: Request, email: str = Form(...), password: str = Form(...), password2: str = Form(...)):
    if password != password2:
        return templates.TemplateResponse("signup.html", {"request": request, "error": "Les mots de passe ne correspondent pas"})
    if user_exists(email):
        return templates.TemplateResponse("signup.html", {"request": request, "error": "Cet email est déjà utilisé"})
    if create_user(email, password):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse("signup.html", {"request": request, "error": "Erreur lors de la création de l'utilisateur"})

@app.get("/analyse", response_class=HTMLResponse)
def analyse_page(request: Request):
    return templates.TemplateResponse("analyse.html", {"request": request})

# --- ROUTE ANALYSE CV EXISTANTE ---


@app.post("/match-cv/")
async def match_cv(file: UploadFile):
    import time
    content = await file.read()

    start = time.time()

    # Extraction du texte
    if file.filename.endswith(".pdf"):
        doc = fitz.open(stream=content, filetype="pdf")
        raw_text = ''.join([page.get_text() for page in doc])
    else:
        image = Image.open(BytesIO(content))
        raw_text = pytesseract.image_to_string(image)

    print("Texte extrait en", time.time() - start, "secondes")

    if not raw_text.strip():
        return {"results": []}

    start = time.time()

    # Matching avec embeddings pré-calculés
    results_idx_scores = match_cv_to_jobs(raw_text, job_texts_clean, model, job_embeddings, top_k=5)
    print("Matching terminé en", time.time() - start, "secondes")

    results = []
    for score, idx in results_idx_scores:
        
            
        original_title = df_jobs.iloc[idx]["job_title"]
        description = df_jobs.iloc[idx]["job_description"]
        clean_description = clean_text(description)

        if re.search(r'monster', original_title, re.IGNORECASE):
            custom_title = extract_title_from_description(description)
        else:
            custom_title = original_title

        results.append({
             "score": round(score, 4),
             "job_title": custom_title,
             "job_description": clean_description,
        })

    return {"results": results}

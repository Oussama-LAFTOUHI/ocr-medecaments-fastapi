import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import pandas as pd
import easyocr
import re
from difflib import get_close_matches, SequenceMatcher
from fastapi import FastAPI, UploadFile, File
import tempfile

# Chargement CSV
df = pd.read_csv("medicaments.csv")
df['nom'] = df['nom'].astype(str).str.lower().str.strip()
noms_list = df['nom'].tolist()

reader = easyocr.Reader(['ar','en'], gpu=False)

def nettoyer_ocr(mot):
    mot = mot.translate(str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789'))
    mot = mot.lower().strip()
    mot = mot.replace('0', 'o').replace('1', 'l').replace('5', 's').replace('8', 'b')
    mot = re.sub(r'(\d)[o]', r'\g<1>0', mot)
    mot = re.sub(r'[l](\d)', r'1\g<1>', mot)
    mot = re.sub(r'[^a-z0-9-]', '', mot)
    return mot

def extraire_dosage(texte_complet, text_list):
    texte_corrige = texte_complet.replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1')
    pattern = r'(\d+[\.,]?\d*)\s*(mg|ml|g|µg|ui)'
    match = re.search(pattern, texte_corrige, re.IGNORECASE)
    if match:
        return f"{match.group(1).replace(',', '.')}{match.group(2).lower()}"
    return None

def read_texte(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "Error: image non trouvée"
    result = reader.readtext(img)
    text = [txt for bbox, txt, conf in result if conf > 0.2]
    if not text:
        return "le texte pas clair"
    texte_complet = " ".join(text)
    dosage = extraire_dosage(texte_complet, text)
    candidats = []
    for mot in text:
        if len(mot) < 4: continue
        mot_clean = nettoyer_ocr(mot)
        if mot_clean in noms_list:
            res = mot_clean
            if dosage: res += f" {dosage} 💊"
            return res
        matches = get_close_matches(mot_clean, noms_list, n=1, cutoff=0.5)
        if matches:
            score = SequenceMatcher(None, mot_clean, matches[0]).ratio()
            candidats.append((score, matches[0]))
    if candidats:
        candidats.sort(reverse=True, key=lambda x: x[0])
        res = candidats[0][1]
        if dosage: res += f" {dosage} 💊"
        return res
    return "non trouvé"

app = FastAPI(title="API Reconnaissance Médicaments OCR")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    result = read_texte(tmp_path)
    os.remove(tmp_path)
    return {"medicament_identifie": result}

@app.get("/")
def home():
    return {"message": "API OCR Médicaments - POST /predict avec image"}
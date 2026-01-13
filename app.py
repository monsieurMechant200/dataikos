from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import numpy as np
import uvicorn
import os

app = FastAPI(
    title="DATAIKÔS - Prédiction de Réussite",
    description="Interface complète avec frontend intégré",
    version="2.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montez un dossier pour les fichiers statiques (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Créez le dossier static s'il n'existe pas
os.makedirs("static", exist_ok=True)

# Coefficients du modèle
THETA = np.array([
    -6.60598966, -0.38887618, -0.05527884, 0.24046097,
    14.34727348, -0.66947624, 0.0156859, -0.17815602,
    0.32959119, -0.01474053, 0.08425352, -0.45180327,
    0.02356171, -0.2706575, -0.9303213
])

MIN_VALS = np.array([16., 0., 1., 7., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
MAX_VALS = np.array([30., 1., 5., 17.05, 3., 1., 2., 2., 1., 2., 418., 758., 230., 6.])

# Modèle de données
class StudentData(BaseModel):
    Age: float
    Gender: int
    Level: int
    GPA: float
    Teaching_Quality: float
    Lab_Sessions: int
    Structured_Plan: int
    Living_Situation: int
    Sleep_Hours_Daily: float
    Physical_Activity: int
    Success_Factors_Len: int
    Improvement_Suggestions_Len: int
    Study_Hours_Weekly: int
    Class_Regularity: float

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

# Route pour servir la page d'accueil HTML
@app.get("/")
async def serve_homepage():
    """Sert la page HTML principale"""
    return FileResponse("templates/index.html")

# Route pour la santé
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Route API pour les prédictions
@app.post("/api/predict")
async def predict_complete(data: StudentData):
    """Endpoint principal pour le frontend avec recommandations"""
    try:
        # 1. Calcul de la prédiction
        X = np.array([[ 
            data.Age, data.Gender, data.Level, data.GPA,
            data.Teaching_Quality, data.Lab_Sessions,
            data.Structured_Plan, data.Living_Situation,
            data.Sleep_Hours_Daily, data.Physical_Activity,
            data.Success_Factors_Len, data.Improvement_Suggestions_Len,
            data.Study_Hours_Weekly, data.Class_Regularity
        ]])
        
        scaled = (X - MIN_VALS) / (MAX_VALS - MIN_VALS + 1e-8)
        Xb = np.c_[np.ones(1), scaled]
        prob = sigmoid(Xb @ THETA)[0]
        prediction = int(prob >= 0.5)
        
        # 2. Calcul des dimensions
        academic_score = min((data.GPA / 20) * 6 + (data.Class_Regularity / 6) * 4, 10)
        study_score = min((data.Study_Hours_Weekly / 30) * 4 + (data.Structured_Plan / 2) * 3 + data.Lab_Sessions * 3, 10)
        wellbeing_score = min((data.Sleep_Hours_Daily / 8) * 6 + (data.Physical_Activity / 2) * 4, 10)
        
        living_bonus = {0: 0.6, 1: 0.4, 2: 0.8}
        env_score = min((data.Teaching_Quality / 3) * 6 + living_bonus.get(data.Living_Situation, 0.5) * 4, 10)
        
        motivation_score = min((data.Success_Factors_Len / 250) * 5 + (data.Improvement_Suggestions_Len / 250) * 5, 10)
        
        dimensions = [
            {
                "name": "Performance Académique",
                "score": round(academic_score, 1),
                "status": "Excellent" if academic_score >= 7 else "Correct" if academic_score >= 5 else "À améliorer",
                "icon": "📚"
            },
            {
                "name": "Méthodes d'Étude", 
                "score": round(study_score, 1),
                "status": "Excellent" if study_score >= 7 else "Correct" if study_score >= 5 else "À améliorer",
                "icon": "📖"
            },
            {
                "name": "Bien-être Physique",
                "score": round(wellbeing_score, 1),
                "status": "Excellent" if wellbeing_score >= 7 else "Correct" if wellbeing_score >= 5 else "À améliorer", 
                "icon": "💪"
            },
            {
                "name": "Environnement d'Étude",
                "score": round(env_score, 1),
                "status": "Excellent" if env_score >= 7 else "Correct" if env_score >= 5 else "À améliorer",
                "icon": "🏠"
            },
            {
                "name": "Motivation & Réflexivité",
                "score": round(motivation_score, 1),
                "status": "Excellent" if motivation_score >= 7 else "Correct" if motivation_score >= 5 else "À améliorer",
                "icon": "🔥"
            }
        ]
        
        overall_score = sum(d["score"] for d in dimensions) / len(dimensions)
        
        # 3. Points forts/faibles
        weak_points = [d["name"] for d in dimensions if d["score"] < 5]
        strong_points = [d["name"] for d in dimensions if d["score"] >= 7]
        
        # 4. Recommandations
        recommendations = []
        rec_id = 1
        
        if data.GPA < 10:
            recommendations.append({
                "id": rec_id,
                "priority": 1,
                "category": "Académique",
                "title": "Plan de rattrapage académique",
                "emoji": "📈",
                "diagnosis": f"GPA ({data.GPA}/20) en dessous du seuil critique de 12/20.",
                "actions": [
                    "Réviser les bases dans les matières principales",
                    "Faire des exercices supplémentaires chaque semaine",
                    "Consulter les enseignants pour des conseils personnalisés"
                ],
                "impact": "Élevé (+20%)",
                "timeframe": "1 mois",
                "difficulty": "Modéré"
            })
            rec_id += 1
            
        if data.Sleep_Hours_Daily < 6:
            recommendations.append({
                "id": rec_id,
                "priority": 1,
                "category": "Bien-être",
                "title": "Améliorer la qualité du sommeil",
                "emoji": "😴",
                "diagnosis": f"Sommeil insuffisant ({data.Sleep_Hours_Daily}h/jour), impact significatif sur la concentration.",
                "actions": [
                    "Établir une heure de coucher régulière",
                    "Éviter les écrans 1 heure avant le sommeil",
                    "Créer un environnement de sommeil calme et sombre"
                ],
                "impact": "Élevé (+15%)",
                "timeframe": "1 semaine",
                "difficulty": "Facile"
            })
            rec_id += 1
            
        if data.Study_Hours_Weekly < 15:
            recommendations.append({
                "id": rec_id,
                "priority": 2,
                "category": "Méthodologie",
                "title": "Augmenter le temps d'étude",
                "emoji": "⏰",
                "diagnosis": f"Volume d'étude ({data.Study_Hours_Weekly}h/semaine) insuffisant pour une bonne performance.",
                "actions": [
                    "Ajouter 5 heures d'étude par semaine progressivement",
                    "Utiliser la technique Pomodoro (25 min travail / 5 min pause)",
                    "Planifier des sessions d'étude régulières"
                ],
                "impact": "Modéré (+12%)",
                "timeframe": "2 semaines",
                "difficulty": "Facile"
            })
            rec_id += 1
            
        if data.Class_Regularity < 3:
            recommendations.append({
                "id": rec_id,
                "priority": 2,
                "category": "Assiduité",
                "title": "Améliorer l'assiduité en classe",
                "emoji": "✅",
                "diagnosis": f"Faible régularité ({data.Class_Regularity}/6), risque de manquer des informations importantes.",
                "actions": [
                    "Assister à tous les cours obligatoires",
                    "Prendre des notes systématiquement",
                    "Participer activement en classe"
                ],
                "impact": "Modéré (+10%)",
                "timeframe": "1 semaine",
                "difficulty": "Facile"
            })
            rec_id += 1
            
        if data.Physical_Activity == 0:
            recommendations.append({
                "id": rec_id,
                "priority": 3,
                "category": "Santé",
                "title": "Intégrer l'activité physique",
                "emoji": "🏃‍♂️",
                "diagnosis": "Aucune activité physique, important pour la santé mentale et la concentration.",
                "actions": [
                    "Marcher 30 minutes par jour",
                    "Pratiquer une activité sportive 2-3 fois par semaine",
                    "Faire des étirements matinaux"
                ],
                "impact": "Modéré (+8%)",
                "timeframe": "1 semaine",
                "difficulty": "Facile"
            })
            rec_id += 1
        
        # Limiter à 5 recommandations maximum
        recommendations = recommendations[:5]
        
        # 5. Message de motivation
        if prediction == 1:
            if prob > 0.8:
                motivational_message = "🎉 Excellent profil ! Tu es sur la bonne voie pour réussir brillamment."
            else:
                motivational_message = "👍 Bonnes chances de réussite ! Quelques ajustements pourraient encore améliorer tes résultats."
        else:
            if prob > 0.4:
                motivational_message = "⚠️ Tu es proche du seuil de réussite ! Avec quelques efforts ciblés, tu peux y arriver."
            else:
                motivational_message = "🔧 Des améliorations sont nécessaires, mais avec un plan d'action adapté, tu peux augmenter significativement tes chances."
        
        # 6. Amélioration potentielle
        potential_improvement = min(prob + 0.25, 0.98)
        
        return {
            "prediction": prediction,
            "probability": float(prob),
            "profile": {
                "dimensions": dimensions,
                "overall_score": round(overall_score, 1),
                "weak_points": weak_points,
                "strong_points": strong_points,
                "risk_level": "Critique" if overall_score < 4 else ("Stable" if overall_score < 6 else "Excellent")
            },
            "recommendations": recommendations,
            "potential_improvement": float(potential_improvement),
            "motivational_message": motivational_message
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement: {str(e)}")

if __name__ == "__main__":

    print("🚀 DATAIKÔS - Interface Complète")

    print("📊 Backend API: http://localhost:8000")
    print("🎨 Interface HTML: http://localhost:8000")
    print("📚 Documentation API: http://localhost:8000/docs")
    print("🔧 Health check: http://localhost:8000/health")

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
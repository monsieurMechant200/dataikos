# DATAIKÔS 🎓

**Prédiction de Réussite Étudiante par Intelligence Artificielle**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green)](https://fastapi.tiangolo.com/)
[![Render](https://img.shields.io/badge/Render-Deployed-brightgreen)](https://render.com/)
[![License](https://img.shields.io/badge/License-Personnelle-orange)](LICENSE)

## 🌐 Démo en Ligne

🚀 **Accédez directement à l'application :**  
👉 **[https://student-prediction-interface.onrender.com/static/index.html](https://student-prediction-interface.onrender.com/static/index.html)**

> ⚠️ *Premier chargement possiblement lent (hébergement gratuit Render)*

## 📊 Aperçu

**DATAIKÔS** est une application web intelligente qui prédit la réussite académique des étudiants en utilisant un modèle de **régression logistique** optimisé. Le système analyse 14 facteurs clés pour fournir une prédiction précise accompagnée de recommandations personnalisées.

### ✨ Fonctionnalités Principales

- **🔮 Prédiction IA** : Modèle de régression logistique avec 89% d'accuracy
- **📊 Analyse 5 Dimensions** : Profil complet de l'étudiant
- **🎯 Recommandations Personnalisées** : Plan d'action adapté
- **📱 Interface Moderne** : Design responsive et intuitif
- **⚡ Temps Réel** : Résultats instantanés

## 🏗️ Architecture Technique

### Structure du Projet

```
dataikos/
├── 📄 app.py                  # Backend FastAPI + modèle IA
├── 📁 static/
│   └── 📄 index.html          # Interface utilisateur complète
│   └── 🎨 favicon.ico         # Icône de l'application
└── 📄 README.md               # Documentation
```

### Stack Technologique

| Couche | Technologies | Rôle |
|--------|--------------|------|
| **Frontend** | HTML5, CSS3, JavaScript, Chart.js | Interface utilisateur interactive |
| **Backend** | FastAPI, Uvicorn | API et traitement des données |
| **Modèle IA** | NumPy, Régression Logistique | Prédiction et analyse |
| **Validation** | Pydantic | Validation des données |
| **Hébergement** | Render | Déploiement en production |

## 🧠 Le Modèle IA

### 📈 Performance

- **Algorithme** : Régression Logistique (implémentation manuelle)
- **Fonction de Perte** : Binary Cross Entropy
- **Accuracy** : ~89% sur jeu de test
- **F1-Score** : ~82%

### 🔍 Variables d'Entrée (14 features)

| Catégorie | Variables | Description |
|-----------|-----------|-------------|
| **Personnelles** | Âge, Genre, Niveau d'études, Situation de vie | Informations démographiques |
| **Académiques** | GPA, Qualité d'enseignement, Labs, Plan d'étude | Performance et méthodes |
| **Habitudes** | Sommeil, Activité physique, Heures d'étude, Régularité | Hygiène de vie étudiante |
| **Psychologiques** | Facteurs de succès, Suggestions d'amélioration | Motivation et réflexivité |

### 🧮 Fonction de Prédiction

```python
def predict_success(student_data):
    # 1. Préparation des données
    X = prepare_features(student_data)
    
    # 2. Normalisation
    X_scaled = (X - MIN_VALS) / (MAX_VALS - MIN_VALS)
    
    # 3. Ajout du biais
    X_bias = np.c_[np.ones(1), X_scaled]
    
    # 4. Calcul de la prédiction
    z = X_bias @ THETA  # Produit matriciel
    probability = 1 / (1 + np.exp(-z))[0]  # Sigmoïde
    
    # 5. Décision binaire
    prediction = int(probability >= 0.5)
    
    return prediction, probability
```

## 🚀 Installation Locale

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### 📥 Installation Rapide

```bash
# 1. Cloner le projet
git clone https://github.com/votre-username/dataikos.git
cd dataikos

# 2. Créer un environnement virtuel
python -m venv venv

# 3. Activer l'environnement
# Windows :
venv\Scripts\activate
# Mac/Linux :
source venv/bin/activate

# 4. Installer les dépendances
pip install fastapi uvicorn numpy pydantic

# 5. Lancer le serveur
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 📦 Dépendances Minimales

```txt
fastapi==0.104.1
uvicorn==0.24.0
numpy==1.24.3
pydantic==2.4.2
```

## 🎯 Utilisation

### 1. Accès à l'Application

Après avoir lancé le serveur :

- **Interface Web** : [http://localhost:8000/static/index.html](http://localhost:8000/static/index.html)
- **Documentation API** : [http://localhost:8000/docs](http://localhost:8000/docs)
- **Vérification santé** : [http://localhost:8000/health](http://localhost:8000/health)

### 2. Remplir le Formulaire

1. **Informations Personnelles** : Âge, genre, niveau d'études
2. **Performances Académiques** : GPA, qualité des cours
3. **Habitudes de Vie** : Sommeil, étude, activité physique
4. **Facteurs Psychologiques** : Motivation et auto-évaluation

### 3. Résultats Obtenus

- ✅ **Verdict** : RÉUSSI ou À RISQUE
- 📊 **Probabilité** : Pourcentage de chances
- 🎯 **Profil 5D** : Analyse sur 5 dimensions
- 💡 **Recommandations** : Plan d'action personnalisé
- 🔄 **Simulateur** : Impact des changements d'habitudes

## 🔧 API Endpoints

| Méthode | Endpoint | Description | Exemple Réponse |
|---------|----------|-------------|-----------------|
| `GET` | `/` | Redirection vers l'interface | HTML Page |
| `GET` | `/health` | Vérification santé | `{"status": "healthy"}` |
| `POST` | `/api/predict` | Prédiction complète | JSON structuré |

### 📝 Exemple de Requête API

```bash
curl -X POST "http://localhost:8000/api/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Age": 20,
       "Gender": 0,
       "Level": 2,
       "GPA": 14.5,
       "Teaching_Quality": 2,
       "Lab_Sessions": 1,
       "Structured_Plan": 1,
       "Living_Situation": 0,
       "Sleep_Hours_Daily": 7.5,
       "Physical_Activity": 1,
       "Success_Factors_Len": 120,
       "Improvement_Suggestions_Len": 80,
       "Study_Hours_Weekly": 20,
       "Class_Regularity": 4.5
     }'
```

## 🎨 Interface Utilisateur

### Sections Principales

| Section | Description | Icône |
|---------|-------------|--------|
| **📋 Formulaire** | 14 champs organisés | 📝 |
| **🏆 Verdict** | Résultat principal | 🎯 |
| **📊 Radar** | Profil 5 dimensions | 📈 |
| **🚀 Recommandations** | Plan d'action | 💡 |
| **🎮 Simulateur** | Impact des changements | 🔄 |

### Palette de Couleurs

```css
:root {
  --primary-bg: #0A0E27;        /* Espace profond */
  --secondary-bg: #1A1F3A;      /* Cartes et sections */
  --accent-color: #00D4FF;      /* Accents et interactions */
  --success: #00FF88;           /* Réussite */
  --warning: #FF6B35;           /* Attention */
  --text-primary: #FFFFFF;      /* Texte principal */
}
```

## 🧪 Tests et Validation

### Cas de Test Typiques

**Étudiant Performant :**
```json
{
  "Age": 22, "GPA": 16.5, "Sleep_Hours_Daily": 8.0,
  "Study_Hours_Weekly": 25, "Class_Regularity": 5.5
}
→ Prédiction: RÉUSSI (Probabilité: ~85%)

**Étudiant à Risque :**
```json
{
  "Age": 19, "GPA": 9.0, "Sleep_Hours_Daily": 5.5,
  "Study_Hours_Weekly": 10, "Class_Regularity": 2.0
}
→ Prédiction: À RISQUE (Probabilité: ~35%)
```

## 🚀 Déploiement sur Render

### Configuration Simple

1. **Créer un compte** sur [render.com](https://render.com)
2. **Nouveau Web Service** → Connecter votre dépôt GitHub
3. **Configuration :**
   - **Build Command** : `pip install -r requirements.txt`
   - **Start Command** : `uvicorn app:app --host 0.0.0.0 --port 10000`
4. **Variables d'environnement** : Aucune nécessaire
5. **Déployer** → Votre application est en ligne !

### Fichiers Nécessaires

- `app.py` (votre backend)
- `requirements.txt` (dépendances)
- `static/index.html` (frontend)
- Optionnel : `Procfile` pour configuration avancée

## 🔍 Dépannage

| Problème | Solution |
|----------|----------|
| **Port déjà utilisé** | Changer le port : `--port 8080` |
| **Module non trouvé** | Vérifier installation : `pip list` |
| **CORS errors** | Le backend inclut déjà CORS middleware |
| **Page blanche** | Vérifier console navigateur (F12) |
| **Slow response** | Hébergement gratuit Render peut être lent au premier chargement |

## 📈 Améliorations Futures

- [ ] **Authentification** pour sauvegarde des profils
- [ ] **Historique** des prédictions passées
- [ ] **Export PDF** des résultats
- [ ] **API mobile** pour applications natives
- [ ] **Dashboard admin** pour statistiques

## 🤝 Contribution

Les contributions sont bienvenues ! Processus :

1. **Fork** le projet
2. **Branche feature** (`git checkout -b feature/AmazingFeature`)
3. **Commit** (`git commit -m 'Add AmazingFeature'`)
4. **Push** (`git push origin feature/AmazingFeature`)
5. **Pull Request**

## 📄 Licence

Ce projet est sous licence **Personnelle**.  
Utilisation libre pour projets éducatifs et non-commerciaux.

```
Copyright © 2024 DATAIKÔS

Droit d'utilisation accordé pour :
- Projets académiques et éducatifs
- Recherche et développement
- Démonstrations non-commerciales

Interdiction de :
- Usage commercial sans autorisation
- Redistribution modifiée sans attribution
```

## 👥 Équipe DATAIKÔS

**Fait avec ❤️ par :**
- **David** 
- **Faysal** 
- **Prudencia** 
- **Randy** 
- **Armstrong** 

## 📞 Contact & Support

- ** Application** : [https://student-prediction-interface.onrender.com](https://student-prediction-interface.onrender.com)
- ** Email** : meilleurd2001@gmail.com
- ** Issues** : [GitHub Issues](https://github.com/monsieurMechant200/dataikos/issues)

---


###  **Prêt à découvrir vos chances de réussite ?**

[![Tester Maintenant](https://img.shields.io/badge/🚀_Tester_l'Application-00D4FF?style=for-the-badge&logo=rocket&logoColor=white)](https://student-prediction-interface.onrender.com/static/index.html)
[![Documentation API](https://img.shields.io/badge/📚_Documentation_API-8A2BE2?style=for-the-badge&logo=readthedocs&logoColor=white)](https://student-prediction-interface.onrender.com/docs)

*"Les données éclairent le chemin, mais c'est ta détermination qui trace la route."*  
**— L'équipe DATAIKÔS**

</div>

---


**✨ Ensemble, faisons de la réussite étudiante une science prédictive !**

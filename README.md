# 💼 Job Recommendation System – Recommandation intelligente d’offres d’emploi

## 📌 Présentation du projet

Ce projet consiste à développer un **système intelligent de recommandation d’offres d’emploi**, basé sur l’analyse sémantique des annonces et des préférences de l’utilisateur.  
L’objectif est d’aider les candidats à trouver rapidement des offres pertinentes en fonction de leurs compétences, localisation, domaine et mots-clés.

Le système repose sur des techniques de **Web Scraping**, de **NLP (Traitement du Langage Naturel)** et de **similarité sémantique**, et propose une interface interactive via **Streamlit**.

---

## 🎯 Objectifs

- Collecter automatiquement des offres d’emploi depuis le web
- Nettoyer et structurer les données textuelles
- Analyser les descriptions d’offres avec des techniques NLP
- Recommander des offres pertinentes selon les critères utilisateur
- Fournir une interface simple, interactive et intuitive

---

## 🏗️ Architecture Générale

Le pipeline du projet est structuré comme suit :

1. **Web Scraping des offres d’emploi**
2. **Stockage et structuration des données**
3. **Prétraitement NLP**
4. **Vectorisation et similarité sémantique**
5. **Calcul de score de pertinence**
6. **Visualisation et interaction via Streamlit**

---

## 📥 Collecte des Données

Les données sont collectées à partir de sites d’offres d’emploi via **Web Scraping**.

### Informations extraites :
- Titre du poste
- Entreprise
- Ville / localisation
- Description du poste
- Compétences requises
- Date de publication
- Lien vers l’offre

Les données sont ensuite stockées sous forme de **DataFrame (CSV / Pandas)** pour traitement.

---

## 🧹 Prétraitement des Données

Les étapes de prétraitement incluent :

- Nettoyage du texte (ponctuation, caractères spéciaux, stopwords)
- Normalisation (minuscules, espaces)
- Fusion des champs textuels (titre + description)
- Suppression des doublons et valeurs manquantes

---

## 🧠 Méthodologie de Recommandation

### 🔹 Vectorisation sémantique
- Utilisation de **SentenceTransformer**
- Transformation des descriptions d’offres en embeddings vectoriels

### 🔹 Similarité
- Calcul de similarité cosinus entre :
  - Les préférences de l’utilisateur
  - Les offres d’emploi disponibles

### 🔹 Score de pertinence
Le score final est basé sur :
- Correspondance sémantique
- Ville sélectionnée
- Domaine / mots-clés
- Filtres choisis par l’utilisateur

Les offres sont ensuite classées par ordre décroissant de pertinence.

---

## 📊 Interface Utilisateur (Streamlit)

L’application permet à l’utilisateur de :

- Sélectionner une **ville**
- Entrer des **mots-clés ou compétences**
- Filtrer par **domaine**
- Visualiser les offres recommandées avec un **score (%)**
- Explorer les détails de chaque offre (lien direct)

---

## 🛠️ Outils & Technologies

| Catégorie | Technologies |
|---------|--------------|
| Langage | Python |
| Web Scraping | BeautifulSoup, Requests |
| Manipulation des données | Pandas, NumPy |
| NLP & Embeddings | SentenceTransformer |
| Similarité | Cosine Similarity |
| Interface | Streamlit |
| Visualisation | Streamlit UI |
| Environnement | Jupyter Notebook, VS Code |

---

## 🧪 Résultats

- Recommandations personnalisées selon le profil utilisateur
- Classement dynamique des offres par score de pertinence
- Amélioration de la recherche d’emploi par approche sémantique
- Interface fluide et facile à utiliser

---

## 🚀 Améliorations Futures

- Intégration du **profil CV utilisateur**
- Ajout de **modèles NLP plus avancés**
- Recommandation en **temps réel**
- Analyse de compétences manquantes
- Déploiement cloud (AWS / GCP)
- Support multilingue (Français / Anglais / Arabe)

---

## 👩‍💻 Réalisé par

- **Maryam Sakouti**
- **Nadia Lahrouri**


---

## 🎓 Contexte Académique

Projet réalisé dans le cadre d’un **stage / projet académique (PFE)**  
Domaine : **Data Analytics & Intelligence Artificielle**

---

## 📄 Licence

Projet à but académique et pédagogique.

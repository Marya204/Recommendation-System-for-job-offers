import streamlit as st
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import fitz
from datetime import datetime
import re
import time

# Configuration
st.set_page_config(
    page_title="💼 Job Recommender Maroc",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CHARGEMENT DES DONNÉES
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("/content/drive/MyDrive/tous_les_jobs_maroc_pretraite.csv")
        df = df.fillna('')
        df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
        
        embeddings = np.load("/content/drive/MyDrive/embeddings_jobs.npy")
        embeddings_tensor = torch.from_numpy(embeddings).float()
        
        return df, embeddings_tensor
    except Exception as e:
        st.error(f"❌ Erreur chargement données: {e}")
        return pd.DataFrame(), torch.tensor([])

@st.cache_resource
def load_model():
    try:
        return SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    except Exception as e:
        st.error(f"❌ Erreur chargement modèle: {e}")
        return None

# FONCTIONS UTILITAIRES
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_keywords(text, min_length=3):
    words = clean_text(text).split()
    stopwords_fr = {'de', 'la', 'le', 'et', 'à', 'en', 'un', 'une', 'des', 'du', 'pour', 'dans', 'avec', 'sur'}
    return [w for w in words if len(w) >= min_length and w not in stopwords_fr]

# NORMALISATION DES VILLES
def normalize_city(city):
    """Normaliser les noms de villes pour un meilleur matching"""
    if pd.isna(city) or str(city).strip() == "":
        return "non_specifie"
    
    city = str(city).lower().strip()
    
    # Mapping des variantes vers la forme canonique
    city_mapping = {
        'casablanca': ['casa', 'casablanca', 'casablanca-settat', 'grand casablanca'],
        'rabat': ['rabat', 'rabat-sale', 'rabat-salé-kenitra', 'rabat salé'],
        'marrakech': ['marrakech', 'marrakesh', 'marrakech-safi'],
        'fes': ['fès', 'fes', 'fès-meknès'],
        'meknes': ['meknès', 'meknes'],
        'tanger': ['tanger', 'tangier', 'tanger-tetouan'],
        'agadir': ['agadir', 'agadir-ida'],
        'oujda': ['oujda', 'oujda-angad'],
        'kenitra': ['kenitra', 'kénitra'],
        'tetouan': ['tétouan', 'tetouan'],
        'sale': ['salé', 'sale'],
        'mohammedia': ['mohammedia', 'mohammédia'],
        'el_jadida': ['el jadida', 'el-jadida'],
        'beni_mellal': ['beni mellal', 'béni mellal']
    }
    
    # Chercher la forme canonique
    for canonical, variants in city_mapping.items():
        for variant in variants:
            if variant in city:
                return canonical
    
    # Nettoyer les caractères spéciaux
    city = re.sub(r'[^\w\s]', '', city).strip()
    return city if city else "non_specifie"


# FONCTION DE RECHERCHE
def search_jobs_final(query, filters_dict=None, use_strict=True, top_k=30, debug=False):
    
    df, embeddings_tensor = load_data()
    model = load_model()
    
    if model is None or len(df) == 0:
        return []
    
    filters = filters_dict or {}
    nb_filters_actifs = len([v for v in filters.values() if v and str(v).strip()])
    
    if debug:
        st.write(f"🔍 Dataset initial: {len(df)} offres")
        st.write(f"🎯 Nombre de filtres actifs: {nb_filters_actifs}")
    
    # CAS 1: SEULEMENT DES FILTRES (pas de requête texte)
    if (not query or query.strip() == "") and nb_filters_actifs > 0:
        df_filtered = df.copy()
        
        # Appliquer les filtres
        for filter_key, filter_value in filters.items():
            if not filter_value or str(filter_value).strip() == "":
                continue
            
            filter_value_clean = str(filter_value).lower().strip()
            
            if filter_key == 'ville':
                ville_normalized = normalize_city(filter_value_clean)
                df_filtered['_ville_norm'] = df_filtered['ville'].apply(normalize_city)
                mask = df_filtered['_ville_norm'] == ville_normalized
                
                if mask.sum() == 0:
                    mask = df_filtered['ville'].astype(str).str.lower().str.contains(
                        filter_value_clean, na=False, regex=False
                    )
                
                df_filtered = df_filtered[mask]
                if '_ville_norm' in df_filtered.columns:
                    df_filtered.drop(columns=['_ville_norm'], inplace=True)
            
            elif filter_key == 'type_contrat':
                mask = df_filtered['type_contrat'].astype(str).str.lower().str.contains(
                    filter_value_clean, na=False, regex=False
                )
                df_filtered = df_filtered[mask]
            
            elif filter_key == 'experience':
                mask = df_filtered['experience'].astype(str).str.lower().str.contains(
                    filter_value_clean, na=False, regex=False
                )
                df_filtered = df_filtered[mask]
            
            elif filter_key == 'niveau_études':
                mask = df_filtered['niveau_études'].astype(str).str.lower().str.contains(
                    filter_value_clean, na=False, regex=False
                )
                df_filtered = df_filtered[mask]
        
        if len(df_filtered) == 0:
            return []
        
        # Calculer le score basé sur les filtres correspondants
        results = []
        for df_idx in df_filtered.index[:top_k]:
            job = df_filtered.loc[df_idx]
            
            # Calculer le score de correspondance des filtres
            filter_matches = 0
            for filter_key, filter_value in filters.items():
                if not filter_value or str(filter_value).strip() == "":
                    continue
                
                filter_value_clean = str(filter_value).lower().strip()
                job_value = str(job.get(filter_key, "")).lower().strip()
                
                # Correspondance exacte ou partielle
                if filter_key == 'ville':
                    if normalize_city(job_value) == normalize_city(filter_value_clean):
                        filter_matches += 1
                elif filter_value_clean in job_value:
                    filter_matches += 1
            
            # Score = (filtres correspondants / filtres actifs)
            score = filter_matches / nb_filters_actifs if nb_filters_actifs > 0 else 1.0
            
            results.append({
                "index": int(df_idx),
                "titre": job.get("intitulé_poste", "Non spécifié"),
                "entreprise": job.get("entreprise", "Non spécifié"),
                "ville": job.get("ville", "Non spécifié"),
                "contrat": job.get("type_contrat", "Non spécifié"),
                "experience": job.get("experience", "Non spécifié"),
                "niveau_etudes": job.get("niveau_études", "Non spécifié"),
                "date": job.get("publication_date"),
                "description": job.get("description", "")[:200] + "...",
                "lien": job.get("lien_offre", "#"),
                "score": score
            })
        
        # Trier par score décroissant
        results.sort(key=lambda x: x["score"], reverse=True)
        
        if debug:
            st.write(f"✅ {len(results)} résultats filtrés avec scores")
        
        return results[:top_k]
    
    # CAS 2: REQUÊTE TEXTE (avec ou sans filtres)
    df_filtered = df.copy()
    
    # Appliquer les filtres si présents
    if nb_filters_actifs > 0:
        for filter_key, filter_value in filters.items():
            if not filter_value or str(filter_value).strip() == "":
                continue
            
            filter_value_clean = str(filter_value).lower().strip()
            
            if filter_key == 'ville':
                ville_normalized = normalize_city(filter_value_clean)
                df_filtered['_ville_norm'] = df_filtered['ville'].apply(normalize_city)
                mask = df_filtered['_ville_norm'] == ville_normalized
                
                if mask.sum() == 0:
                    mask = df_filtered['ville'].astype(str).str.lower().str.contains(
                        filter_value_clean, na=False, regex=False
                    )
                
                df_filtered = df_filtered[mask]
                if '_ville_norm' in df_filtered.columns:
                    df_filtered.drop(columns=['_ville_norm'], inplace=True)
            
            elif filter_key == 'type_contrat':
                mask = df_filtered['type_contrat'].astype(str).str.lower().str.contains(
                    filter_value_clean, na=False, regex=False
                )
                df_filtered = df_filtered[mask]
            
            elif filter_key == 'experience':
                mask = df_filtered['experience'].astype(str).str.lower().str.contains(
                    filter_value_clean, na=False, regex=False
                )
                df_filtered = df_filtered[mask]
            
            elif filter_key == 'niveau_études':
                mask = df_filtered['niveau_études'].astype(str).str.lower().str.contains(
                    filter_value_clean, na=False, regex=False
                )
                df_filtered = df_filtered[mask]
    
    if len(df_filtered) == 0:
        return []
    
    # Calculer les scores sémantiques
    filtered_indices = df_filtered.index.tolist()
    filtered_embeddings = embeddings_tensor[filtered_indices]
    
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarity_scores = util.cos_sim(query_embedding, filtered_embeddings)[0]
    
    if debug:
        st.write(f"📊 Calcul de similarité sémantique pour: '{query[:50]}...'")
    
    # Construire les résultats
    results = []
    keywords = extract_keywords(query) if use_strict else []
    
    for score_idx, df_idx in enumerate(filtered_indices):
        job = df.loc[df_idx]
        semantic_score = float(similarity_scores[score_idx])
        
        # Seuil minimal
        if semantic_score < 0.15:
            continue
        
        # Vérification mots-clés stricte
        if use_strict and keywords:
            job_text = f"{job.get('intitulé_poste', '')} {job.get('text_clean', '')}"
            job_text_clean = clean_text(job_text)
            
            matches = sum(1 for kw in keywords if kw in job_text_clean)
            if matches < len(keywords) * 0.5:
                continue
        
        # Calculer le score de correspondance des filtres
        filter_score = 1.0
        if nb_filters_actifs > 0:
            filter_matches = 0
            for filter_key, filter_value in filters.items():
                if not filter_value or str(filter_value).strip() == "":
                    continue
                
                filter_value_clean = str(filter_value).lower().strip()
                job_value = str(job.get(filter_key, "")).lower().strip()
                
                if filter_key == 'ville':
                    if normalize_city(job_value) == normalize_city(filter_value_clean):
                        filter_matches += 1
                elif filter_value_clean in job_value:
                    filter_matches += 1
            
            filter_score = filter_matches / nb_filters_actifs
        
        # Score final = moyenne pondérée (60% sémantique + 40% filtres)
        final_score = (0.6 * semantic_score) + (0.4 * filter_score)
        
        results.append({
            "index": int(df_idx),
            "titre": job.get("intitulé_poste", "Non spécifié"),
            "entreprise": job.get("entreprise", "Non spécifié"),
            "ville": job.get("ville", "Non spécifié"),
            "contrat": job.get("type_contrat", "Non spécifié"),
            "experience": job.get("experience", "Non spécifié"),
            "niveau_etudes": job.get("niveau_études", "Non spécifié"),
            "date": job.get("publication_date"),
            "description": job.get("description", "")[:200] + "...",
            "lien": job.get("lien_offre", "#"),
            "score": final_score
        })
    
    # Trier par score décroissant
    results.sort(key=lambda x: x["score"], reverse=True)
    
    if debug and results:
        st.write(f"🎯 {len(results)} résultats avec scores combinés")
        st.write("Top 3 scores:")
        for i, r in enumerate(results[:3]):
            st.write(f"  {i+1}. {r['titre'][:50]}... → Score: {r['score']:.2%}")
    
    return results[:top_k]


# INTERFACE STREAMLIT
def main():
    st.title("💼 Recommandation Intelligente d'Emplois - Maroc")
    
    # Initialisation session state
    if 'filters' not in st.session_state:
        st.session_state.filters = {}
    if 'saved_jobs' not in st.session_state:
        st.session_state.saved_jobs = []
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Navigation")
        menu = st.radio(
            "Choisir une option :",
            ["🔍 Recherche avancée", "📄 Analyser mon CV", "💾 Offres sauvegardées", "📈 Statistiques"]
        )
        
        st.markdown("---")
        st.caption("🔧 Options de debug")
        debug_mode = st.checkbox("Mode Debug", value=False)
    
    # Charger les données
    df, _ = load_data()
    
    # Pages
    if menu == "🔍 Recherche avancée":
        render_search_tab(df, debug_mode)
    elif menu == "📄 Analyser mon CV":
        render_cv_tab()
    elif menu == "💾 Offres sauvegardées":
        render_saved_tab()
    else:
        render_stats_tab(df)

def render_search_tab(df, debug_mode=False):
    st.header("🔍 Recherche avancée d'emplois")
    
    # Zone de recherche principale
    query = st.text_input(
        "🔤 Décrivez le poste recherché",
        placeholder="Ex: développeur python, ingénieur commercial, chef de projet...",
        help="Entrez des mots-clés décrivant le poste que vous recherchez"
    )
    
    st.markdown("---")
    
    # Section Filtres
    st.subheader("🎯 Filtres de recherche")
    
    col1, col2 = st.columns(2)
    
    # Initialiser toutes les variables de sélection
    selected_ville = None
    selected_contrat = None
    selected_exp = None
    selected_niveau = None
    
    # Colonne 1: Ville et Contrat
    with col1:
        st.markdown("**📍 Localisation**")
        if 'ville' in df.columns:
            # Obtenir les villes uniques et les nettoyer
            villes_raw = df['ville'].dropna().astype(str).unique()
            villes_clean = sorted([v for v in villes_raw if v.strip() and v != 'Non spécifiée'])
            villes = [''] + villes_clean
            
            selected_ville = st.selectbox(
                "Ville",
                villes,
                help="Filtrer par ville spécifique",
                key="ville_select"
            )
        
        st.markdown("**📝 Type de contrat**")
        if 'type_contrat' in df.columns:
            contrats = [''] + sorted([c for c in df['type_contrat'].dropna().unique() if str(c).strip() and c != 'Non spécifié'])
            selected_contrat = st.selectbox(
                "Contrat",
                contrats,
                help="CDI, CDD, Stage, etc.",
                key="contrat_select"
            )
    
    # Colonne 2: Expérience et Niveau d'études
    with col2:
        st.markdown("**⚡ Expérience**")
        if 'experience' in df.columns:
            experiences = [''] + sorted([e for e in df['experience'].dropna().unique() if str(e).strip() and e != 'Non spécifié'])
            selected_exp = st.selectbox(
                "Niveau d'expérience",
                experiences,
                help="Débutant, 1-2 ans, 2-5 ans, etc.",
                key="exp_select"
            )
        
        st.markdown("**🎓 Niveau d'études**")
        if 'niveau_études' in df.columns:
            niveaux = [''] + sorted([n for n in df['niveau_études'].dropna().unique() if str(n).strip() and n != 'Non spécifié'])
            selected_niveau = st.selectbox(
                "Diplôme requis",
                niveaux,
                help="BAC, BAC+2, BAC+3, etc.",
                key="niveau_select"
            )
    
    # Boutons d'action pour les filtres
    st.markdown("---")
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("➕ Appliquer les filtres", type="primary", use_container_width=True):
            # Construire le dictionnaire de filtres
            new_filters = {}
            
            if selected_ville:
                new_filters['ville'] = selected_ville
            if selected_contrat:
                new_filters['type_contrat'] = selected_contrat
            if selected_exp:
                new_filters['experience'] = selected_exp
            if selected_niveau:
                new_filters['niveau_études'] = selected_niveau
            
            st.session_state.filters = new_filters
            st.success("✅ Filtres appliqués !")
            time.sleep(0.5)
            st.rerun()
    
    with col_btn2:
        if st.button("🗑️ Réinitialiser les filtres", use_container_width=True):
            st.session_state.filters = {}
            st.info("Filtres réinitialisés")
            time.sleep(0.5)
            st.rerun()
    
    # Afficher les filtres actifs
    if st.session_state.filters:
        st.markdown("---")
        st.subheader("✅ Filtres actifs")
        
        for key, value in st.session_state.filters.items():
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.info(f"**{key.replace('_', ' ').title()}**: {value}")
            with col_b:
                if st.button("❌", key=f"rm_{key}"):
                    del st.session_state.filters[key]
                    st.rerun()
    
    # Options de recherche
    st.markdown("---")
    st.subheader("⚙️ Options de recherche")
    
    col_opt1, col_opt2 = st.columns(2)
    
    with col_opt1:
        use_strict = st.checkbox(
            "🔍 Recherche stricte",
            value=True,
            help="Exige que les résultats contiennent au moins 50% des mots-clés"
        )
    
    with col_opt2:
        top_k = st.slider(
            "📊 Nombre de résultats",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="Nombre maximum de résultats à afficher"
        )
    
    # Bouton de recherche principal
    st.markdown("---")
    
    if st.button("🚀 LANCER LA RECHERCHE", type="primary", use_container_width=True):
        if not query or query.strip() == "":
            st.error("⚠️ Veuillez entrer des mots-clés pour la recherche")
        else:
            with st.spinner("🔄 Recherche en cours..."):
                results = search_jobs_final(
                    query=query,
                    filters_dict=st.session_state.filters,
                    use_strict=use_strict,
                    top_k=top_k,
                    debug=debug_mode
                )
                
                st.session_state.search_results = results
    
    # Afficher les résultats
    if st.session_state.search_results:
        st.markdown("---")
        st.header(f"📋 Résultats ({len(st.session_state.search_results)} offres)")
        
        for idx, job in enumerate(st.session_state.search_results, 1):
            with st.expander(f"**{idx}. {job['titre']}** - {job['entreprise']}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**🏢 Entreprise:** {job['entreprise']}")
                    st.markdown(f"**📍 Ville:** {job['ville']}")
                    st.markdown(f"**📝 Type de contrat:** {job['contrat']}")
                    st.markdown(f"**⚡ Expérience:** {job['experience']}")
                    st.markdown(f"**🎓 Niveau d'études:** {job['niveau_etudes']}")
                    
                    if pd.notna(job['date']):
                        st.markdown(f"**📅 Date de publication:** {job['date'].strftime('%d/%m/%Y')}")
                
                with col2:
                    if st.button("💾 Sauvegarder", key=f"save_{job['index']}"):
                        if job not in st.session_state.saved_jobs:
                            st.session_state.saved_jobs.append(job)
                            st.success("✅ Offre sauvegardée !")
                    
                    if job['lien'] != "#":
                        st.markdown(f"[🔗 Voir l'offre]({job['lien']})")
                
                if job['description']:
                    st.markdown("**📄 Description:**")
                    st.text(job['description'])
    
    elif st.session_state.filters or query:
        st.info("👆 Cliquez sur 'Lancer la recherche' pour voir les résultats")

def render_cv_tab():
    st.header("📄 Analyse de CV")
    
    uploaded_file = st.file_uploader(
        "Téléchargez votre CV (PDF)",
        type=['pdf'],
        help="Uploadez votre CV au format PDF pour obtenir des recommandations personnalisées"
    )
    
    if uploaded_file:
        try:
            # Extraire le texte du PDF
            pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            cv_text = ""
            for page in pdf_document:
                cv_text += page.get_text()
            
            st.success("✅ CV analysé avec succès !")
            
            with st.expander("📄 Aperçu du texte extrait"):
                st.text(cv_text[:1000] + "...")
            
            if st.button("🔍 Rechercher des offres correspondantes"):
                with st.spinner("Analyse en cours..."):
                    results = search_jobs_final(
                        query=cv_text,
                        filters_dict={},
                        use_strict=False,
                        top_k=15
                    )
                    
                    if results:
                        st.success(f"✅ {len(results)} offres trouvées correspondant à votre profil !")
                        
                        # DEBUG: Afficher les scores
                        st.write("🔍 DEBUG - Premiers scores:", [f"{r['score']:.2f}" for r in results[:3]])
                        
                        st.markdown("---")
                        st.header(f"📋 Offres recommandées pour votre profil ({len(results)} résultats)")
                        
                        for idx, job in enumerate(results, 1):
                            score_value = job.get('score', 0)
                            score_percent = int(score_value * 100)
                            
                            # DEBUG: Afficher le score pour chaque job
                            st.write(f"DEBUG Job {idx}: score brut = {score_value}, score % = {score_percent}%")
                            
                            with st.expander(f"**{idx}. {job['titre']}** - {job['entreprise']} | 🎯 Correspondance: {score_percent}%"):
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.markdown(f"**🏢 Entreprise:** {job['entreprise']}")
                                    st.markdown(f"**📍 Ville:** {job['ville']}")
                                    st.markdown(f"**📝 Type de contrat:** {job['contrat']}")
                                    st.markdown(f"**⚡ Expérience:** {job['experience']}")
                                    st.markdown(f"**🎓 Niveau d'études:** {job['niveau_etudes']}")
                                    
                                    if pd.notna(job['date']):
                                        st.markdown(f"**📅 Date de publication:** {job['date'].strftime('%d/%m/%Y')}")
                                
                                with col2:
                                    score_display = int(job.get('score', 0) * 100)
                                    st.metric("Compatibilité CV", f"{score_display}%")
                                    
                                    if st.button("💾 Sauvegarder", key=f"save_cv_{job['index']}"):
                                        if job not in st.session_state.saved_jobs:
                                            st.session_state.saved_jobs.append(job)
                                            st.success("✅ Offre sauvegardée !")
                                    
                                    if job['lien'] != "#":
                                        st.markdown(f"[🔗 Voir l'offre]({job['lien']})")
                                
                                if job['description']:
                                    st.markdown("**📄 Description:**")
                                    st.text(job['description'])
                    else:
                        st.warning("⚠️ Aucune offre ne correspond à votre profil. Essayez d'élargir vos critères.")
        
        except Exception as e:
            st.error(f"❌ Erreur lors de l'analyse du CV: {e}")
    else:
        st.info("📤 Uploadez votre CV pour commencer l'analyse")

def render_saved_tab():
    st.header("💾 Offres sauvegardées")
    
    if 'saved_jobs' in st.session_state and st.session_state.saved_jobs:
        st.success(f"📊 {len(st.session_state.saved_jobs)} offre(s) sauvegardée(s)")
        
        for idx, job in enumerate(st.session_state.saved_jobs, 1):
            with st.expander(f"{idx}. {job['titre']} - {job['entreprise']}"):
                st.markdown(f"**📍 Ville:** {job['ville']}")
                st.markdown(f"**📝 Contrat:** {job['contrat']}")
                st.markdown(f"**⚡ Expérience:** {job['experience']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if job['lien'] != "#":
                        st.markdown(f"[🔗 Voir l'offre]({job['lien']})")
                
                with col2:
                    if st.button("🗑️ Supprimer", key=f"del_{idx}"):
                        st.session_state.saved_jobs.pop(idx - 1)
                        st.rerun()
        
        if st.button("🗑️ Tout supprimer"):
            st.session_state.saved_jobs = []
            st.rerun()
    else:
        st.info("📭 Aucune offre sauvegardée pour le moment")
        st.markdown("💡 Astuce: Sauvegardez des offres depuis la page de recherche pour les retrouver ici !")

def render_stats_tab(df):
    st.header("📈 Statistiques du dataset")
    
    if not df.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Total d'offres", len(df))
        
        with col2:
            st.metric("🏢 Entreprises", df['entreprise'].nunique())
        
        with col3:
            st.metric("🏙️ Villes", df['ville'].nunique())
        
        st.markdown("---")
        
        # Top villes
        st.subheader("🏙️ Top 10 des villes")
        if 'ville' in df.columns:
            ville_counts = df['ville'].value_counts().head(10)
            st.bar_chart(ville_counts)
        
        # Top contrats
        st.subheader("📝 Répartition des types de contrats")
        if 'type_contrat' in df.columns:
            contrat_counts = df['type_contrat'].value_counts()
            st.bar_chart(contrat_counts)
        
    else:
        st.error("❌ Aucune donnée disponible")

if __name__ == "__main__":
    main()
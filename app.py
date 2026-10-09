import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

from xgboost import XGBClassifier

# ---------------------------
# Config Streamlit
# ---------------------------
st.set_page_config(page_title="Médicaments → Effets Secondaires", layout="wide")
st.title("💊 Prédiction d'Effets Secondaires par Médicaments")


# ---------------------------
# Classe FixedBlendingPredictor (identique à votre pipeline)
# ---------------------------
class FixedBlendingPredictor:
    def __init__(self):
        self.base_models = {}
        self.meta_model = None
        self.feature_names = None
        self.top_drugs = None
        self.top_reactions = None
        self.per_label_thresholds = None

    def create_base_models(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from xgboost import XGBClassifier
        from sklearn.multiclass import OneVsRestClassifier

        self.base_models = {
            'xgb1': OneVsRestClassifier(
                XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, random_state=42), n_jobs=-1),
            'rf1': OneVsRestClassifier(
                RandomForestClassifier(n_estimators=100, max_depth=15, random_state=44, n_jobs=-1), n_jobs=-1),
            'lr1': OneVsRestClassifier(
                LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=1000, random_state=46), n_jobs=-1),
        }

    def train_base_models(self, X_train, y_train):
        for name, model in self.base_models.items():
            try:
                model.fit(X_train, y_train)
            except Exception as e:
                print(f"Erreur entraînement {name}: {e}")

    def create_meta_features(self, X):
        meta_features = []
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(X)
                meta_features.append(probas)
        return np.hstack(meta_features)

    def train_meta_model(self, X_train, y_train):
        X_meta = self.create_meta_features(X_train)
        self.meta_model = XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=49)
        self.meta_model.fit(X_meta, y_train)

    def predict_proba_blended(self, X):
        X_meta = self.create_meta_features(X)
        return self.meta_model.predict_proba(X_meta)

    def predict(self, X, threshold=0.05, min_predictions=5):
        probabilities = self.predict_proba_blended(X)

        if self.per_label_thresholds is None:
            preds = (probabilities > threshold).astype(int)
        else:
            preds = (probabilities > self.per_label_thresholds[np.newaxis, :]).astype(int)

        if min_predictions is not None and min_predictions > 0:
            n_samples, n_labels = probabilities.shape
            for i in range(n_samples):
                current_pos = int(preds[i].sum())
                if current_pos >= min_predictions:
                    continue
                sorted_idx = np.argsort(-probabilities[i])
                for idx in sorted_idx:
                    if preds[i, idx] == 0:
                        preds[i, idx] = 1
                    if int(preds[i].sum()) >= min_predictions:
                        break

        return preds, probabilities


# ---------------------------
# Chargement du modèle
# ---------------------------
@st.cache_resource
def load_model():
    try:
        # Charger votre modèle entraîné
        model_path = r"blender_final_per_label_thresholds.pkl"

        if not os.path.exists(model_path):
            st.error(f"Fichier modèle non trouvé : {model_path}")
            return None

        saved = joblib.load(model_path)
        blender = saved['predictor']

        st.success("✅ Modèle chargé avec succès !")
        st.info(f"• {len(blender.top_drugs)} médicaments dans le vocabulaire")
        st.info(f"• {len(blender.top_reactions)} effets secondaires possibles")

        return blender

    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None


# Chargement du modèle
blender = load_model()

if blender is None:
    st.stop()


# ---------------------------
# Fonction pour créer les features patient
# ---------------------------
def create_patient_features(drugs_input, age, sex, is_sider=0):
    """Crée les features pour un patient à partir des médicaments"""

    # Nettoyer et normaliser les noms de médicaments
    drugs_cleaned = [drug.upper().strip() for drug in drugs_input]

    # Créer le dictionnaire de features
    features = {}

    # Features pour chaque médicament du vocabulaire
    for drug in blender.top_drugs:
        features[f"drug_{drug}"] = 1 if drug in drugs_cleaned else 0

    # Features démographiques
    features['age'] = age
    features['sex'] = sex
    features['nb_drugs'] = len(drugs_cleaned)
    features['nb_reactions'] = 0  # Inconnu au moment de la prédiction
    features['is_sider'] = is_sider

    # Créer le DataFrame
    patient_df = pd.DataFrame([features])

    # S'assurer que toutes les features sont présentes
    if blender.feature_names:
        for col in blender.feature_names:
            if col not in patient_df.columns:
                patient_df[col] = 0
        patient_df = patient_df[blender.feature_names]

    return patient_df


# ---------------------------
# Interface utilisateur
# ---------------------------
st.sidebar.header("⚙️ Paramètres de prédiction")
MIN_PREDICTIONS = st.sidebar.slider("Nombre minimum de prédictions", 1, 20, 5)
CONFIDENCE_THRESHOLD = st.sidebar.slider("Seuil de confiance (%)", 0, 100, 40) / 100

st.subheader("💊 Entrez les médicaments du patient")

# Saisie des médicaments
medications_input = st.text_area(
    "Médicaments (un par ligne ou séparés par des virgules) :",
    height=100,
    placeholder="Exemple:\nMETFORMIN\nINSULIN GLARGINE\nATORVASTATIN\n\nOu: METFORMIN, INSULIN, ATORVASTATIN"
)

# Informations patient
st.subheader("👤 Informations du patient")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Âge", min_value=0, max_value=120, value=45)
with col2:
    sex = st.selectbox("Sexe", options=[1, 2], format_func=lambda x: "Masculin" if x == 1 else "Féminin")
with col3:
    is_sider = st.selectbox("Source", options=[0, 1], format_func=lambda x: "FAERS" if x == 0 else "SIDER")

# Suggestions de médicaments populaires
st.subheader("💡 Médicaments courants (cliquez pour ajouter)")
popular_drugs = ["METFORMIN", "INSULIN", "ATORVASTATIN", "LISINOPRIL", "LEVOTHYROXINE",
                 "AMLODIPINE", "OMEPRAZOLE", "ALBUTEROL", "GABAPENTIN", "HYDROCHLOROTHIAZIDE"]

cols = st.columns(5)
selected_drugs = []

for i, drug in enumerate(popular_drugs):
    with cols[i % 5]:
        if st.button(f"💊 {drug}", key=f"drug_{i}"):
            if 'selected_medications' not in st.session_state:
                st.session_state.selected_medications = []
            if drug not in st.session_state.selected_medications:
                st.session_state.selected_medications.append(drug)

# Afficher les médicaments sélectionnés
if 'selected_medications' in st.session_state and st.session_state.selected_medications:
    st.write("**Médicaments sélectionnés :**")
    for i, drug in enumerate(st.session_state.selected_medications):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"• {drug}")
        with col2:
            if st.button("❌", key=f"remove_{i}"):
                st.session_state.selected_medications.remove(drug)
                st.rerun()

# Bouton de prédiction
if st.button("🔍 Prédire les effets secondaires", type="primary"):
    # Préparer la liste des médicaments
    all_medications = []

    # Ajouter les médicaments sélectionnés
    if 'selected_medications' in st.session_state:
        all_medications.extend(st.session_state.selected_medications)

    # Ajouter les médicaments saisis manuellement
    if medications_input.strip():
        # Séparer par lignes ou virgules
        manual_meds = [med.strip().upper() for med in medications_input.replace(',', '\n').split('\n') if med.strip()]
        all_medications.extend(manual_meds)

    # Supprimer les doublons
    all_medications = list(dict.fromkeys(all_medications))

    if not all_medications:
        st.warning("Veuillez entrer au moins un médicament !")
    else:
        try:
            with st.spinner("🔬 Analyse en cours..."):
                # Créer les features du patient
                patient_df = create_patient_features(all_medications, age, sex, is_sider)

                # Faire la prédiction
                preds, probas = blender.predict(patient_df, min_predictions=MIN_PREDICTIONS)

                # Traiter les résultats
                results_list = []
                for i, reaction in enumerate(blender.top_reactions):
                    if preds[0, i] == 1:
                        prob = probas[0][i]
                        if prob >= CONFIDENCE_THRESHOLD:
                            confidence = '🔴 HAUTE' if prob > 0.7 else '🟡 MOYENNE' if prob > 0.4 else '🟢 FAIBLE'
                            results_list.append({
                                'reaction': reaction,
                                'probability': float(prob),
                                'confidence': confidence
                            })

                # Afficher les résultats
                st.success(f"✅ Analyse terminée - {len(results_list)} effet(s) secondaire(s) prédit(s)")

                # Informations sur l'entrée
                st.subheader("📋 Informations de l'analyse")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Médicaments analysés", len(all_medications))
                with col2:
                    st.metric("Âge du patient", age)
                with col3:
                    st.metric("Sexe", "Homme" if sex == 1 else "Femme")

                st.write("**Médicaments :**")
                for med in all_medications:
                    st.write(f"• {med}")

                # Afficher les effets secondaires prédits
                if results_list:
                    results_list.sort(key=lambda x: x['probability'], reverse=True)

                    st.subheader("🎯 Effets secondaires prédits")

                    for j, pred in enumerate(results_list, 1):
                        with st.container():
                            cols = st.columns([3, 1, 1])
                            with cols[0]:
                                st.write(f"**{j}. {pred['reaction']}**")
                            with cols[1]:
                                st.metric("Probabilité", f"{pred['probability']:.1%}")
                            with cols[2]:
                                st.write(pred['confidence'])

                            # Barre de progression colorée
                            progress_value = float(pred['probability'])
                            st.progress(progress_value)

                            st.markdown("---")
                else:
                    st.info("ℹ️ Aucun effet secondaire détecté avec le niveau de confiance sélectionné.")

        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction : {e}")

# ---------------------------
# Section informations
# ---------------------------
with st.expander("ℹ️ À propos de ce système"):
    st.write("""
    **Système de prédiction d'effets secondaires par modèle de blending**

    **Fonctionnement :**
    1. **Entrée :** Liste des médicaments pris par le patient
    2. **Traitement :** Le modèle analyse les associations médicaments → effets secondaires
    3. **Sortie :** Liste des effets secondaires prédits avec leurs probabilités

    **Base de données :**
    - **FAERS :** FDA Adverse Event Reporting System
    - **SIDER :** Side Effect Resource
    - **Dataset hybride :** Combinaison des deux sources pour une meilleure couverture

    **Algorithme :**
    - Modèle de blending (combinaison de plusieurs algorithmes)
    - Seuils adaptatifs par effet secondaire
    - Garantie d'un nombre minimum de prédictions
    """)

    if blender and blender.top_reactions is not None:
        st.write(f"**Couverture du modèle :**")
        st.write(f"• {len(blender.top_drugs)} médicaments différents")
        st.write(f"• {len(blender.top_reactions)} effets secondaires détectables")

# ---------------------------
# Section pour réinitialiser
# ---------------------------
with st.sidebar:
    st.markdown("---")
    if st.button("🔄 Réinitialiser la sélection"):
        if 'selected_medications' in st.session_state:
            del st.session_state.selected_medications
        st.rerun()
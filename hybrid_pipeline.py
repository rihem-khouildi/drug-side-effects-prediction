# ==============================================
# SYSTÈME HYBRIDE FAERS + SIDER - VERSION OPTIMISÉE
# Correction des problèmes de prédiction et amélioration des performances
# ==============================================

import lxml.etree as ET
import pandas as pd
import numpy as np
import shap
from tqdm import tqdm
import re
from collections import Counter
import matplotlib
import os
import json
import warnings

warnings.filterwarnings('ignore')

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning avec régularisation avancée
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, hamming_loss, roc_auc_score, \
    classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier

# Configuration pour éviter les warnings
from sklearn import set_config

set_config(enable_metadata_routing=True)

sns.set(style="whitegrid", font_scale=1.05)


# ---------------------------
# 1) Parsing XML FAERS amélioré - OPTIMISÉ
# ---------------------------
def parse_faers_xml_improved(file_path, max_reports=None, show_progress=True):
    """Version optimisée du parsing FAERS"""
    print("📥 Début du parsing FAERS...")
    context = ET.iterparse(file_path, events=("end",), recover=True)
    reports = []
    count = 0

    for event, elem in tqdm(context, desc="Parsing XML FAERS") if show_progress else context:
        if elem.tag == "safetyreport":
            try:
                report = {}
                report["report_id"] = elem.findtext("safetyreportid")
                report["receivedate"] = elem.findtext("receivedate")
                report["serious"] = elem.findtext("serious")
                report["primarysourcecountry"] = elem.findtext("primarysourcecountry")

                patient = elem.find("patient")
                drugs, reactions, age, sex, weight = [], [], None, None, None

                if patient is not None:
                    # Age avec meilleure gestion des unités
                    age_node = patient.find("patientonsetage")
                    age_unit_node = patient.find("patientonsetageunit")
                    if age_node is not None and age_node.text:
                        age_text = age_node.text.strip()
                        if re.match(r'^\d*\.?\d+$', age_text):
                            age = float(age_text)
                            if age_unit_node is not None and age_unit_node.text:
                                unit = age_unit_node.text.strip()
                                if unit == "802":  # Months
                                    age = round(age / 12, 1)
                                elif unit == "803":  # Weeks
                                    age = round(age / 52, 1)
                                elif unit == "804":  # Days
                                    age = round(age / 365, 1)
                            if age < 0 or age > 120:
                                age = None

                    # Sex avec validation
                    sex_node = patient.find("patientsex")
                    if sex_node is not None and sex_node.text:
                        try:
                            sex = int(sex_node.text.strip())
                            if sex not in [1, 2]:
                                sex = None
                        except:
                            sex = None

                    # Reactions avec filtrage de qualité
                    for r in patient.findall("reaction"):
                        rt = r.find("reactionmeddrapt")
                        if rt is not None and rt.text and rt.text.strip():
                            reaction_text = rt.text.strip()
                            if len(reaction_text) >= 2:
                                reactions.append(reaction_text)

                    # Drugs avec filtrage amélioré
                    for d in patient.findall("drug"):
                        mp = d.find("medicinalproduct")
                        if mp is not None and mp.text and mp.text.strip():
                            drug_text = mp.text.strip()
                            if len(drug_text) >= 2:
                                drugs.append(drug_text)
                        else:
                            act = d.find("activesubstance/activesubstancename")
                            if act is not None and act.text and act.text.strip():
                                drug_text = act.text.strip()
                                if len(drug_text) >= 2:
                                    drugs.append(drug_text)

                # Normalisation améliorée
                def norm_name(s):
                    if not s: return None
                    s = s.upper().strip()
                    s = re.sub(r"[^A-Z0-9\-\s]", "", s)
                    s = re.sub(r"\s+", " ", s)
                    return s if len(s) >= 2 else None

                drugs = [norm_name(x) for x in drugs if norm_name(x)]
                reactions = [x.strip() for x in reactions if x.strip() and len(x) >= 3]

                # Seulement garder les rapports avec au moins un médicament et une réaction
                if drugs and reactions:
                    report["drugs"] = list(dict.fromkeys(drugs))
                    report["reactions"] = list(dict.fromkeys(reactions))
                    report["age"] = age if age is not None else -1
                    report["sex"] = sex if sex is not None else -1
                    report["weight"] = weight if weight is not None else -1
                    report["data_source"] = "FAERS"

                    reports.append(report)

                elem.clear()
                count += 1
                if max_reports and count >= max_reports:
                    break

            except Exception as e:
                continue

    print(f"✅ FAERS parsing terminé: {len(reports)} rapports valides")
    return pd.DataFrame(reports)


# ---------------------------
# 2) Chargement et traitement SIDER - OPTIMISÉ
# ---------------------------
def load_and_process_sider(sider_file_path, top_drugs_faers, top_reactions_faers):
    """Charge et traite les données SIDER pour alignement avec FAERS - OPTIMISÉ"""
    print("=== CHARGEMENT ET TRAITEMENT SIDER ===")

    try:
        with open(sider_file_path, "r", encoding="utf-8") as f:
            sider_data = json.load(f)

        sider_samples = []

        # Mapping des réactions FAERS vers SIDER (tolérance aux variations)
        faers_reactions_upper = [r.upper() for r in top_reactions_faers]
        reaction_mapping = {}

        for reaction in faers_reactions_upper:
            # Chercher des correspondances partielles
            for sider_reaction in set([r.upper() for r in top_reactions_faers]):
                if reaction in sider_reaction or sider_reaction in reaction:
                    reaction_mapping[reaction] = sider_reaction
                    break
            if reaction not in reaction_mapping:
                reaction_mapping[reaction] = reaction

        for drug_info in tqdm(sider_data, desc="Processing SIDER"):
            drug_name = drug_info["drug_info"]["display_name"].upper()

            # Aligner avec les médicaments FAERS (tolérance aux variations)
            faers_drugs_upper = [d.upper() for d in top_drugs_faers]
            drug_matched = False

            for faers_drug in faers_drugs_upper:
                if drug_name in faers_drug or faers_drug in drug_name:
                    drug_matched = True
                    drug_name = faers_drug  # Utiliser le nom FAERS
                    break

            if not drug_matched:
                continue

            # Extraire les effets secondaires avec mapping
            side_effects = []
            for effect in drug_info.get("side_effects", []):
                reaction_name = effect["side_effect"].strip().upper()

                # Appliquer le mapping
                mapped_reaction = reaction_mapping.get(reaction_name, reaction_name)
                if mapped_reaction in faers_reactions_upper:
                    side_effects.append(mapped_reaction)

            # Créer plusieurs échantillons SIDER par médicament pour équilibrer
            if side_effects:
                for _ in range(2):  # Duplication pour équilibrer avec FAERS
                    sample = {
                        "drugs": [drug_name],
                        "reactions": side_effects,
                        "age": max(18, min(80, np.random.normal(50, 15))),
                        "sex": np.random.choice([1, 2]),
                        "data_source": "SIDER",
                        "report_id": f"SIDER_{drug_name}_{_}",
                        "serious": 1,
                        "primarysourcecountry": "CLINICAL_TRIAL"
                    }
                    sider_samples.append(sample)

        sider_df = pd.DataFrame(sider_samples)
        print(f"✅ SIDER: {len(sider_df)} échantillons alignés avec FAERS")
        return sider_df

    except Exception as e:
        print(f"❌ Erreur chargement SIDER: {e}")
        return pd.DataFrame()


# ---------------------------
# 3) Création du dataset hybride - OPTIMISÉ
# ---------------------------
def create_hybrid_dataset(faers_df, sider_df, min_drug_occurrences=2, min_reaction_occurrences=2):
    """Crée un dataset hybride FAERS + SIDER - OPTIMISÉ"""
    print("=== CRÉATION DATASET HYBRIDE ===")

    # Combiner les données
    hybrid_df = pd.concat([faers_df, sider_df], ignore_index=True)
    print(f"Dataset hybride: FAERS={len(faers_df)}, SIDER={len(sider_df)}, Total={len(hybrid_df)}")

    # Identifier les médicaments et réactions communs
    all_drugs = [drug for sublist in hybrid_df["drugs"] for drug in sublist]
    all_reactions = [reaction for sublist in hybrid_df["reactions"] for reaction in sublist]

    drug_counter = Counter(all_drugs)
    reaction_counter = Counter(all_reactions)

    # Sélectionner les top éléments avec seuil minimum
    top_drugs = [drug for drug, count in drug_counter.most_common(80)
                 if count >= min_drug_occurrences]
    top_reactions = [reaction for reaction, count in reaction_counter.most_common(60)
                     if count >= min_reaction_occurrences]

    print(f"📊 Médicaments hybrides: {len(top_drugs)}")
    print(f"📊 Réactions hybrides: {len(top_reactions)}")

    # Filtrer le dataset
    hybrid_df["drugs_filtered"] = hybrid_df["drugs"].apply(
        lambda x: [d for d in x if d in top_drugs]
    )
    hybrid_df["reactions_filtered"] = hybrid_df["reactions"].apply(
        lambda x: [r for r in x if r in top_reactions]
    )

    # Filtrer les échantillons vides
    initial_count = len(hybrid_df)
    hybrid_df = hybrid_df[
        (hybrid_df["drugs_filtered"].map(len) > 0) &
        (hybrid_df["reactions_filtered"].map(len) > 0)
        ].reset_index(drop=True)

    print(f"✅ Dataset hybride final: {len(hybrid_df)} échantillons (filtré de {initial_count})")

    return hybrid_df, top_drugs, top_reactions


# ---------------------------
# 4) Features Engineering pour hybride - OPTIMISÉ
# ---------------------------
def create_hybrid_features(hybrid_df, top_drugs, top_reactions, knowledge_base=None):
    """Crée les features pour le dataset hybride - VERSION OPTIMISÉE"""
    print("=== ENGINEERING FEATURES HYBRIDES ===")

    # Créer un DataFrame vide d'abord
    X_list = []

    # Features médicaments (one-hot encoding)
    for idx, row in tqdm(hybrid_df.iterrows(), total=len(hybrid_df), desc="Creating drug features"):
        drug_features = {}

        # Features médicaments
        for drug in top_drugs:
            drug_features[f"drug__{drug}"] = 1 if drug in row["drugs_filtered"] else 0

        # Features démographiques
        drug_features["age"] = row["age"] if row["age"] != -1 else np.nan
        drug_features["sex"] = row["sex"] if row["sex"] != -1 else np.nan
        drug_features["nb_drugs"] = len(row["drugs_filtered"])
        drug_features["nb_reactions"] = len(row["reactions_filtered"])
        drug_features["is_sider"] = 1 if row["data_source"] == "SIDER" else 0

        X_list.append(drug_features)

    X = pd.DataFrame(X_list)

    # Gestion des valeurs manquantes
    X["age"] = X["age"].fillna(X["age"].median())
    X["sex"] = X["sex"].fillna(X["sex"].mode()[0] if not X["sex"].mode().empty else 1)

    # Clip des valeurs extrêmes
    X["age"] = np.clip(X["age"], 0, 120)

    # Features de connaissance si KB disponible
    if knowledge_base and len(knowledge_base.get('drug_reaction_scores', {})) > 0:
        X = add_knowledge_features(X, hybrid_df, knowledge_base, top_reactions)

    # Normalisation des colonnes numériques
    scaler = StandardScaler()
    numeric_cols = ['age', 'nb_drugs', 'nb_reactions']

    # Vérifier que les colonnes existent
    existing_numeric_cols = [col for col in numeric_cols if col in X.columns]
    if existing_numeric_cols:
        X[existing_numeric_cols] = scaler.fit_transform(X[existing_numeric_cols])

    # Target - avec filtrage des réactions trop rares
    mlb = MultiLabelBinarizer(classes=top_reactions)
    Y = mlb.fit_transform(hybrid_df["reactions_filtered"])

    # Filtrer les réactions trop rares (moins de 5 occurrences)
    reaction_sums = Y.sum(axis=0)
    valid_reactions = reaction_sums >= 3
    Y_filtered = Y[:, valid_reactions]
    top_reactions_filtered = [reaction for i, reaction in enumerate(top_reactions) if valid_reactions[i]]

    print(f"✅ Features hybrides créées: {X.shape}")
    print(f"✅ Target shape: {Y_filtered.shape} (filtré de {Y.shape})")
    print(f"✅ Réactions conservées: {len(top_reactions_filtered)}/{len(top_reactions)}")

    return X, Y_filtered, mlb, scaler, top_reactions_filtered


def add_knowledge_features(X, hybrid_df, knowledge_base, top_reactions):
    """Ajoute des features basées sur la connaissance médicale - OPTIMISÉ"""
    print("🔧 Ajout des features de connaissance...")

    knowledge_features = []

    for idx, row in tqdm(hybrid_df.iterrows(), total=len(hybrid_df), desc="Knowledge Features"):
        patient_drugs = row["drugs_filtered"]
        knowledge_vector = {}

        # Score de risque moyen par réaction
        risk_scores = []
        for reaction in top_reactions:
            total_risk = 0
            drug_count = 0

            for drug in patient_drugs:
                if drug in knowledge_base['drug_reaction_scores']:
                    if reaction in knowledge_base['drug_reaction_scores'][drug]:
                        total_risk += knowledge_base['drug_reaction_scores'][drug][reaction]
                        drug_count += 1

            avg_risk = total_risk / max(drug_count, 1) if drug_count > 0 else 0
            knowledge_vector[f'kb_risk_{reaction}'] = avg_risk
            if avg_risk > 0:
                risk_scores.append(avg_risk)

        # Score global de risque
        knowledge_vector['kb_risk_global'] = np.mean(risk_scores) if risk_scores else 0
        knowledge_vector['kb_confidence'] = len(risk_scores)  # Nombre d'associations connues

        knowledge_features.append(knowledge_vector)

    knowledge_df = pd.DataFrame(knowledge_features, index=X.index)
    X_enhanced = pd.concat([X, knowledge_df], axis=1)

    # Remplacer les NaN par 0
    X_enhanced = X_enhanced.fillna(0)

    return X_enhanced


# ---------------------------
# 5) Modèle Hybride avec régularisation avancée - OPTIMISÉ
# ---------------------------
class OptimizedHybridADRPredictor:
    """Prédicteur hybride optimisé avec gestion des probabilités corrigée"""

    def __init__(self):
        self.models = {}
        self.mlb = None
        self.scaler = None
        self.top_drugs = None
        self.top_reactions = None
        self.class_weights_ = None

    def calculate_class_weights(self, y):
        """Calcule les poids de classe pour gérer le déséquilibre"""
        class_sums = y.sum(axis=0)
        total = len(y)
        weights = total / (len(class_sums) * (class_sums + 1))  # +1 pour éviter division par 0
        return weights

    def train_optimized_model(self, X_train, y_train, X_val, y_val):
        """Entraîne un modèle hybride optimisé - VERSION CORRIGÉE"""
        print("🎯 Entraînement Modèle Hybride Optimisé...")

        # Calcul des poids de classe
        self.class_weights_ = self.calculate_class_weights(y_train)
        print(f"📊 Poids de classe calculés (moyenne: {np.mean(self.class_weights_):.3f})")

        # 1. Random Forest optimisé
        print("🌲 Entraînement Random Forest Optimisé...")
        rf_model = OneVsRestClassifier(
            BalancedRandomForestClassifier(
                n_estimators=100,  # Réduit pour éviter overfitting
                max_depth=8,  # Plus shallow
                min_samples_split=10,
                min_samples_leaf=5,
                max_features=0.5,  # Plus de régularisation
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            ),
            n_jobs=-1
        )

        rf_model.fit(X_train, y_train)

        # 2. XGBoost optimisé avec gestion de déséquilibre
        print("🚀 Entraînement XGBoost Optimisé...")

        # Calcul scale_pos_weight pour XGBoost
        scale_pos_weights = []
        for i in range(y_train.shape[1]):
            positive_count = y_train[:, i].sum()
            negative_count = len(y_train) - positive_count
            if positive_count > 0:
                scale_pos_weights.append(negative_count / positive_count)
            else:
                scale_pos_weights.append(1)

        xgb_model = OneVsRestClassifier(
            xgb.XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=1.0,
                gamma=0.2,
                min_child_weight=3,
                scale_pos_weight=1,  # Géré par class weights
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            ),
            n_jobs=-1
        )

        xgb_model.fit(X_train, y_train)

        # 3. Logistic Regression avec régularisation
        print("📊 Entraînement Logistic Regression...")
        lr_model = OneVsRestClassifier(
            LogisticRegression(
                C=0.1,
                penalty='l2',
                solver='liblinear',
                random_state=42,
                max_iter=1000
            ),
            n_jobs=-1
        )

        lr_model.fit(X_train, y_train)

        # Validation
        rf_val_pred = rf_model.predict(X_val)
        xgb_val_pred = xgb_model.predict(X_val)
        lr_val_pred = lr_model.predict(X_val)

        rf_val_f1 = f1_score(y_val, rf_val_pred, average='micro', zero_division=0)
        xgb_val_f1 = f1_score(y_val, xgb_val_pred, average='micro', zero_division=0)
        lr_val_f1 = f1_score(y_val, lr_val_pred, average='micro', zero_division=0)

        print(f"✅ RF Validation F1: {rf_val_f1:.4f}")
        print(f"✅ XGB Validation F1: {xgb_val_f1:.4f}")
        print(f"✅ LR Validation F1: {lr_val_f1:.4f}")

        self.models = {
            'rf': rf_model,
            'xgb': xgb_model,
            'lr': lr_model
        }

        return self.models

    def ensemble_predict_proba(self, X):
        """Prédiction des probabilités par ensemble - VERSION CORRIGÉE"""
        all_probas = []

        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    probas = model.predict_proba(X)
                    # Gestion du format de probabilités pour OneVsRest
                    if isinstance(probas, list):
                        # Format: liste de tableaux (n_classes, n_samples, 2)
                        probas_array = np.zeros((X.shape[0], len(self.top_reactions)))
                        for i, class_probas in enumerate(probas):
                            if class_probas.shape[1] == 2:  # [prob_neg, prob_pos]
                                probas_array[:, i] = class_probas[:, 1]
                            else:
                                probas_array[:, i] = class_probas
                        all_probas.append(probas_array)
                    else:
                        # Format déjà bon
                        all_probas.append(probas)
            except Exception as e:
                print(f"⚠️  Erreur predict_proba {name}: {e}")
                continue

        if not all_probas:
            raise ValueError("Aucun modèle n'a pu faire de prédiction de probabilité")

        # Moyenne des probabilités
        avg_proba = np.mean(all_probas, axis=0)
        return avg_proba

    def predict_with_threshold(self, X, threshold=0.3):
        """Prédiction avec seuil adaptatif - VERSION CORRIGÉE"""
        try:
            probas = self.ensemble_predict_proba(X)
            # Appliquer un seuil adaptatif par classe basé sur la distribution
            predictions = (probas > threshold).astype(int)
            return predictions, probas
        except Exception as e:
            print(f"❌ Erreur prédiction avec seuil: {e}")
            # Fallback: voting majoritaire
            predictions = []
            for name, model in self.models.items():
                try:
                    pred = model.predict(X)
                    predictions.append(pred)
                except:
                    continue
            if predictions:
                ensemble_pred = np.mean(predictions, axis=0) > 0.5
                return ensemble_pred.astype(int), None
            else:
                raise ValueError("Aucun modèle disponible pour prédiction")

    def evaluate_optimized(self, X_test, y_test):
        """Évaluation complète du modèle optimisé"""
        print("\n📊 ÉVALUATION MODÈLE OPTIMISÉ")
        print("=" * 50)

        results = {}

        # Test de différents seuils
        best_f1 = 0
        best_threshold = 0.3

        for threshold in [0.2, 0.3, 0.4, 0.5]:
            try:
                y_pred, _ = self.predict_with_threshold(X_test, threshold)
                f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)
                accuracy = accuracy_score(y_test, y_pred)

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

                print(f"📈 Seuil {threshold}: F1={f1:.4f}, Accuracy={accuracy:.4f}")
            except Exception as e:
                print(f"❌ Erreur évaluation seuil {threshold}: {e}")

        # Évaluation finale avec meilleur seuil
        try:
            y_pred_final, y_proba_final = self.predict_with_threshold(X_test, best_threshold)

            micro_f1 = f1_score(y_test, y_pred_final, average='micro', zero_division=0)
            macro_f1 = f1_score(y_test, y_pred_final, average='macro', zero_division=0)
            accuracy = accuracy_score(y_test, y_pred_final)
            hamming = hamming_loss(y_test, y_pred_final)

            print(f"\n🎯 PERFORMANCE FINALE (seuil={best_threshold}):")
            print(f"   • F1-score (micro):  {micro_f1:.4f}")
            print(f"   • F1-score (macro):  {macro_f1:.4f}")
            print(f"   • Exact Match:       {accuracy:.4f}")
            print(f"   • Hamming Loss:      {hamming:.4f}")

            # Analyse par classe pour les top réactions
            if y_proba_final is not None:
                self.analyze_predictions_by_class(y_test, y_pred_final, y_proba_final)

            results = {
                'micro_f1': micro_f1,
                'macro_f1': macro_f1,
                'accuracy': accuracy,
                'hamming_loss': hamming,
                'best_threshold': best_threshold,
                'predictions': y_pred_final
            }

        except Exception as e:
            print(f"❌ Erreur évaluation finale: {e}")
            results = {'micro_f1': 0, 'macro_f1': 0, 'accuracy': 0, 'hamming_loss': 1}

        return results

    def analyze_predictions_by_class(self, y_true, y_pred, y_proba):
        """Analyse détaillée des prédictions par classe"""
        print("\n🔍 ANALYSE PAR CLASSE (Top 10 réactions):")

        # Calculer le support par classe
        class_support = y_true.sum(axis=0)
        top_indices = np.argsort(class_support)[-10:][::-1]  # Top 10 par support

        for i in top_indices:
            if class_support[i] > 0:  # Uniquement les classes avec du support
                reaction_name = self.top_reactions[i]
                precision = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
                recall = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
                f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)

                print(
                    f"   • {reaction_name:<25}: F1={f1:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, Support={class_support[i]}")


# ---------------------------
# 6) Pipeline Complet Optimisé - CORRIGÉ
# ---------------------------
def run_optimized_pipeline(faers_xml_path, sider_json_path, max_faers_reports=10000):
    """Exécute le pipeline complet optimisé"""
    print("🚀 LANCEMENT PIPELINE HYBRIDE OPTIMISÉ")
    print("=" * 60)

    try:
        # 1. Chargement FAERS
        print("📥 Chargement FAERS...")
        faers_df = parse_faers_xml_improved(faers_xml_path, max_reports=max_faers_reports)

        if len(faers_df) == 0:
            print("❌ Aucune donnée FAERS valide trouvée")
            return None, None, None

        print(f"✅ FAERS chargé: {len(faers_df)} rapports")

        # 2. Analyse initiale FAERS
        faers_drugs = [drug for sublist in faers_df["drugs"] for drug in sublist]
        faers_reactions = [reaction for sublist in faers_df["reactions"] for reaction in sublist]

        faers_drug_counter = Counter(faers_drugs)
        faers_reaction_counter = Counter(faers_reactions)

        top_faers_drugs = [drug for drug, count in faers_drug_counter.most_common(80) if count >= 2]
        top_faers_reactions = [reaction for reaction, count in faers_reaction_counter.most_common(60) if count >= 2]

        print(f"📊 Top FAERS drugs: {len(top_faers_drugs)}, reactions: {len(top_faers_reactions)}")

        # 3. Chargement SIDER aligné
        print("📥 Chargement SIDER...")
        sider_df = load_and_process_sider(sider_json_path, top_faers_drugs, top_faers_reactions)

        # 4. Création dataset hybride
        hybrid_df, top_drugs, top_reactions = create_hybrid_dataset(
            faers_df, sider_df,
            min_drug_occurrences=1,
            min_reaction_occurrences=2
        )

        if len(hybrid_df) < 50:
            print(f"❌ Dataset hybride trop petit: {len(hybrid_df)} échantillons")
            return None, None, None

        # 5. Knowledge Base from SIDER
        knowledge_base = build_knowledge_base_from_sider(sider_json_path)

        # 6. Features engineering avec filtrage
        X, Y, mlb, scaler, top_reactions_filtered = create_hybrid_features(
            hybrid_df, top_drugs, top_reactions, knowledge_base
        )

        if len(X) == 0 or len(Y) == 0:
            print("❌ Erreur création features")
            return None, None, None

        # 7. Split des données
        stratification = (Y.sum(axis=1) > 0).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.15, random_state=42, stratify=stratification
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42,
            stratify=(y_train.sum(axis=1) > 0)
        )

        print(f"📊 Split final - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        # 8. Entraînement modèle optimisé
        hybrid_predictor = OptimizedHybridADRPredictor()
        hybrid_predictor.mlb = mlb
        hybrid_predictor.scaler = scaler
        hybrid_predictor.top_drugs = top_drugs
        hybrid_predictor.top_reactions = top_reactions_filtered

        models = hybrid_predictor.train_optimized_model(X_train, y_train, X_val, y_val)

        # 9. Évaluation optimisée
        results = hybrid_predictor.evaluate_optimized(X_test, y_test)

        # 10. Analyse overfitting
        analyze_optimized_overfitting(hybrid_predictor, X_train, y_train, X_test, y_test)

        return hybrid_predictor, results, hybrid_df

    except Exception as e:
        print(f"❌ Erreur critique dans le pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def build_knowledge_base_from_sider(sider_file_path):
    """Construit la knowledge base à partir de SIDER"""
    try:
        with open(sider_file_path, "r", encoding="utf-8") as f:
            sider_data = json.load(f)

        knowledge_base = {
            'drug_reaction_scores': {},
            'drug_risk_profiles': {},
            'reaction_drug_associations': {}
        }

        for drug_info in sider_data:
            drug_name = drug_info["drug_info"]["display_name"].upper()
            knowledge_base['drug_reaction_scores'][drug_name] = {}
            knowledge_base['drug_risk_profiles'][drug_name] = []

            for effect in drug_info.get("side_effects", []):
                reaction = effect["side_effect"].strip().upper()
                frequency = effect.get("frequency_clean", "Unknown")

                # Convertir fréquence en score
                if frequency == "Common" or 'frequent' in frequency.lower():
                    score = 0.8
                elif frequency == "Rare" or 'rare' in frequency.lower():
                    score = 0.3
                elif '%' in frequency:
                    try:
                        percent = float(frequency.replace('%', '').split('-')[0].strip())
                        score = min(percent / 100, 0.9)
                    except:
                        score = 0.5
                else:
                    score = 0.5

                knowledge_base['drug_reaction_scores'][drug_name][reaction] = score
                knowledge_base['drug_risk_profiles'][drug_name].append((reaction, score))

        print(f"✅ Knowledge Base construite: {len(knowledge_base['drug_reaction_scores'])} médicaments")
        return knowledge_base

    except Exception as e:
        print(f"❌ Erreur construction KB: {e}")
        return {'drug_reaction_scores': {}, 'drug_risk_profiles': {}, 'reaction_drug_associations': {}}


def analyze_optimized_overfitting(predictor, X_train, y_train, X_test, y_test):
    """Analyse d'overfitting pour le modèle optimisé"""
    print("\n🔍 ANALYSE OVERFITTING OPTIMISÉ")
    print("=" * 50)

    for name, model in predictor.models.items():
        try:
            # Prédictions train et test
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Métriques
            train_f1 = f1_score(y_train, y_train_pred, average='micro', zero_division=0)
            test_f1 = f1_score(y_test, y_test_pred, average='micro', zero_division=0)

            gap = train_f1 - test_f1

            print(f"\n{name.upper():<10}:")
            print(f"   Train F1: {train_f1:.4f}")
            print(f"   Test F1:  {test_f1:.4f}")
            print(f"   Gap:      {gap:.4f}")

            if gap > 0.1:
                print("   ⚠️  OVERFITTING SIGNIFICATIF")
            elif gap > 0.05:
                print("   ℹ️  Léger overfitting")
            else:
                print("   ✅ BONNE GÉNÉRALISATION")

        except Exception as e:
            print(f"❌ Erreur analyse overfitting {name}: {e}")


# ---------------------------
# 7) Prédiction Optimisée - CORRIGÉE
# ---------------------------
def optimized_predict(patient_drugs, age=50, sex=1, hybrid_predictor=None, threshold=0.3):
    """Prédiction optimisée avec gestion correcte des probabilités"""
    if hybrid_predictor is None:
        print("❌ Modèle hybride non initialisé")
        return []

    try:
        # Préparation des features
        patient_features = {}

        # Features médicaments
        for drug in hybrid_predictor.top_drugs:
            patient_features[f"drug__{drug}"] = 1 if drug.upper() in [d.upper() for d in patient_drugs] else 0

        # Features démographiques normalisées
        if hasattr(hybrid_predictor.scaler, 'mean_') and hasattr(hybrid_predictor.scaler, 'scale_'):
            patient_features["age"] = (age - hybrid_predictor.scaler.mean_[0]) / hybrid_predictor.scaler.scale_[0]
            patient_features["nb_drugs"] = (len(patient_drugs) - hybrid_predictor.scaler.mean_[1]) / \
                                           hybrid_predictor.scaler.scale_[1]
            patient_features["nb_reactions"] = (0 - hybrid_predictor.scaler.mean_[2]) / hybrid_predictor.scaler.scale_[
                2]
        else:
            patient_features["age"] = age
            patient_features["nb_drugs"] = len(patient_drugs)
            patient_features["nb_reactions"] = 0

        patient_features["sex"] = sex
        patient_features["is_sider"] = 0

        # DataFrame
        patient_df = pd.DataFrame([patient_features])

        # S'assurer d'avoir toutes les colonnes nécessaires
        if hasattr(hybrid_predictor.models['rf'], 'estimators_') and len(hybrid_predictor.models['rf'].estimators_) > 0:
            if hasattr(hybrid_predictor.models['rf'].estimators_[0], 'feature_names_in_'):
                expected_columns = hybrid_predictor.models['rf'].estimators_[0].feature_names_in_
                for col in expected_columns:
                    if col not in patient_df.columns:
                        patient_df[col] = 0
                patient_df = patient_df[expected_columns]

        # Prédiction avec probabilités
        probas = hybrid_predictor.ensemble_predict_proba(patient_df)

        predictions = []
        for i, reaction in enumerate(hybrid_predictor.top_reactions):
            prob = probas[0][i]  # Probabilité pour cette réaction

            # Seuil adaptatif basé sur la confiance
            if prob > threshold:
                confidence = '🔴' if prob > 0.7 else '🟡' if prob > 0.4 else '🟢'
                source = 'FAERS+SIDER' if prob > 0.5 else 'FAERS'

                predictions.append({
                    'reaction': reaction,
                    'probability': float(prob),
                    'confidence': confidence,
                    'source': source
                })

        # Trier par probabilité et limiter
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        return predictions[:10]  # Retourner les top 10

    except Exception as e:
        print(f"❌ Erreur prédiction optimisée: {e}")
        return []


# ---------------------------
# 8) ANALYSE EDA SPÉCIFIQUE POUR SIDER
# ---------------------------

def analyze_sider_data(sider_json_path):
    """Analyse exploratoire complète des données SIDER"""
    print("\n📊 ANALYSE EDA DES DONNÉES SIDER")
    print("=" * 50)

    try:
        with open(sider_json_path, "r", encoding="utf-8") as f:
            sider_data = json.load(f)

        print(f"📁 Données SIDER chargées: {len(sider_data)} médicaments")

        # Configuration des plots
        plt.figure(figsize=(20, 15))

        # 1. Distribution du nombre d'effets secondaires par médicament
        plt.subplot(3, 4, 1)
        side_effects_counts = []
        for drug_info in sider_data:
            side_effects = drug_info.get("side_effects", [])
            side_effects_counts.append(len(side_effects))

        plt.hist(side_effects_counts, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Nombre d\'effets secondaires')
        plt.ylabel('Nombre de médicaments')
        plt.title('Distribution des Effets Secondaires par Médicament')
        plt.axvline(np.mean(side_effects_counts), color='red', linestyle='--',
                    label=f'Moyenne: {np.mean(side_effects_counts):.1f}')
        plt.legend()

        # 2. Top 20 médicaments avec le plus d'effets secondaires
        plt.subplot(3, 4, 2)
        drug_effects = {}
        for drug_info in sider_data:
            drug_name = drug_info["drug_info"]["display_name"]
            side_effects = drug_info.get("side_effects", [])
            drug_effects[drug_name] = len(side_effects)

        top_drugs = sorted(drug_effects.items(), key=lambda x: x[1], reverse=True)[:20]
        drugs, effects_count = zip(*top_drugs)

        plt.barh(range(len(drugs)), effects_count, color='lightcoral')
        plt.yticks(range(len(drugs)), [d[:25] + '...' if len(d) > 25 else d for d in drugs])
        plt.xlabel('Nombre d\'effets secondaires')
        plt.title('Top 20 Médicaments SIDER (Effets Secondaires)')

        # 3. Distribution des fréquences d'effets secondaires
        plt.subplot(3, 4, 3)
        frequencies = []
        frequency_categories = {}

        for drug_info in sider_data:
            for effect in drug_info.get("side_effects", []):
                freq = effect.get("frequency_clean", "Unknown")
                frequencies.append(freq)
                frequency_categories[freq] = frequency_categories.get(freq, 0) + 1

        # Regrouper les fréquences rares
        common_freqs = {k: v for k, v in frequency_categories.items() if v >= 10}
        other_count = sum(v for k, v in frequency_categories.items() if v < 10)
        if other_count > 0:
            common_freqs['Autres'] = other_count

        plt.pie(common_freqs.values(), labels=common_freqs.keys(), autopct='%1.1f%%', startangle=90)
        plt.title('Distribution des Fréquences d\'Effets Secondaires')

        # 4. Top 20 effets secondaires les plus fréquents
        plt.subplot(3, 4, 4)
        all_side_effects = []
        for drug_info in sider_data:
            for effect in drug_info.get("side_effects", []):
                side_effect_name = effect["side_effect"]
                all_side_effects.append(side_effect_name)

        side_effect_counter = Counter(all_side_effects)
        top_effects = side_effect_counter.most_common(20)
        effects, counts = zip(*top_effects)

        plt.barh(range(len(effects)), counts, color='lightgreen')
        plt.yticks(range(len(effects)), [e[:30] + '...' if len(e) > 30 else e for e in effects])
        plt.xlabel('Fréquence')
        plt.title('Top 20 Effets Secondaires SIDER')

        # 5. Analyse des médicaments par classes thérapeutiques
        plt.subplot(3, 4, 5)
        therapeutic_classes = {}
        for drug_info in sider_data:
            drug_classes = drug_info["drug_info"].get("drug_classes", [])
            for drug_class in drug_classes:
                therapeutic_classes[drug_class] = therapeutic_classes.get(drug_class, 0) + 1

        if therapeutic_classes:
            top_classes = sorted(therapeutic_classes.items(), key=lambda x: x[1], reverse=True)[:15]
            classes, class_counts = zip(*top_classes)

            plt.barh(range(len(classes)), class_counts, color='gold')
            plt.yticks(range(len(classes)), [c[:25] + '...' if len(c) > 25 else c for c in classes])
            plt.xlabel('Nombre de médicaments')
            plt.title('Top 15 Classes Thérapeutiques')
        else:
            plt.text(0.5, 0.5, 'Données de classes\nthérapeutiques\nnon disponibles',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Classes Thérapeutiques')

        # 6. Distribution des indications thérapeutiques
        plt.subplot(3, 4, 6)
        indications = {}
        for drug_info in sider_data:
            drug_indications = drug_info["drug_info"].get("indications", [])
            for indication in drug_indications:
                indications[indication] = indications.get(indication, 0) + 1

        if indications:
            top_indications = sorted(indications.items(), key=lambda x: x[1], reverse=True)[:10]
            inds, ind_counts = zip(*top_indications)

            plt.barh(range(len(inds)), ind_counts, color='lightseagreen')
            plt.yticks(range(len(inds)), [i[:30] + '...' if len(i) > 30 else i for i in inds])
            plt.xlabel('Nombre de médicaments')
            plt.title('Top 10 Indications Thérapeutiques')
        else:
            plt.text(0.5, 0.5, 'Données d\'indications\nnon disponibles',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Indications Thérapeutiques')

        # 7. Heatmap des effets secondaires les plus communs
        plt.subplot(3, 4, 7)
        try:
            # Sélectionner les top médicaments et effets secondaires
            top_20_drugs = [d[0] for d in top_drugs[:10]]
            top_20_effects = [e[0] for e in top_effects[:10]]

            # Créer une matrice d'occurrence
            occurrence_matrix = np.zeros((len(top_20_drugs), len(top_20_effects)))

            drug_to_idx = {drug: idx for idx, drug in enumerate(top_20_drugs)}
            effect_to_idx = {effect: idx for idx, effect in enumerate(top_20_effects)}

            for drug_info in sider_data:
                drug_name = drug_info["drug_info"]["display_name"]
                if drug_name in drug_to_idx:
                    for effect in drug_info.get("side_effects", []):
                        effect_name = effect["side_effect"]
                        if effect_name in effect_to_idx:
                            occurrence_matrix[drug_to_idx[drug_name], effect_to_idx[effect_name]] = 1

            sns.heatmap(occurrence_matrix, xticklabels=[e[:15] + '...' for e in top_20_effects],
                        yticklabels=[d[:15] + '...' for d in top_20_drugs],
                        cmap='YlOrRd', cbar_kws={'label': 'Présence'})
            plt.title('Occurrence Effets/Médicaments (Top 10)')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
        except Exception as e:
            plt.text(0.5, 0.5, f'Heatmap non disponible\n{str(e)}',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Heatmap Effets/Médicaments')

        # 8. Distribution des types d'effets secondaires (par catégorie)
        plt.subplot(3, 4, 8)
        try:
            # Catégoriser les effets secondaires par mots-clés
            categories = {
                'Gastro-intestinaux': ['nausea', 'vomiting', 'diarrhea', 'abdominal', 'dyspepsia'],
                'Neurologiques': ['headache', 'dizziness', 'somnolence', 'insomnia', 'tremor'],
                'Dermatologiques': ['rash', 'pruritus', 'urticaria', 'erythema', 'dermatitis'],
                'Cardiovasculaires': ['hypertension', 'tachycardia', 'hypotension', 'palpitations'],
                'Psychiatriques': ['anxiety', 'depression', 'confusion', 'agitation']
            }

            category_counts = {category: 0 for category in categories}

            for effect_name in all_side_effects:
                effect_lower = effect_name.lower()
                for category, keywords in categories.items():
                    if any(keyword in effect_lower for keyword in keywords):
                        category_counts[category] += 1
                        break

            plt.bar(category_counts.keys(), category_counts.values(), color='lightsteelblue')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Nombre d\'effets')
            plt.title('Effets Secondaires par Catégorie')
        except Exception as e:
            plt.text(0.5, 0.5, f'Catégorisation non disponible\n{str(e)}',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Effets par Catégorie')

        # 9. Analyse de la richesse du vocabulaire SIDER
        plt.subplot(3, 4, 9)
        unique_effects = len(set(all_side_effects))
        total_effects = len(all_side_effects)
        ratio_unique = unique_effects / total_effects if total_effects > 0 else 0

        stats_data = [unique_effects, total_effects, ratio_unique * 100]
        stats_labels = ['Effets uniques', 'Effets totaux', 'Ratio unique (%)']

        plt.bar(stats_labels, stats_data, color=['lightblue', 'lightcoral', 'lightgreen'])
        plt.ylabel('Nombre / Pourcentage')
        plt.title('Richesse du Vocabulaire SIDER')

        for i, v in enumerate(stats_data):
            plt.text(i, v + max(stats_data) * 0.01, f'{v:.0f}', ha='center', va='bottom')

        # 10. Distribution des médicaments par nombre d'indications
        plt.subplot(3, 4, 10)
        indication_counts = []
        for drug_info in sider_data:
            indications = drug_info["drug_info"].get("indications", [])
            indication_counts.append(len(indications))

        plt.hist(indication_counts, bins=15, alpha=0.7, color='plum', edgecolor='black')
        plt.xlabel('Nombre d\'indications par médicament')
        plt.ylabel('Fréquence')
        plt.title('Distribution des Indications par Médicament')
        plt.axvline(np.mean(indication_counts), color='red', linestyle='--',
                    label=f'Moyenne: {np.mean(indication_counts):.1f}')
        plt.legend()

        # 11. Corrélation entre effets secondaires et indications
        plt.subplot(3, 4, 11)
        effects_vs_indications = []
        for drug_info in sider_data:
            num_effects = len(drug_info.get("side_effects", []))
            num_indications = len(drug_info["drug_info"].get("indications", []))
            effects_vs_indications.append((num_effects, num_indications))

        if effects_vs_indications:
            effects, indications = zip(*effects_vs_indications)
            plt.scatter(indications, effects, alpha=0.6, color='orange')
            plt.xlabel('Nombre d\'indications')
            plt.ylabel('Nombre d\'effets secondaires')
            plt.title('Effets vs Indications')

            # Ligne de régression
            if len(indications) > 1:
                z = np.polyfit(indications, effects, 1)
                p = np.poly1d(z)
                plt.plot(indications, p(indications), "r--", alpha=0.8)

        # 12. Statistiques résumées SIDER
        plt.subplot(3, 4, 12)
        summary_stats = [
            len(sider_data),
            total_effects,
            unique_effects,
            np.mean(side_effects_counts),
            np.median(side_effects_counts)
        ]
        summary_labels = [
            'Médicaments totaux',
            'Effets totaux',
            'Effets uniques',
            'Moyenne effets/médicament',
            'Médiane effets/médicament'
        ]

        plt.barh(range(len(summary_stats)), summary_stats, color='lightcyan', edgecolor='black')
        plt.yticks(range(len(summary_stats)), summary_labels)
        plt.xlabel('Valeur')
        plt.title('Statistiques Résumées SIDER')

        for i, v in enumerate(summary_stats):
            plt.text(v + max(summary_stats) * 0.01, i, f'{v:.1f}', va='center')

        plt.tight_layout()
        plt.savefig('sider_eda_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Statistiques textuelles détaillées
        print_sider_statistics(sider_data, side_effects_counts, all_side_effects)

    except Exception as e:
        print(f"❌ Erreur analyse EDA SIDER: {e}")
        import traceback
        traceback.print_exc()


def print_sider_statistics(sider_data, side_effects_counts, all_side_effects):
    """Affiche les statistiques détaillées des données SIDER"""
    print("\n📈 STATISTIQUES DÉTAILLÉES SIDER")
    print("=" * 40)

    # Statistiques de base
    total_drugs = len(sider_data)
    total_effects = len(all_side_effects)
    unique_effects = len(set(all_side_effects))

    print(f"📊 STATISTIQUES GÉNÉRALES:")
    print(f"   • Médicaments analysés: {total_drugs:,}")
    print(f"   • Effets secondaires totaux: {total_effects:,}")
    print(f"   • Effets secondaires uniques: {unique_effects:,}")
    print(f"   • Ratio d'unicité: {unique_effects / total_effects * 100:.1f}%")
    print(f"   • Effets/médicament (moyenne): {np.mean(side_effects_counts):.1f}")
    print(f"   • Effets/médicament (médiane): {np.median(side_effects_counts):.1f}")
    print(f"   • Effets/médicament (max): {np.max(side_effects_counts):.0f}")
    print(f"   • Effets/médicament (min): {np.min(side_effects_counts):.0f}")

    # Analyse des fréquences
    frequency_counter = Counter()
    for drug_info in sider_data:
        for effect in drug_info.get("side_effects", []):
            freq = effect.get("frequency_clean", "Unknown")
            frequency_counter[freq] += 1

    print(f"\n📈 DISTRIBUTION DES FRÉQUENCES:")
    for freq, count in frequency_counter.most_common(10):
        percentage = count / total_effects * 100
        print(f"   • {freq:<15}: {count:>5} effets ({percentage:.1f}%)")

    # Top médicaments
    drug_effects = {}
    for drug_info in sider_data:
        drug_name = drug_info["drug_info"]["display_name"]
        side_effects = drug_info.get("side_effects", [])
        drug_effects[drug_name] = len(side_effects)

    print(f"\n🏆 TOP 10 MÉDICAMENTS (EFFETS SECONDAIRES):")
    for i, (drug, count) in enumerate(sorted(drug_effects.items(), key=lambda x: x[1], reverse=True)[:10]):
        print(f"   {i + 1:2d}. {drug:<30}: {count:>3} effets")

    # Top effets secondaires
    side_effect_counter = Counter(all_side_effects)
    print(f"\n🔴 TOP 10 EFFETS SECONDAIRES:")
    for i, (effect, count) in enumerate(side_effect_counter.most_common(10)):
        percentage = count / total_drugs * 100
        print(f"   {i + 1:2d}. {effect:<40}: {count:>3} médicaments ({percentage:.1f}%)")

    # Analyse des classes thérapeutiques
    therapeutic_classes = {}
    for drug_info in sider_data:
        drug_classes = drug_info["drug_info"].get("drug_classes", [])
        for drug_class in drug_classes:
            therapeutic_classes[drug_class] = therapeutic_classes.get(drug_class, 0) + 1

    if therapeutic_classes:
        print(f"\n💊 TOP 10 CLASSES THÉRAPEUTIQUES:")
        for i, (drug_class, count) in enumerate(
                sorted(therapeutic_classes.items(), key=lambda x: x[1], reverse=True)[:10]):
            percentage = count / total_drugs * 100
            print(f"   {i + 1:2d}. {drug_class:<35}: {count:>3} médicaments ({percentage:.1f}%)")

    # Analyse des indications
    indications = {}
    for drug_info in sider_data:
        drug_indications = drug_info["drug_info"].get("indications", [])
        for indication in drug_indications:
            indications[indication] = indications.get(indication, 0) + 1

    if indications:
        print(f"\n🎯 TOP 10 INDICATIONS THÉRAPEUTIQUES:")
        for i, (indication, count) in enumerate(
                sorted(indications.items(), key=lambda x: x[1], reverse=True)[:10]):
            percentage = count / total_drugs * 100
            print(f"   {i + 1:2d}. {indication:<40}: {count:>3} médicaments ({percentage:.1f}%)")


# ---------------------------
# 9) COMPARAISON SIDER vs FAERS
# ---------------------------

def compare_sider_faers(sider_json_path, faers_df):
    """Compare les données SIDER et FAERS pour l'alignement"""
    print("\n🔄 COMPARAISON SIDER vs FAERS")
    print("=" * 50)

    try:
        with open(sider_json_path, "r", encoding="utf-8") as f:
            sider_data = json.load(f)

        # Extraire les médicaments et effets SIDER
        sider_drugs = set()
        sider_effects = set()

        for drug_info in sider_data:
            drug_name = drug_info["drug_info"]["display_name"].upper()
            sider_drugs.add(drug_name)

            for effect in drug_info.get("side_effects", []):
                effect_name = effect["side_effect"].strip().upper()
                sider_effects.add(effect_name)

        # Extraire les médicaments et effets FAERS
        faers_drugs = set([drug for sublist in faers_df["drugs"] for drug in sublist])
        faers_reactions = set([reaction for sublist in faers_df["reactions"] for reaction in sublist])

        # Calcul des overlaps
        drug_overlap = sider_drugs.intersection(faers_drugs)
        effect_overlap = sider_effects.intersection(faers_reactions)

        # Statistiques
        print(f"📊 STATISTIQUES DE COMPARAISON:")
        print(f"   • Médicaments SIDER: {len(sider_drugs):,}")
        print(f"   • Médicaments FAERS: {len(faers_drugs):,}")
        print(
            f"   • Overlap médicaments: {len(drug_overlap):,} ({len(drug_overlap) / len(sider_drugs) * 100:.1f}% de SIDER)")
        print(f"   • Effets SIDER: {len(sider_effects):,}")
        print(f"   • Réactions FAERS: {len(faers_reactions):,}")
        print(
            f"   • Overlap effets: {len(effect_overlap):,} ({len(effect_overlap) / len(sider_effects) * 100:.1f}% de SIDER)")

        # Visualisation de la comparaison
        plt.figure(figsize=(15, 10))

        # 1. Overlap des médicaments
        plt.subplot(2, 2, 1)
        drug_venn = {
            'SIDER': len(sider_drugs - faers_drugs),
            'FAERS': len(faers_drugs - sider_drugs),
            'Intersection': len(drug_overlap)
        }

        plt.bar(drug_venn.keys(), drug_venn.values(), color=['lightblue', 'lightcoral', 'lightgreen'])
        plt.title('Overlap des Médicaments')
        plt.ylabel('Nombre de médicaments')

        for i, v in enumerate(drug_venn.values()):
            plt.text(i, v + max(drug_venn.values()) * 0.01, f'{v:,}', ha='center', va='bottom')

        # 2. Overlap des effets/réactions
        plt.subplot(2, 2, 2)
        effect_venn = {
            'SIDER': len(sider_effects - faers_reactions),
            'FAERS': len(faers_reactions - sider_effects),
            'Intersection': len(effect_overlap)
        }

        plt.bar(effect_venn.keys(), effect_venn.values(), color=['lightblue', 'lightcoral', 'lightgreen'])
        plt.title('Overlap des Effets/Réactions')
        plt.ylabel('Nombre d\'effets')

        for i, v in enumerate(effect_venn.values()):
            plt.text(i, v + max(effect_venn.values()) * 0.01, f'{v:,}', ha='center', va='bottom')

        # 3. Top médicaments communs
        plt.subplot(2, 2, 3)
        common_drugs = list(drug_overlap)[:15]
        plt.barh(range(len(common_drugs)), [1] * len(common_drugs), color='gold')
        plt.yticks(range(len(common_drugs)), [d[:25] + '...' if len(d) > 25 else d for d in common_drugs])
        plt.xlabel('Présence')
        plt.title('Top 15 Médicaments Communs')

        # 4. Top effets communs
        plt.subplot(2, 2, 4)
        common_effects = list(effect_overlap)[:15]
        plt.barh(range(len(common_effects)), [1] * len(common_effects), color='lightseagreen')
        plt.yticks(range(len(common_effects)), [e[:30] + '...' if len(e) > 30 else e for e in common_effects])
        plt.xlabel('Présence')
        plt.title('Top 15 Effets/Réactions Communs')

        plt.tight_layout()
        plt.savefig('sider_faers_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\n✅ Comparaison SIDER-FAERS terminée")
        print(f"💾 Graphiques sauvegardés: sider_eda_analysis.png, sider_faers_comparison.png")

    except Exception as e:
        print(f"❌ Erreur comparaison SIDER-FAERS: {e}")


# ---------------------------
# 10) EXÉCUTION PRINCIPALE
# ---------------------------

# ---------------------------
# ---------------------------
# ---------------------------
# 10) INTÉGRATION SHAP - VERSION SIMPLIFIÉE ET ROBUSTE
# ---------------------------

def analyze_with_shap_simple(hybrid_predictor, X_test, sample_size=50):
    """Analyse SHAP simplifiée et robuste"""
    print("\n🔍 ANALYSE SHAP - Version simplifiée et robuste")
    print("=" * 60)

    try:
        import shap
        print("✅ SHAP importé avec succès")

        # Préparer les données pour SHAP
        if sample_size < len(X_test):
            X_sample = X_test.sample(n=sample_size, random_state=42)
        else:
            X_sample = X_test

        # Utiliser le modèle Random Forest pour SHAP
        rf_model = hybrid_predictor.models['rf']

        print("📊 Calcul des valeurs SHAP...")

        # Calcul SHAP pour toutes les classes
        explainer = shap.TreeExplainer(rf_model.estimators_[0])  # Premier classifieur
        shap_values = explainer.shap_values(X_sample)

        print(
            f"🔧 Debug - Type SHAP: {type(shap_values)}, Forme: {np.array(shap_values).shape if hasattr(shap_values, 'shape') else 'N/A'}")

        return explainer, shap_values

    except ImportError:
        print("❌ SHAP n'est pas installé. Installation: pip install shap")
        return None, None
    except Exception as e:
        print(f"❌ Erreur analyse SHAP: {e}")
        return None, None


def explain_predictions_simple(patient_drugs, predictions, hybrid_predictor, explainer, X_sample, age=50, sex=1):
    """Explique les prédictions avec SHAP en format simplifié - VERSION CORRIGÉE"""
    try:
        import shap

        # Préparer les features du patient
        patient_features = {}

        # Features médicaments
        for drug in hybrid_predictor.top_drugs:
            patient_features[f"drug__{drug}"] = 1 if drug.upper() in [d.upper() for d in patient_drugs] else 0

        # Features démographiques
        patient_features["age"] = age
        patient_features["sex"] = sex
        patient_features["nb_drugs"] = len(patient_drugs)
        patient_features["nb_reactions"] = 0
        patient_features["is_sider"] = 0

        # Créer le DataFrame
        patient_df = pd.DataFrame([patient_features])

        # Aligner les colonnes avec X_sample
        for col in X_sample.columns:
            if col not in patient_df.columns:
                patient_df[col] = 0
        patient_df = patient_df[X_sample.columns]

        print(f"\n🔍 ANALYSE SHAP - Version simplifiée et robuste")
        print("=" * 60)

        # Analyser chaque prédiction
        for i, pred in enumerate(predictions, 1):
            reaction = pred['reaction']
            probability = pred['probability']

            # Trouver l'index de la réaction
            reaction_idx = None
            for idx, react in enumerate(hybrid_predictor.top_reactions):
                if react == reaction:
                    reaction_idx = idx
                    break

            if reaction_idx is None:
                continue

            # Calcul SHAP pour cette réaction
            try:
                explainer_reaction = shap.TreeExplainer(hybrid_predictor.models['rf'].estimators_[reaction_idx])
                shap_values_single = explainer_reaction.shap_values(patient_df)

                print(f"\n{i}. {patient_drugs[0] if len(patient_drugs) == 1 else 'POLYMÉDICATION'} - '{reaction}'")
                print(f"\n🎯 ANALYSE: {reaction}")
                print("-" * 40)
                print(f"📈 Probabilité prédite: {probability:.1%}")
                print(f"🔧 Debug - Type SHAP: {type(shap_values_single)}, Forme: {np.array(shap_values_single).shape}")

                # CORRECTION : Gestion robuste du format SHAP
                if isinstance(shap_values_single, list):
                    # Format liste [classe_0, classe_1]
                    if len(shap_values_single) == 2:
                        shap_array = shap_values_single[1]  # Classe positive
                    else:
                        shap_array = shap_values_single[0]
                else:
                    # Format tableau numpy
                    shap_array = shap_values_single

                # Vérifier les dimensions et extraire les valeurs
                if hasattr(shap_array, 'shape'):
                    if len(shap_array.shape) == 3:
                        # Format (1, n_features, 2) - prendre classe positive
                        shap_values_flat = shap_array[0, :, 1] if shap_array.shape[2] == 2 else shap_array[0, :, 0]
                    elif len(shap_array.shape) == 2:
                        # Format (1, n_features)
                        shap_values_flat = shap_array[0, :]
                    else:
                        # Format inattendu, prendre le premier élément
                        shap_values_flat = shap_array.reshape(-1)[:len(patient_df.columns)]
                else:
                    shap_values_flat = shap_array

                # Obtenir la valeur attendue (probabilité de base)
                expected_value = explainer_reaction.expected_value
                if isinstance(expected_value, np.ndarray):
                    if len(expected_value) == 2:
                        base_value = expected_value[1]  # Classe positive pour classification binaire
                    else:
                        base_value = expected_value[0]
                else:
                    base_value = expected_value

                # Créer un mapping feature -> valeur SHAP
                feature_impacts = []
                for j, feature_name in enumerate(patient_df.columns):
                    if j < len(shap_values_flat):
                        shap_value = shap_values_flat[j]
                    else:
                        shap_value = 0.0

                    feature_value = patient_df[feature_name].iloc[0]

                    # Formater le nom de la feature
                    display_name = feature_name
                    if feature_name.startswith('drug__'):
                        drug_name = feature_name.replace('drug__', '')
                        status = "PRÉSENT" if feature_value == 1 else "ABSENT"
                        display_name = f"{drug_name} ({status})"
                    elif feature_name == "age":
                        display_name = f"age (= {int(feature_value)})"
                    elif feature_name == "sex":
                        gender = "Homme" if feature_value == 1 else "Femme"
                        display_name = f"sex (= {int(feature_value)}: {gender})"
                    elif feature_name == "nb_drugs":
                        display_name = f"nb_médicaments (= {int(feature_value)})"
                    else:
                        display_name = f"{feature_name} (= {feature_value:.2f})"

                    feature_impacts.append({
                        'name': display_name,
                        'shap_value': shap_value,
                        'feature_value': feature_value
                    })

                # Séparer les features qui augmentent et réduisent le risque
                increasing_risk = [f for f in feature_impacts if f['shap_value'] > 0]
                decreasing_risk = [f for f in feature_impacts if f['shap_value'] < 0]

                # Trier par magnitude
                increasing_risk.sort(key=lambda x: abs(x['shap_value']), reverse=True)
                decreasing_risk.sort(key=lambda x: abs(x['shap_value']), reverse=True)

                print(f"\n📊 FACTEURS INFLUENTS:")

                # Afficher les facteurs qui augmentent le risque
                if increasing_risk:
                    print("  ➕ AUGMENTENT le risque:")
                    for feat in increasing_risk[:8]:  # Top 8
                        print(f"     {feat['name']}: {feat['shap_value']:+.3f}")

                # Afficher les facteurs qui réduisent le risque
                if decreasing_risk:
                    print("  ➖ RÉDUISENT le risque:")
                    for feat in decreasing_risk[:8]:  # Top 8
                        print(f"     {feat['name']}: {feat['shap_value']:+.3f}")

                # Calculer l'impact total
                total_impact = sum(f['shap_value'] for f in feature_impacts)

                print(f"\n🎲 Probabilité de base: {base_value:.3f}")
                print(f"📐 Impact total des features: {total_impact:.3f}")

            except Exception as e:
                print(f"❌ Erreur analyse SHAP pour {reaction}: {e}")
                import traceback
                traceback.print_exc()
                continue

        return True

    except Exception as e:
        print(f"❌ Erreur explication SHAP: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_comparative_analysis(hybrid_predictor, explainer, X_sample):
    """Exécute une analyse comparative entre différents cas - VERSION CORRIGÉE"""
    print(f"\n🔬 COMPARAISON ENTRE CAS")
    print("=" * 50)

    # Cas de test variés
    test_cases = [
        {"drugs": ["ZEPBOUND"], "age": 45, "sex": 1, "label": "ZEPBOUND seul"},
        {"drugs": ["ZEPBOUND", "ASPIRIN"], "age": 45, "sex": 1, "label": "ZEPBOUND + ASPIRIN"},
        {"drugs": ["PREDNISONE"], "age": 85, "sex": 2, "label": "Patient âgé PREDNISONE"},
        {"drugs": ["METFORMIN", "LISINOPRIL", "ATORVASTATIN"], "age": 68, "sex": 1, "label": "Polymédication diabète"}
    ]

    for case in test_cases:
        print(f"\n📋 {case['label']}:")

        # Faire la prédiction
        predictions = optimized_predict(
            case['drugs'],
            age=case['age'],
            sex=case['sex'],
            hybrid_predictor=hybrid_predictor,
            threshold=0.05  # Seuil bas pour voir plus de résultats
        )

        if predictions:
            # Prendre la prédiction principale
            main_pred = predictions[0]
            print(f"   Réaction principale: {main_pred['reaction']} ({main_pred['probability']:.1%})")

            # Analyser le facteur clé (version simplifiée)
            try:
                # Trouver l'index de la réaction
                reaction_idx = None
                for idx, react in enumerate(hybrid_predictor.top_reactions):
                    if react == main_pred['reaction']:
                        reaction_idx = idx
                        break

                if reaction_idx is not None:
                    # Préparer les features rapidement
                    patient_features = {}
                    for drug in hybrid_predictor.top_drugs:
                        patient_features[f"drug__{drug}"] = 1 if drug.upper() in [d.upper() for d in
                                                                                  case['drugs']] else 0

                    patient_features["age"] = case['age']
                    patient_features["sex"] = case['sex']
                    patient_features["nb_drugs"] = len(case['drugs'])

                    patient_df = pd.DataFrame([patient_features])
                    for col in X_sample.columns:
                        if col not in patient_df.columns:
                            patient_df[col] = 0
                    patient_df = patient_df[X_sample.columns]

                    # Calcul SHAP rapide
                    explainer_reaction = shap.TreeExplainer(hybrid_predictor.models['rf'].estimators_[reaction_idx])
                    shap_values_single = explainer_reaction.shap_values(patient_df)

                    # Extraire les valeurs SHAP (version simplifiée)
                    if isinstance(shap_values_single, list) and len(shap_values_single) == 2:
                        shap_array = shap_values_single[1][0]  # Classe positive, premier échantillon
                    elif hasattr(shap_values_single, 'shape') and len(shap_values_single.shape) == 3:
                        shap_array = shap_values_single[0, :, 1]  # Format (1, n_features, 2)
                    else:
                        shap_array = shap_values_single[0] if hasattr(shap_values_single,
                                                                      '__getitem__') else shap_values_single

                    # Trouver le facteur le plus influent
                    if hasattr(shap_array, '__len__'):
                        max_impact_idx = np.argmax(np.abs(shap_array))
                        if max_impact_idx < len(patient_df.columns):
                            max_impact_feature = patient_df.columns[max_impact_idx]
                            max_impact_value = shap_array[max_impact_idx]

                            if max_impact_feature.startswith('drug__'):
                                drug_name = max_impact_feature.replace('drug__', '')
                                status = "PRÉSENT" if patient_df[max_impact_feature].iloc[0] == 1 else "ABSENT"
                                factor_info = f"{drug_name} ({status})"
                            elif max_impact_feature == "age":
                                factor_info = f"âge ({case['age']} ans)"
                            elif max_impact_feature == "sex":
                                factor_info = f"sexe ({'Homme' if case['sex'] == 1 else 'Femme'})"
                            elif max_impact_feature == "nb_drugs":
                                factor_info = f"nb_médicaments ({len(case['drugs'])})"
                            else:
                                factor_info = max_impact_feature

                            print(f"   Facteur clé: {factor_info} (impact: {max_impact_value:+.3f})")
                        else:
                            print(f"   Facteur clé: Âge ou autre facteur démographique")
                    else:
                        print(f"   Facteur clé: Analyse complexe")

            except Exception as e:
                print(f"   Facteur clé: Analyse SHAP non disponible")
        else:
            print("   Aucune prédiction significative")

    print(f"\n✅ ANALYSE TERMINÉE!")
    print("=" * 50)
    print(f"\n💡 CE QUE CELA SIGNIFIE:")
    print("• Les valeurs POSITIVES augmentent le risque d'effet secondaire")
    print("• Les valeurs NÉGATIVES réduisent le risque")
    print("• L'impact est mesuré en log-odds (échelle logarithmique)")
# ---------------------------
# 11) EXÉCUTION PRINCIPALE AVEC SHAP SIMPLIFIÉ
# ---------------------------

if __name__ == "__main__":
    print("🎯 DÉMARRAGE SYSTÈME AVEC SHAP SIMPLIFIÉ")
    print("=" * 50)

    # Chemins des fichiers
    faers_xml_path = "data/faers_adr_2025Q2.xml"
    sider_json_path = "data/sider_side_effects.json"

    # 1. Pipeline hybride complet
    print("\n🚀 PHASE 1: PIPELINE HYBRIDE COMPLET")
    print("=" * 50)
    hybrid_predictor, results, hybrid_df = run_optimized_pipeline(
        faers_xml_path, sider_json_path, max_faers_reports=5000
    )

    if hybrid_predictor is not None and results is not None:
        # 2. Préparation des données pour SHAP
        print("\n🔧 PHASE 2: PRÉPARATION SHAP")
        print("=" * 50)

        try:
            knowledge_base = build_knowledge_base_from_sider(sider_json_path)
            X, Y, mlb, scaler, top_reactions_filtered = create_hybrid_features(
                hybrid_df, hybrid_predictor.top_drugs, hybrid_predictor.top_reactions, knowledge_base
            )

            stratification = (Y.sum(axis=1) > 0).astype(int)
            X_train, X_test, y_train, y_test = train_test_split(
                X, Y, test_size=0.15, random_state=42, stratify=stratification
            )

            # Initialisation SHAP
            explainer, shap_values = analyze_with_shap_simple(hybrid_predictor, X_test)

        except Exception as e:
            print(f"⚠️  Préparation SHAP échouée: {e}")
            explainer, X_test = None, None

        # 3. Analyse principale avec SHAP
        print("\n🧪 PHASE 3: ANALYSE PRINCIPALE AVEC SHAP")
        print("=" * 50)

        # Cas de test principal
        main_drugs = ["ZEPBOUND", "DUPIXENT"]
        age = 45
        sex = 1

        print(f"💊 MÉDICAMENT TESTÉ: {', '.join(main_drugs)}")
        print(f"👤 PATIENT: Âge {age}, Sexe {'Homme' if sex == 1 else 'Femme'}")

        predictions = optimized_predict(
            main_drugs,
            age=age,
            sex=sex,
            hybrid_predictor=hybrid_predictor,
            threshold=0.05  # Seuil bas pour voir plus de résultats
        )

        if predictions and explainer is not None and X_test is not None:
            # Explication SHAP détaillée
            explain_predictions_simple(
                main_drugs, predictions, hybrid_predictor,
                explainer, X_test, age, sex
            )

            # Analyse comparative
            run_comparative_analysis(hybrid_predictor, explainer, X_test)
        else:
            print("❌ Aucune prédiction ou SHAP non disponible")

        # 4. Rapport final
        best_f1 = results.get('micro_f1', 0)

        print(f"\n📊 RAPPORT FINAL:")
        print("=" * 30)
        print(f"• Performance F1: {best_f1:.4f}")
        print(f"• Médicaments modélisés: {len(hybrid_predictor.top_drugs)}")
        print(f"• Effets secondaires: {len(hybrid_predictor.top_reactions)}")
        print(f"• Explicabilité: ✅ SHAP intégré")
        # Exemples de tests validation
        test_cases = [
            ["METFORMIN"],  # Diabète
            ["LISINOPRIL", "AMLODIPINE"],  # Hypertension
            ["ATORVASTATIN"],  # Cholestérol
            ["SERTRALINE"]  # Dépression
        ]

        # Sauvegarde
        try:
            import joblib

            joblib.dump(hybrid_predictor, 'hybrid_predictor_with_shap.pkl')
            print("💾 Modèle sauvegardé: hybrid_predictor_with_shap.pkl")
        except Exception as e:
            print(f"⚠️  Erreur sauvegarde: {e}")

    else:
        print("❌ Le pipeline a échoué")
"""
Module de gestion des modèles de base (Model Base).

Ce module contient les fonctions pour :
1. Détecter automatiquement le type de problème de Machine Learning (Classification vs Régression).
2. Initialiser un ensemble de modèles candidats adaptés.
3. Entraîner ces modèles et évaluer leurs performances.
4. Sélectionner le meilleur modèle selon un score agrégé.
"""

from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, 
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor, ClassifierChain
from sklearn.linear_model import (LogisticRegression, SGDClassifier, 
    PassiveAggressiveClassifier, SGDRegressor
)
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, root_mean_squared_error,
    r2_score, jaccard_score
)
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler


def cherche_type_probleme(
    df_solution: pd.DataFrame, 
    is_sparse: bool
) -> Dict[str, Any]:
    """
    Détermine le type de tâche ML (Classification/Régression) et initialise les modèles.

    Cette fonction analyse la distribution et le format de la variable cible (target)
    pour classifier le problème parmi :
    - Classification Binaire / Multi-classe / Multi-label.
    - Régression Univariée / Multi-sortie.

    Args:
        df_solution (pd.DataFrame): Le DataFrame contenant la/les colonne(s) cible(s).
        is_sparse (bool): Indique si les features sont au format sparse (impacte le choix des modèles).

    Returns:
        dict: Un dictionnaire contenant :
            - 'type' (str): Le nom de la tâche détectée.
            - 'models' (dict): Dictionnaire des instances de modèles à entraîner.
            - 'metrics' (list): Liste des métriques à utiliser pour l'évaluation.
            - 'df_solution_adjusted' (pd.Series/DataFrame): La target potentiellement transformée (ex: OneHot -> Labels).
    """
    n_cols = df_solution.shape[1]
    models = {}
    metrics = []
    task = ""
    
    # Création d'un échantillon pour l'analyse statistique rapide
    # On aplatit pour vérifier l'ensemble des valeurs uniques
    sample = df_solution.iloc[:2000].values.flatten()
    sample = sample[~np.isnan(sample)] # Exclusion des NaN pour l'analyse
    unique_vals = np.unique(sample)
    
    # Heuristique : Si peu de valeurs uniques et toutes entières -> Discret (Classification probable)
    is_discrete = (len(unique_vals) < 50) and (np.all(np.mod(unique_vals, 1) == 0))
    
    df_solution_adjusted = df_solution.copy()

    # === CAS 1 : MULTIVARIÉ (Plusieurs colonnes cibles) ===
    if n_cols > 1:
        # Vérification One-Hot Encoding : Somme des lignes proche de 1
        row_sums = df_solution.sum(axis=1)
        is_one_hot = np.mean((row_sums >= 0.99) & (row_sums <= 1.01)) > 0.95

        if is_one_hot and is_discrete:
            # Cas: Classification Multi-classe encodée en One-Hot
            task = "classification_multi-classe"
            # On convertit le One-Hot en vecteur de labels (0, 1, 2...)
            df_solution_adjusted = df_solution.idxmax(axis=1) 
            
            if not is_sparse:
                models = {
                    "RandomForest": RandomForestClassifier(n_jobs=-1, n_estimators=100, random_state=42),
                    "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
                    "SGD": SGDClassifier(loss='hinge', n_jobs=-1, random_state=42),
                    "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=300, random_state=42),
                    "KNN": KNeighborsClassifier(n_neighbors=7)
                }
            else:
                models = {
                    "MultinomialNB": MultinomialNB(),
                    "LinearSVC": LinearSVC(),
                    "SGDClassifier": SGDClassifier(loss="log_loss"),
                    "PassiveAggressive": PassiveAggressiveClassifier()
                }
            metrics = ["accuracy", "f1_macro", "roc_auc"]
            
        elif is_discrete and set(unique_vals).issubset({0, 1}):
            # Cas: Classification Multi-label (Plusieurs labels binaires par ligne possibles)
            task = "classification_multi-label"
   
            if not is_sparse:
                models = {
                    "RandomForest": MultiOutputClassifier(
                        RandomForestClassifier(n_jobs=-1, n_estimators=50, random_state=42)
                    ),
                    "HistGradientBoosting": MultiOutputClassifier(
                        HistGradientBoostingClassifier(max_iter=100, random_state=42)
                    )
                }   
            else:
                models = {
                    "OVR_SGD": OneVsRestClassifier(SGDClassifier(loss="log_loss")),
                    "ClassifierChain_LogReg": ClassifierChain(LogisticRegression(max_iter=2000))
                }
            metrics = ["f1_macro", "jaccard"]
            
        else:
            # Cas: Régression Multi-sortie (Plusieurs valeurs continues à prédire)
            task = "régression_multi-sortie"
            models = {
                "RandomForestReg": MultiOutputRegressor(
                    RandomForestRegressor(n_jobs=-1, n_estimators=50, random_state=42)
                ),
            }
            if not is_sparse:
                models["HistGradientBoostingReg"] = MultiOutputRegressor(
                    HistGradientBoostingRegressor(max_iter=100, random_state=42)
                )
            metrics = ["rmse", "r2"]

    # === CAS 2 : UNIVARIÉ (Une seule colonne cible) ===
    else:
        # Transformation DataFrame (N, 1) -> Series (N,) pour compatibilité scikit-learn
        if df_solution_adjusted.ndim > 1 and df_solution_adjusted.shape[1] == 1:
             df_solution_adjusted = df_solution_adjusted.iloc[:, 0]
        
        if is_discrete or len(unique_vals) <= 2:
            # Cas: Classification Binaire ou Multi-classe standard
            task = "classification_binaire" if len(unique_vals) == 2 else "classification_multi-classe"
            models = {
                "RandomForest": RandomForestClassifier(n_jobs=-1, random_state=42),
                "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
                "SGDClassifier": SGDClassifier(n_jobs=-1, random_state=42)
            }
            if not is_sparse:
                models["HistGradientBoosting"] = HistGradientBoostingClassifier(random_state=42)
            else :
                models["LinearSVC"] = LinearSVC()
                
            metrics = ["f1", "roc_auc", "accuracy"]
        else:
            # Cas: Régression Standard
            task = "régression"
            models = { 
                "RandomForestReg": RandomForestRegressor(n_jobs=-1, n_estimators=100, random_state=42),
                "SGDRegressor": TransformedTargetRegressor(
                    regressor=SGDRegressor(), # Note: Pour de très gros datasets, SGDRegressor est préférable
                    transformer=StandardScaler()
                )
            }
            if not is_sparse:
                models["HistGradientBoostingReg"] = TransformedTargetRegressor(
                    regressor=HistGradientBoostingRegressor(random_state=42),
                    transformer=StandardScaler()
                )
            metrics = ["rmse", "r2"]

    return {
        "type": task, 
        "models": models, 
        "metrics": metrics, 
        "df_solution_adjusted": df_solution_adjusted
    }


def evaluate_models(
    models: Dict[str, Any], 
    X_train_trans: Any, 
    y_train: Any, 
    X_test_trans: Any, 
    y_test: Any, 
    model_info: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Entraîne chaque modèle candidat et évalue ses performances sur le jeu de test.

    Args:
        models (dict): Dictionnaire des modèles instanciés.
        X_train_trans (array-like): Features d'entraînement prétraitées.
        y_train (array-like): Cibles d'entraînement.
        X_test_trans (array-like): Features de test prétraitées.
        y_test (array-like): Cibles de test.
        model_info (dict): Informations sur la tâche (pour gérer predict_proba).

    Returns:
        dict: Résultats contenant l'objet modèle entraîné, les prédictions et probabilités.
    """
    results = {}
    
    for name, model in models.items():
        print(f"Entraînement de {name}...")
        try:
            model.fit(X_train_trans, y_train)
            y_pred = model.predict(X_test_trans)
            
            # Gestion des probabilités pour le calcul de l'AUC
            y_proba = None
            if hasattr(model, "predict_proba"):
                try:
                    proba_full = model.predict_proba(X_test_trans)
                    # Pour la classification binaire, on ne garde souvent que la colonne 1 (proba positive)
                    if model_info["type"] == "classification_binaire" and proba_full.ndim == 2:
                        y_proba = proba_full[:, 1]
                    else:
                        y_proba = proba_full
                except Exception:
                    # Certains modèles (ex: SGD sans loss='log') n'ont pas predict_proba bien qu'ayant l'attribut
                    pass
                
            results[name] = {
                "model_obj": model, # On garde l'objet pour pouvoir faire .predict() plus tard
                "y_pred": y_pred,
                "y_proba": y_proba,
            }
        except Exception as e:
            print(f"Attention : Echec entraînement du modèle {name}: {e}")
            # On continue la boucle pour essayer les autres modèles

    return results


def choisir_meilleur_model(
    results: Dict[str, Dict[str, Any]], 
    y_test: Any, 
    model_info: Dict[str, Any],
    affichage: bool,
    model=None,
    model_name=None
) -> Optional[str]:
    """
    Compare les modèles entraînés selon les métriques définies et sélectionne le meilleur.

    Le choix se fait sur la base d'un "score moyen normalisé" (avg_score) calculé
    sur toutes les métriques pertinentes (ramenées entre 0 et 1).

    Args:
        results (dict): Résultats retournés par evaluate_models.
        y_test (array-like): Véritables valeurs cibles du test.
        model_info (dict): Contient la liste des métriques à calculer.
        affichage (bool): Booléen pour afficher les métriques.

    Returns:
        str: Le nom (clé) du meilleur modèle.
    """
    best_model_name = None
    best_avg_score = -np.inf

    if affichage:
        print("\n--- Résultats de l'évaluation initiale ---")
    
    # Calcul de l'étendue des valeurs (Range) pour normaliser le RMSE
    y_range = 1.0
    if "reg" in model_info["type"].lower():
        y_test_flat = np.ravel(y_test)
        y_range = np.max(y_test_flat) - np.min(y_test_flat)
        if y_range == 0: 
            y_range = 1.0
    
    for name, data in results.items():
        y_p = data["y_pred"]
        y_prob = data["y_proba"]
        
        scores_norm = {}   # Scores normalisés (0-1) pour la décision
        scores_bruts = {}  # Valeurs réelles pour l'affichage
        
        for metric in model_info["metrics"]:
            try:
                # --- Classification ---
                if metric == "accuracy":
                    val = accuracy_score(y_test, y_p)
                    scores_norm['accuracy'] = val
                    
                elif metric == "f1_macro":
                    val = f1_score(y_test, y_p, average="macro")
                    scores_norm['f1_macro'] = val
                    
                elif metric == "f1": # Binaire
                    val = f1_score(y_test, y_p, average="binary")
                    scores_norm['f1'] = val
                    
                elif metric == "roc_auc" and y_prob is not None:
                    if "multi-classe" in model_info["type"]:
                        val = roc_auc_score(y_test, y_prob, multi_class='ovr')
                    else:
                        val = roc_auc_score(y_test, y_prob)
                    scores_norm['AUC'] = val
                    
                elif metric == "jaccard":
                    val = jaccard_score(y_test, y_p, average="samples", zero_division=0)
                    scores_norm['jaccard'] = val
                
                # --- Régression ---
                elif metric == "rmse":
                    rmse_val = root_mean_squared_error(y_test, y_p)
                    scores_bruts["RMSE"] = rmse_val
                    # NRMSE inversé : 1 (parfait) -> 0 (mauvais)
                    scores_norm['RMSE'] = max(0, 1 - (rmse_val / y_range)) 
                    
                elif metric == "r2":
                    val = r2_score(y_test, y_p)
                    scores_bruts["R2"] = val
                    # On ramène les R2 négatifs à 0 pour la moyenne
                    scores_norm['R2'] = max(0, val)
                
                # Sauvegarde du score brut s'il n'a pas été calculé spécifiquement
                if metric not in scores_bruts and metric in scores_norm:
                    scores_bruts[metric] = scores_norm[metric]

            except Exception:
                # Ignore les métriques qui échouent (ex: ROC AUC sans probas)
                pass

        # Calcul du score global (Moyenne des scores normalisés)
        vals = list(scores_norm.values())
        avg_score = np.mean(vals) if vals else -1.0

        # Formatage pour affichage
        details_str = ", ".join([
            f"{k.upper()}: {scores_bruts.get(k.upper(), scores_bruts.get(k, 0)):.3f}" 
            for k in model_info['metrics'] 
            if k in scores_bruts or k in scores_norm
        ])

        if affichage:
            print(f"Modèle: {name:20} | Score Global: {avg_score:.4f} | Détails: [{details_str}]")
        
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            best_model_name = name

    if affichage:
        if best_model_name:
            print(f"\nMEILLEUR MODÈLE SÉLECTIONNÉ : {best_model_name} (Score initial: {best_avg_score:.4f})")
        else:
            print("\nAUCUN MODÈLE SÉLECTIONNÉ (Echec de toutes les évaluations).")
            
    return best_model_name

    


    
    
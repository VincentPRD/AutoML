"""
Module de prétraitement des données (Preprocessing).

Ce module contient les fonctions nécessaires pour nettoyer les types de données,
préparer les pipelines de transformation (imputation, mise à l'échelle, encodage)
et diviser les données en ensembles d'entraînement et de test.
"""

from typing import Tuple, Union, Optional, List

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def nettoyer_data_types(
    df_data: Optional[pd.DataFrame], 
    df_probleme: Optional[pd.DataFrame]
) -> Optional[pd.DataFrame]:
    """
    Nettoie et convertit les types de données des colonnes pour les DataFrames denses.

    Cette fonction est cruciale pour éviter les erreurs de type lors du OneHotEncoding.
    Par exemple, si une colonne catégorielle contient des 'NaN', pandas peut la 
    considérer comme 'float'. Cette fonction force la conversion en chaîne de caractères 
    pour les colonnes identifiées comme catégorielles dans les métadonnées.

    Args:
        df_data (pd.DataFrame | None): Le DataFrame des features.
        df_probleme (pd.DataFrame | None): Le DataFrame des métadonnées contenant les types.

    Returns:
        pd.DataFrame | None: Le DataFrame nettoyé.
    """
    # Vérifications de sécurité
    if df_data is None or df_probleme is None:
        return df_data
    
    types = df_probleme["Type"].values
    columns = df_data.columns
    
    # Si le nombre de colonnes ne correspond pas aux métadonnées, on ne touche à rien
    if len(columns) != len(types):
        return df_data

    for col, t in zip(columns, types):
        if "categorical" in t.lower():
            # Conversion explicite en string pour les catégories.
            # On remplace ensuite la chaîne 'nan' (résultat du cast string sur un null)
            # par un vrai np.nan pour que le SimpleImputer le détecte plus tard.
            df_data[col] = df_data[col].astype(str).replace('nan', np.nan)
            
    return df_data


def prepare_preprocessor(
    X: Union[pd.DataFrame, scipy.sparse.spmatrix], 
    df_probleme: pd.DataFrame, 
    is_sparse: bool
) -> Union[Pipeline, ColumnTransformer]:
    """
    Construit le pipeline de prétraitement adapté au format des données (Dense ou Sparse).

    Stratégies appliquées :
    - Données Creuses (Sparse) : Mise à l'échelle sans centrage (pour garder la sparsité).
    - Données Denses : 
        - Numérique : Imputation par la moyenne + StandardScaler.
        - Catégoriel : Imputation constante ('missing') + OneHotEncoder.
        - Booléen : Imputation par le plus fréquent + OneHotEncoder binaire.

    Args:
        X (pd.DataFrame | scipy.sparse.spmatrix): Les données d'entraînement (utilisées pour déduire les colonnes).
        df_probleme (pd.DataFrame): Les métadonnées des colonnes.
        is_sparse (bool): Indique si les données sont au format matrice creuse.

    Returns:
        Pipeline | ColumnTransformer: L'objet scikit-learn prêt à être 'fit'.
    """
    
    # === CAS 1 : DONNÉES SPARSE (Matrice Creuse) ===
    if is_sparse:
        # On ne peut pas utiliser ColumnTransformer sur une matrice sans noms de colonnes.
        # On applique un scaling global.
        # IMPORTANT : with_mean=False est obligatoire pour ne pas "densifier" la matrice
        # (ce qui exploserait la mémoire RAM).
        return Pipeline(steps=[
            ('scaler', StandardScaler(with_mean=False)) 
        ])

    # === CAS 2 : DONNÉES DENSE (DataFrame Pandas) ===
    numeric_features: List[str] = []
    categorical_features: List[str] = []
    bool_features: List[str] = []

    types = df_probleme["Type"].values
    columns = X.columns

    # Classification des colonnes selon le fichier .type
    for col, t in zip(columns, types):
        t_lower = t.lower()
        if "numerical" in t_lower:
            numeric_features.append(col)
        elif "categorical" in t_lower:
            categorical_features.append(col)
        elif "boolean" in t_lower:
            bool_features.append(col)

    # Pipeline pour variables numériques
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Pipeline pour variables catégorielles
    # sparse_output=True permet d'économiser de la mémoire si le nombre de catégories est élevé
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ])
    
    # Pipeline pour variables booléennes
    bool_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=True))
    ])

    # Assemblage final
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("bool", bool_transformer, bool_features)
        ],
        remainder="drop", # Les colonnes non listées (ex: IDs) sont supprimées
        sparse_threshold=0.1 # Si le résultat est < 10% dense, on retourne une matrice sparse
    )
    
    return preprocessor


def split_data(
    df_data: Union[pd.DataFrame, scipy.sparse.spmatrix], 
    df_solution: Union[pd.Series, pd.DataFrame], 
    model_info: dict
) -> Tuple[Union[np.ndarray, scipy.sparse.spmatrix], Union[np.ndarray, scipy.sparse.spmatrix], np.ndarray, np.ndarray]:
    """
    Divise les données en ensembles d'entraînement et de test.

    Tente d'appliquer une stratification (garder la même proportion de classes)
    si la tâche est une classification standard.

    Args:
        df_data (pd.DataFrame | spmatrix): Features.
        df_solution (pd.Series | pd.DataFrame): Target.
        model_info (dict): Informations sur le type de tâche (pour décider de la stratification).

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    stratify = None
    
    # On stratifie uniquement si :
    # 1. Ce n'est pas une matrice sparse (sklearn gère mal stratify sur certains formats sparse parfois)
    # 2. C'est une classification standard (binaire ou multi-classe)
    # 3. Ce n'est pas du multi-label (stratification complexe non supportée nativement simplement)
    if (not scipy.sparse.issparse(df_data) and 
        "classification" in model_info["type"] and 
        "classification_multi-label" not in model_info["type"]):
        stratify = df_solution
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            df_data, 
            df_solution, 
            test_size=0.3, 
            random_state=42, 
            stratify=stratify
        )
    except ValueError:
        # Fallback : Si la stratification échoue (ex: une classe n'a qu'un seul membre),
        # on recommence sans stratification.
        X_train, X_test, y_train, y_test = train_test_split(
            df_data, 
            df_solution, 
            test_size=0.3, 
            random_state=42
        )
        
    # Aplatissage du vecteur cible (y) si nécessaire.
    # Scikit-learn préfère des vecteurs 1D (shape (n,)) pour la classification/régression simple.
    # Pour le multi-output/multi-label, on doit garder la dimension 2D (shape (n, k)).
    if "multi" not in model_info["type"]:
        y_train = np.ravel(y_train)
        y_test = np.ravel(y_test)
        
    return X_train, X_test, y_train, y_test
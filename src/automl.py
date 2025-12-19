"""
Module principal du pipiline de Machine Learning Automatisé (AutoML)

Ce module contient la classe principale AutoML permettant d'assembler les différentes
parties du pipeline.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import scipy.sparse

# Imports locaux
from optimization.optimize_unique import AutoOptimizer
from data.data_io import chargement_donnees_complet
from data.data_preprocessing import nettoyer_data_types, prepare_preprocessor, split_data
from model.model_base import cherche_type_probleme, evaluate_models, choisir_meilleur_model


class AutoML:
    """
    Classe principale du pipeline de Machine Learning Automatisé (AutoML).

    Cette classe orchestre le flux complet d'apprentissage automatique :
    1. Chargement des données (supporte DataFrames denses et Matrices creuses).
    2. Détection automatique du type de problème (Classification, Régression, etc.).
    3. Prétraitement des données (Nettoyage, Encodage, Mise à l'échelle).
    4. Entraînement et évaluation initiale des modèles.
    5. Optimisation des hyperparamètres.
    6. Génération des prédictions finales.

    Attributes:
        df_data (pd.DataFrame | scipy.sparse.spmatrix): Données d'entrée (features).
        df_solution (pd.Series | pd.DataFrame): Valeurs cibles (target).
        df_probleme (pd.DataFrame): Métadonnées décrivant les types de colonnes.
        model_info (dict): Informations sur la tâche détectée, les modèles éligibles et métriques.
        models (dict): Dictionnaire des instances de modèles initialisés.
        best_model_name (str): Nom du meilleur modèle identifié.
        best_model_obj (sklearn.base.BaseEstimator): L'objet modèle entraîné le plus performant.
        preprocessor (sklearn.compose.ColumnTransformer): Pipeline de prétraitement ajusté (fitted).
        is_sparse (bool): Indicateur si les données d'entrée sont une matrice creuse.
    """

    def __init__(self):
        """Initialise le pipeline AutoML avec des attributs vides."""
        self.df_data = None
        self.df_solution = None
        self.df_probleme = None
        
        self.model_info = None
        self.models = {}
        self.best_model_name = None
        self.best_model_obj = None 
        
        # Données transformées et splits
        self.X_train_trans = None
        self.X_test_trans = None
        self.y_train = None
        self.y_test = None

        self.results = {}
        self.preprocessor = None
        self.is_sparse = False
        
    def fit(self, data_dest: str) -> None:
        """
        Exécute le pipeline complet d'entraînement sur le jeu de données.

        Étapes :
            1. Chargement et détection du format (Dense vs Sparse).
            2. Identification de la tâche ML.
            3. Nettoyage et split Train/Test.
            4. Création et ajustement (fit) du préprocesseur.
            5. Entraînement des modèles candidats.
            6. Sélection du meilleur modèle de base.
            7. Optimisation des hyperparamètres.

        Args:
            data_dest (str): Chemin vers le dossier contenant les fichiers du dataset 
                             (doit contenir .data, .solution, et optionnellement .type).
        
        Raises:
            Exception: Si les données n'ont pas été chargées.
        """
        # Extraction du nom du dataset pour l'affichage
        dataset_name = data_dest.rstrip('/').split(sep='/')[-1]
        print(f"--- Démarrage AutoML sur {dataset_name} ---")
        
        # 1. Chargement des données (via module data_io)
        self.df_data, self.df_solution, self.df_probleme, self.is_sparse = \
            chargement_donnees_complet(data_dest, load_solution=True)
            
        if self.df_data is None:
            raise Exception("Erreur : Données non chargées.")
            return 
            
        print(f" -> Format des données : {'Matrice creuse' if self.is_sparse else 'DataFrame'}")
        
        # 2. Détection de Tâche (via module model_base)
        # Analyse la distribution de la target pour déterminer le type de problème
        info = cherche_type_probleme(self.df_solution, self.is_sparse)
        
        self.model_info = {k: info[k] for k in ['type', 'models', 'metrics']}
        self.models = info['models']
        # La solution peut être ajustée (ex: conversion OneHot -> Labels)
        self.df_solution = info['df_solution_adjusted']
        
        print(f"Tâche détectée : {self.model_info['type']}")

        # 3. Préprocessing : Nettoyage & Split (via module data_preprocessing)
        # Le nettoyage des types n'est nécessaire que pour les DataFrames denses
        if not self.is_sparse:
            self.df_data = nettoyer_data_types(self.df_data, self.df_probleme)

        X_train, X_test, self.y_train, self.y_test = split_data(
            self.df_data, self.df_solution, self.model_info
        )

        # 4. Préprocessing : Transformation (via module data_preprocessing)
        print("Configuration du Preprocessing...")
        self.preprocessor = prepare_preprocessor(X_train, self.df_probleme, self.is_sparse)
        
        # Transformation des données
        # IMPORTANT : On fit uniquement sur le Train pour éviter la fuite de données (Data Leakage)
        self.X_train_trans = self.preprocessor.fit_transform(X_train)
        self.X_test_trans = self.preprocessor.transform(X_test)
        
        # 5. Entrainement et Évaluation initiale (via module model_base)
        self.results = evaluate_models(
            self.models, 
            self.X_train_trans, self.y_train, 
            self.X_test_trans, self.y_test, 
            self.model_info
        )
        
        # Sélection du meilleur modèle selon les métriques définies
        self.best_model_name = choisir_meilleur_model(self.results, self.y_test, self.model_info)
        # On sauvegarde l'objet modèle initial comme "meilleur" par défaut
        if self.best_model_name in self.results:
            self.best_model_obj = self.results[self.best_model_name]

        # 6. Optimisation (via module optimization)
        print(f"\n--- Optimisation du meilleur modèle : {self.best_model_name} ---")
        optimize = AutoOptimizer()
        best_model_name_optimized, best_params = optimize.optimize(
            self.X_train_trans, 
            self.y_train, 
            self.model_info, 
            self.best_model_name
        )
        
        # TODO : Implémenter la reconstruction finale du modèle avec les hyperparamètres optimisés
        # self._build_final_model(best_model_name_optimized, best_params)

    def eval(self) -> None:
        """
        Affiche le résumé de l'évaluation des modèles entraînés.
        Utile pour inspecter les résultats après l'entraînement sans relancer le calcul.

        Raises:
            Exception: Si aucun modèle n'a été entraîné.
        """
        if not self.results:
            raise Exception("Aucun modèle n'a été entraîné.")
            return

        # Du module model_base
        choisir_meilleur_model(self.results, self.y_test, self.model_info)

    def predict(self, data_dest: str):
        """
        Génère des prédictions sur de nouvelles données.

        Args:
            data_dest (str): 
                Un chemin vers le fichier de test.

        Returns:
            np.array: Tableau des prédictions.

        Raises:
            Exception: Si le modèle n'a pas encore été entraîné.
        """
        print(f"\n--- Prédiction en cours ---")
        
        if self.best_model_obj is None or self.preprocessor is None:
            raise Exception("Le modèle n'est pas entraîné ou le meilleur modèle n'a pas été défini.")

        # 1. Chargement/Préparation des données de prédiction
        df_to_predict = None
        
        if isinstance(data_dest, str):
            # Chargement depuis un fichier (on ignore la solution si elle existe)
            df_to_predict, _, df_probleme_pred, _ = chargement_donnees_complet(
                data_dest, load_solution=False
            )
            
            if df_to_predict is None:
                return None
            
            # Nettoyage des types si nécessaire (Dense seulement)
            if not self.is_sparse and isinstance(df_to_predict, pd.DataFrame):
                # On utilise les métadonnées du fichier test s'il existe, sinon celles du train
                metadata = df_probleme_pred if df_probleme_pred is not None else self.df_probleme
                df_to_predict = nettoyer_data_types(df_to_predict, metadata)
        else:
            raise ValueError("Format d'entrée non reconnu (attendu: str path).")
            
        # 2. Transformation et Prédiction
        try:
            # IMPORTANT : Toujours utiliser transform(), jamais fit_transform() sur les données de test
            X_pred = self.preprocessor.transform(df_to_predict)
            predictions = self.best_model_obj.predict(X_pred)
            return predictions
        except Exception as e:
            print(f"Erreur lors de la prédiction : {e}")
            return None

    def _build_final_model(self, model_name: str, params: dict):
        """
        Méthode interne pour reconstruire et réentraîner le modèle final 
        avec les hyperparamètres optimisés (TODO).
        
        Args:
            model_name (str): Nom du modèle à construire.
            params (dict): Hyperparamètres optimisés.
        """
        print(f"Finalisation du modèle {model_name} avec les meilleurs paramètres.")
        # Logique d'instanciation et de fit final ici...
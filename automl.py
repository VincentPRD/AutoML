import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier,
    HistGradientBoostingRegressor
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, SVC
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, root_mean_squared_error,
    r2_score, jaccard_score, mean_absolute_error
)
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.impute import SimpleImputer

class AutoML:

    def __init__(self):
        self.df_data = None
        self.df_solution = None
        self.df_probleme = None
        self.model_info = None
        self.models = {}
        self.best_model_name = None
        
        # Données splitées
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Résultats
        self.results = {} # Stockera {nom_model: {'model': obj, 'y_pred': array, 'metrics': dict}}
        self.preprocessor = None

    ##
    # ===== Méthodes publiques =====
    ##
    def fit(self, data_dest):
        """
        Orchestration complète : Chargement -> Typage -> Split -> Preprocessing -> Train
        """
        print(f"--- Démarrage AutoML sur {data_dest} ---")
        
        # 1. Chargement
        if not self._chargement_donnees(data_dest):
            return 
            
        # 2. Analyse du type de problème (Target)
        self.model_info = self._cherche_type_probleme()
        print("Tâche détectée :", self.model_info["type"])
        
        # On convertit les colonnes catégorielles en STRING avant le split
        # pour éviter l'erreur "Cannot cast 'missing' to float64"
        types = self.df_probleme["Type"].values
        columns = self.df_data.columns
        
        for col, t in zip(columns, types):
            if "categorical" in t.lower():
                # On convertit en string, puis on remet les 'nan' string en vrais np.nan
                self.df_data[col] = self.df_data[col].astype(str).replace('nan', np.nan)

        # 3. Split des données
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df_data, self.df_solution, test_size=0.3, random_state=42
        )
        
        # Aplatir y si nécessaire
        #if self.y_train.shape[1] == 1:
        #    self.y_train = self.y_train.values.ravel()
        #    self.y_test = self.y_test.values.ravel()

        # 4. Préparation et fit du Preprocessor
        self._prepare_preprocessor()
        
        # Transformation
        print("Preprocessing en cours...")
        self.X_train_trans = self.preprocessor.fit_transform(self.X_train)
        self.X_test_trans = self.preprocessor.transform(self.X_test)
        
        # 5. Entrainement
        self._fit_predict_models()  
        
    def eval(self):
        """
        Évaluation des modèles et choix du meilleur.
        """
        if not self.results:
            print("Aucun modèle n'a été entraîné.")
            return

        self._choisi_meilleur_model()

    ##
    # ===== Méthodes privées =====
    ##
    def _chargement_donnees(self, data_dest):
        """
        Chargement robuste + Correction des noms de colonnes
        """
        base_path = "/info/corpus/ChallengeMachineLearning/" 
        full_path = os.path.join(base_path, data_dest)
        
        if not os.path.exists(full_path):
            print(f"Mode local : recherche dans {data_dest}")
            full_path = data_dest 

        try:
            name = os.path.basename(data_dest)
            fichiers = {
                'data': os.path.join(full_path, name + '.data'),
                'solution': os.path.join(full_path, name + '.solution'),
                'type': os.path.join(full_path, name + '.type')
            }

            # Chargement Data
            self.df_data = pd.read_csv(fichiers['data'], sep=' ', header=None) 
            self.df_data.dropna(axis=1, how='all', inplace=True)
            # IMPORTANT : Forcer les noms de colonnes en string immédiatement
            self.df_data.columns = self.df_data.columns.astype(str)
            
            # Chargement Solution
            self.df_solution = pd.read_csv(fichiers['solution'], sep=' ', header=None) 
            self.df_solution.dropna(axis=1, how='all', inplace=True)

            # Chargement Types
            self.df_probleme = pd.read_csv(fichiers['type'], sep=r'\s+', header=None, names=['Type'])
            
            if len(self.df_data) != len(self.df_solution):
                raise ValueError("Data et Solution n'ont pas la même taille !")
                
            return True

        except Exception as e:
            print(f"Erreur chargement : {e}")
            return False

    def _prepare_preprocessor(self):
        """
        Crée le pipeline de preprocessing (Imputation + Scaling + Encoding)
        """
        # On suppose que l'ordre des lignes de .type correspond à l'ordre des colonnes de .data
        
        numeric_features = []
        categorical_features = []
        bool_features = []

        types = self.df_probleme["Type"].values
        columns = self.df_data.columns # Ils sont déjà en string grâce au chargement

        for col, t in zip(columns, types):
            if "numerical" in t.lower():
                numeric_features.append(col)
            elif "categorical" in t.lower():
                categorical_features.append(col)
            elif "boolean" in t.lower():
                bool_features.append(col)

        # Pipelines spécifiques
        # 1. Numérique : On remplace les trous par la médiane, puis on normalise
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # 2. Catégorique : On remplace les trous par "missing", puis OneHot
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # 3. Booléen
        bool_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='if_binary', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
                ("bool", bool_transformer, bool_features)
            ],
            remainder="drop" # On jette ce qu'on ne connait pas
        )

    def _cherche_type_probleme(self):
        """
        Détecte le type de tâche et configure les modèles
        """
        n_cols = self.df_solution.shape[1]
        models = {}
        metrics = []
        task = ""

        # Analyse des valeurs uniques pour savoir si c'est discret ou continu
        # On regarde un échantillon pour aller vite
        sample = self.df_solution.iloc[:1000].values.flatten()
        unique_vals = np.unique(sample[~np.isnan(sample)]) # ignore nan pour check
        is_discrete = (len(unique_vals) < 20) and (np.all(np.mod(unique_vals, 1) == 0))

        # --- Cas Multivarié (Plusieurs colonnes cibles) ---
        if n_cols > 1:
            # Vérifions si c'est du One-Hot (Classification Multi-classe encodée)
            # Somme des lignes doit être 1 partout (ou 0 et 1)
            row_sums = self.df_solution.sum(axis=1)
            is_one_hot = np.all(row_sums.isin([0, 1])) and is_discrete

            if is_one_hot:
                task = "classification_multi-classe"
                # Pour multi-classe, il faut reconstruire y en une seule colonne de labels
                self.df_solution = self.df_solution.idxmax(axis=1) # Transforme OneHot en labels
                
                models = {
                    "HistGradientBoosting": HistGradientBoostingClassifier(),
                    "MLP": MLPClassifier(max_iter=500),
                    "LogisticRegression": LogisticRegression(max_iter=1000, multi_class="multinomial"),
                    "SVC": SVC()
                }
                metrics = ["accuracy", "f1_macro"]
            else:
                # Si valeurs discrètes (0/1) mais sommes > 1 => Multi-label
                if is_discrete and set(unique_vals).issubset({0, 1}):
                    task = "classification_multi-label"
                    models = {
                        # RandomForest gère nativement le multi-output, pas besoin de wrapper parfois, mais plus sûr avec Wrapper pour d'autres
                        "RandomForest": RandomForestClassifier(n_jobs=-1),
                        "MultiOutput_GB": MultiOutputClassifier(HistGradientBoostingClassifier()),
                        "MLP": MLPClassifier(max_iter=500) # MLP gère le multi-label nativement
                    }
                    metrics = ["f1_micro", "f1_macro", "jaccard"]
                else:
                    task = "régression_multi-sortie"
                    models = {
                        "RandomForestReg": RandomForestRegressor(n_jobs=-1),
                        "MultiOutput_GBReg": MultiOutputClassifier(HistGradientBoostingRegressor()),
                        "SVR": SVR()
                    }
                    metrics = ["rmse", "r2", "mae"]

        # --- Cas Univarié (Une seule colonne cible) ---
        else:
            if is_discrete or len(unique_vals) == 2:
                task = "classification_binaire"
                models = {
                    "HistGradientBoosting": HistGradientBoostingClassifier(),
                    "LogisticRegression": LogisticRegression(max_iter=1000),
                    "MLP": MLPClassifier(max_iter=500),
                    "SVC": SVC()
                }
                metrics = ["f1", "roc_auc", "accuracy"]
            else:
                task = "régression"
                # On utilise TransformedTargetRegressor pour que le modèle apprenne 
                # sur des données normalisées, mais renvoie des prédictions à l'échelle réelle.
                models = { 
                    "HistGradientBoostingReg": TransformedTargetRegressor(
                        regressor=HistGradientBoostingRegressor(),
                        transformer=StandardScaler()
                    ),
                    "RandomForestReg": TransformedTargetRegressor(
                        regressor=RandomForestRegressor(n_jobs=-1),
                        transformer=StandardScaler()
                    ),
                    "MLPReg": TransformedTargetRegressor(
                        regressor=MLPRegressor(max_iter=500),
                        transformer=StandardScaler()
                    ),
                    "SVR": SVR()
                    
        
                }
                metrics = ["rmse", "r2", "mae"]

        self.models = models
        return {"type": task, "models": models, "metrics": metrics}
    
    def _fit_predict_models(self):
        """
        Entraîne et prédit, stocke tout dans self.results
        """
        for name, model in self.models.items():
            print(f"Entraînement de {name}...")
            try:
                model.fit(self.X_train_trans, self.y_train)
                y_pred = model.predict(self.X_test_trans)
                
                # Gestion probas pour ROC_AUC
                y_proba = None
                if hasattr(model, "predict_proba"):
                    try:
                        y_proba = model.predict_proba(self.X_test_trans)
                        # Cas binaire: on garde proba de la classe 1
                        if self.model_info["type"] == "classification_binaire":
                            y_proba = y_proba[:, 1]
                    except:
                        pass
                
                self.results[name] = {
                    "model_obj": model,
                    "y_pred": y_pred,
                    "y_proba": y_proba,
                    "scores": {}
                }
            except Exception as e:
                print(f"Echec entraînement {name}: {e}")

    def _choisi_meilleur_model(self):
        """
        Sélectionne le meilleur modèle et affiche des scores normalisés (0 à 1).
        Pour la régression, on calcule un "Score de Précision" basé sur l'erreur relative.
        """
        best_model_name = None
        best_avg_score = -np.inf
        
        print("\n--- Résultats de l'évaluation ---")
        
        # Calcul de l'étendue des données (Range) pour la normalisation
        # Si y est encodé (classes), range_y n'a pas de sens, mais on gère ça via le type de tâche
        y_range = 1.0
        if "reg" in self.model_info["type"].lower():
             y_range = np.max(self.y_test) - np.min(self.y_test)
             if y_range == 0: y_range = 1.0 # Évite division par zéro

        for name, data in self.results.items():
            y_p = data["y_pred"]
            y_prob = data["y_proba"]
            
            # Stockage des scores pour ce modèle
            scores_bruts = {}      # RMSE, MAE tels quels
            scores_norm = []       # Scores entre 0 et 1 (où 1 est le meilleur)
            
            # --- BOUCLE SUR LES MÉTRIQUES ---
            for metric in self.model_info["metrics"]:
                val = 0
                try:
                    # === CLASSIFICATION ===
                    if metric == "accuracy":
                        val = accuracy_score(self.y_test, y_p)
                        scores_norm.append(val)
                    elif metric == "f1_macro":
                        val = f1_score(self.y_test, y_p, average="macro")
                        scores_norm.append(val)
                    elif metric == "f1_micro":
                        val = f1_score(self.y_test, y_p, average="micro")
                        scores_norm.append(val)
                    elif metric == "roc_auc" and y_prob is not None:
                        if self.model_info["type"] == "classification_multi-classe":
                            val = roc_auc_score(self.y_test, y_prob, multi_class='ovr')
                        else:
                            val = roc_auc_score(self.y_test, y_prob)
                        scores_norm.append(val)
                    elif metric == "jaccard":
                        val = jaccard_score(self.y_test, y_p, average="samples", zero_division=0)
                        scores_norm.append(val)

                    # === RÉGRESSION (Normalisation ici) ===
                    elif metric == "rmse":
                        # Valeur brute
                        rmse_val = root_mean_squared_error(self.y_test, y_p)
                        scores_bruts["RMSE"] = rmse_val
                        
                        # Normalisation : NRMSE (Normalized RMSE)
                        # Score = 1 - (Erreur / Etendue). Si < 0, on met 0.
                        # Cela donne un score type "Précision"
                        nrmse = rmse_val / y_range
                        score_precision = max(0, 1 - nrmse) 
                        scores_norm.append(score_precision)
                        
                    elif metric == "mae":
                        mae_val = mean_absolute_error(self.y_test, y_p)
                        scores_bruts["MAE"] = mae_val
                        
                        # Normalisation similaire
                        nmae = mae_val / y_range
                        score_precision = max(0, 1 - nmae)
                        scores_norm.append(score_precision)
                        
                    elif metric == "r2":
                        val = r2_score(self.y_test, y_p)
                        scores_bruts["R2"] = val
                        # R2 est déjà (presque) normalisé. S'il est négatif, on le ramène à 0 pour la moyenne
                        scores_norm.append(max(0, val))

                    # Sauvegarde valeur brute si non régression
                    if metric not in ["rmse", "mae", "r2"]:
                        scores_bruts[metric] = val

                except Exception as e:
                    print(f"Erreur métrique {metric} : {e}")

            # --- CALCUL DU SCORE FINAL ---
            # Moyenne des scores normalisés (tous sont maintenant orientés : 1 = parfait)
            avg_score = np.mean(scores_norm) if scores_norm else -1

            print(f"Modèle: {name:25} | Score Global (0-1): {avg_score:.4f}")
            if "reg" in self.model_info["type"].lower():
                print(f"    -> Détails: RMSE={scores_bruts.get('RMSE',0):.2f}, R2={scores_bruts.get('R2',0):.4f}")
            else:
                print(f"    -> Détails: {scores_bruts}")
            
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                best_model_name = name

        print(f"\nMEILLEUR MODÈLE : {best_model_name} avec un score de {best_avg_score:.4f}")
        return self.results[best_model_name]
import numpy as np
import pandas as pd
import os
from io import BytesIO
import scipy.sparse
from sklearn.datasets import load_svmlight_file

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, 
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVR, SVC
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, root_mean_squared_error,
    r2_score, jaccard_score, mean_absolute_error
)
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MaxAbsScaler
from sklearn.impute import SimpleImputer

class AutoML:

    def __init__(self):
        self.df_data = None
        self.df_solution = None
        self.df_probleme = None
        self.model_info = None
        self.models = {}
        self.best_model_name = None
        self.best_model_obj = None 
        
        # Données
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.results = {}
        self.preprocessor = None
        self.is_sparse = False

    def fit(self, data_dest):
        print(f"--- Démarrage AutoML sur {data_dest.split(sep='/')[-2]} ---")
        
        # 1. Chargement (Data + Solution) avec détection automatique
        if not self._chargement_donnees(data_dest, load_solution=True):
            return 
            
        # 2. Analyse du type de problème
        self.model_info = self._cherche_type_probleme()
        print(f"Tâche détectée : {self.model_info['type']} (Données creuses: {self.is_sparse})")
        
        # Nettoyage des types (Uniquement si DataFrame dense)
        if not self.is_sparse:
            self._nettoie_data_types()

        # 3. Split des données
        # Stratify ne fonctionne pas toujours bien avec les matrices creuses si pas géré, 
        # on simplifie pour le sparse ou on tente le coup.
        stratify = None
        if not self.is_sparse and "classification" in self.model_info["type"] and "multi" not in self.model_info["type"]:
             stratify = self.df_solution

        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.df_data, self.df_solution, test_size=0.3, random_state=42, stratify=stratify
            )
        except ValueError:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.df_data, self.df_solution, test_size=0.3, random_state=42
            )

        # Aplatissage y si nécessaire
        if "multi" not in self.model_info["type"]:
             self.y_train = np.ravel(self.y_train)
             self.y_test = np.ravel(self.y_test)

        # 4. Préparation et fit du Preprocessor
        print("Configuration du Preprocessing...")
        self._prepare_preprocessor()
        
        # Transformation
        # Note : Si sparse, le preprocessor gère le format sparse
        self.X_train_trans = self.preprocessor.fit_transform(self.X_train)
        self.X_test_trans = self.preprocessor.transform(self.X_test)
        
        # 5. Entrainement
        self._fit_predict_models()
    

    def eval(self):
        if not self.results:
            print("Aucun modèle n'a été entraîné.")
            return

        self.best_model_name = self._choisi_meilleur_model()
        self.best_model_obj = self.results[self.best_model_name]['model_obj']

    def predict(self, data_dest_or_df):
        print(f"\n--- Prédiction en cours ---")
        
        if isinstance(data_dest_or_df, str):
            # On charge seulement .data
            if not self._chargement_donnees(data_dest_or_df, load_solution=False):
                return None
            df_to_predict = self.df_data 
            
            # Nettoyage seulement si dense
            if not self.is_sparse:
                 self._nettoie_data_types(is_predict=True)

        elif isinstance(data_dest_or_df, (pd.DataFrame, scipy.sparse.spmatrix)):
            df_to_predict = data_dest_or_df
            if isinstance(df_to_predict, pd.DataFrame):
                 df_to_predict = df_to_predict.copy()
                 df_to_predict.columns = df_to_predict.columns.astype(str)
                 self.is_sparse = False
            else:
                 self.is_sparse = True
        else:
            raise ValueError("Format d'entrée non reconnu.")

        if self.preprocessor is None:
            raise Exception("Le modèle n'est pas entraîné.")
            
        try:
            X_pred = self.preprocessor.transform(df_to_predict)
            predictions = self.best_model_obj.predict(X_pred)
            return predictions
        except Exception as e:
            print(f"Erreur lors de la prédiction : {e}")
            return None

    ##
    # ===== Méthodes privées =====
    ##
    
    def _nettoie_data_types(self, is_predict=False):
        """
        Uniquement pour les DataFrames Pandas (Dense)
        """
        if self.is_sparse or self.df_data is None:
            return

        types = self.df_probleme["Type"].values
        columns = self.df_data.columns
        
        # Vérifie que les dimensions correspondent
        if len(columns) != len(types):
            return 

        for col, t in zip(columns, types):
            if "categorical" in t.lower():
                self.df_data[col] = self.df_data[col].astype(str).replace('nan', np.nan)
    
    def _detecte_et_charge_data(self, filepath):
        """
        Détecte le format (Dense vs Sparse) et gère le cas des fichiers Sparse sans labels.
        """
        # 1. Analyse de la première ligne pour déterminer le format
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
            
        if not first_line:
            print(" -> Fichier vide détecté.")
            return None, False

        tokens = first_line.split()
        
        # Heuristique :
        # - Si on trouve ":" dans la ligne, c'est probablement du format Sparse (LibSVM).
        # - Si le PREMIER token contient ":", c'est qu'il n'y a PAS de label au début (ex: "542:1 ...").
        has_colon = ":" in first_line
        first_token_is_feature = ":" in tokens[0] if tokens else False
        
        if has_colon:
            print(" -> Format détecté : Sparse Matrix (LibSVM)")
            
            # Cas A : Fichier commençant par "Index:Value" (Pas de label) -> Hack nécessaire
            if first_token_is_feature:
                print("    -> Détection : Format sans colonne cible (Label). Chargement adapté...")
                return self._chargement_sparse_non_label(filepath), True
                
            # Cas B : Fichier Standard "Label Index:Value"
            else:
                try:
                    data, _ = load_svmlight_file(filepath)
                    return data, True
                except ValueError as e:
                    # Si ça plante malgré tout (ex: label non numérique bizarre), on tente le mode sans label
                    print(f"    -> Erreur chargement standard ({e}), tentative mode 'sans label'...")
                    return self._chargement_sparse_non_label(filepath), True
        else:
            print(" -> Format détecté : Dense DataFrame (CSV/Space-separated)")
            # Lecture classique Pandas
            data = pd.read_csv(filepath, sep=' ', header=None)
            data.dropna(axis=1, how='all', inplace=True)
            data.columns = data.columns.astype(str)
            return data, False

    def _chargement_sparse_non_label(self, filepath):
        """
        Charge un fichier LibSVM qui n'a pas de target au début de la ligne.
        Astuce : On lit le fichier en binaire, on ajoute un dummy label '0' 
        au début de chaque ligne, puis on passe le tout à load_svmlight_file.
        """
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
            
            # On insère "0 " au début du fichier et après chaque saut de ligne
            # Cela transforme "542:1 ..." en "0 542:1 ..."
            new_content = b'0 ' + content.replace(b'\n', b'\n0 ')
            
            # Nettoyage si le fichier finissait par une ligne vide (évite un "0 " orphelin à la fin)
            if new_content.endswith(b'\n0 '):
                new_content = new_content[:-3]
                
            # On utilise BytesIO pour simuler un fichier en mémoire
            f_obj = BytesIO(new_content)
            
            X, _ = load_svmlight_file(f_obj)
            return X
            
        except Exception as e:
            print(f"Erreur fatale lors du chargement sparse manuel : {e}")
            raise e

    def _chargement_donnees(self, data_dest, load_solution=True):
        try:
            name = data_dest.rstrip('/').split(sep='/')[-1]
            if name == "": name = data_dest.split(sep='/')[-2]

            fichiers = {
                'data': os.path.join(data_dest, name + '.data'),
                'solution': os.path.join(data_dest, name + '.solution'),
                'type': os.path.join(data_dest, name + '.type')
            }

            if not os.path.exists(fichiers['data']):
                print(f"Fichier introuvable: {fichiers['data']}")
                return False
                
            # --- CHARGEMENT ---
            self.df_data, self.is_sparse = self._detecte_et_charge_data(fichiers['data'])
            
            # Chargement Type
            if os.path.exists(fichiers['type']):
                self.df_probleme = pd.read_csv(fichiers['type'], sep=r'\s+', header=None, names=['Type'])
            
            # Chargement Solution
            if load_solution:
                if not os.path.exists(fichiers['solution']):
                    print("Fichier solution manquant.")
                    return False
                self.df_solution = pd.read_csv(fichiers['solution'], sep=' ', header=None) 
                self.df_solution.dropna(axis=1, how='all', inplace=True)
                
                # Vérif taille
                if self.df_data.shape[0] != len(self.df_solution):
                    raise ValueError(f"Taille mismatch: Data={self.df_data.shape[0]}, Solution={len(self.df_solution)}")
                
            return True

        except Exception as e:
            print(f"Erreur chargement : {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def _cherche_type_probleme(self):
        n_cols = self.df_solution.shape[1]
        models = {}
        metrics = []
        task = ""

        sample = self.df_solution.iloc[:2000].values.flatten()
        sample = sample[~np.isnan(sample)]
        unique_vals = np.unique(sample)
        is_discrete = (len(unique_vals) < 50) and (np.all(np.mod(unique_vals, 1) == 0))

        # Pour les données SPARSES, HistGradientBoosting nécessite d'être converti en dense (lourd)
        # On privilégie RandomForest, SGD, LinearModels qui gèrent le sparse nativement
        
        # --- Cas Multivarié ---
        if n_cols > 1:
            row_sums = self.df_solution.sum(axis=1)
            is_one_hot = np.mean((row_sums >= 0.99) & (row_sums <= 1.01)) > 0.95

            if is_one_hot and is_discrete:
                task = "classification_multi-classe"
                self.df_solution = self.df_solution.idxmax(axis=1)
                
                models = {
                    "MLP": MLPClassifier(max_iter=300),
                    "LogisticRegression": LogisticRegression(max_iter=500),
                    "SGD": SGDClassifier(loss='hinge', n_jobs=-1) 
                }
                # HistGradientBoosting ne gère pas sparse nativement => on l'enlève si sparse
                if not self.is_sparse:
                     models["HistGradientBoosting"] = HistGradientBoostingClassifier(max_iter=300)
                     
                metrics = ["accuracy", "f1_macro"]
            else:
                if is_discrete and set(unique_vals).issubset({0, 1}):
                    task = "classification_multi-label"
                    models = {
                        "RandomForest": RandomForestClassifier(n_jobs=-1, n_estimators=50),
                        "MLP": MLPClassifier(max_iter=300)
                    }
                    if not self.is_sparse:
                        models["MultiOutput_GB"] = MultiOutputClassifier(HistGradientBoostingClassifier(max_iter=100))
                        
                    metrics = ["f1_macro", "jaccard"]
                else:
                    task = "régression_multi-sortie"
                    models = {
                        "RandomForestReg": RandomForestRegressor(n_jobs=-1, n_estimators=50),
                    }
                    if not self.is_sparse:
                         models["MultiOutput_GBReg"] = MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=100))
                    metrics = ["rmse", "r2"]

        # --- Cas Univarié ---
        else:
            if is_discrete or len(unique_vals) <= 2:
                task = "classification_binaire" if len(unique_vals) == 2 else "classification_multi-classe"
                models = {
                    "LogisticRegression": LogisticRegression(max_iter=500),
                    "RandomForest": RandomForestClassifier(n_jobs=-1),
                    "SGD": SGDClassifier(n_jobs=-1)
                }
                if not self.is_sparse:
                     models["HistGradientBoosting"] = HistGradientBoostingClassifier()
                     
                metrics = ["f1", "roc_auc", "accuracy"]
            else:
                task = "régression"
                # Pour regression sparse, SGDRegressor est top, RandomForest aussi
                models = { 
                    "RandomForestReg": RandomForestRegressor(n_jobs=-1, n_estimators=100),
                    "Ridge": TransformedTargetRegressor(
                        regressor=SGDClassifier(loss='squared_error') if len(self.df_solution) > 10000 else SVR(),
                        transformer=StandardScaler()
                    )
                }
                if not self.is_sparse:
                    models["HistGradientBoostingReg"] = TransformedTargetRegressor(
                        regressor=HistGradientBoostingRegressor(),
                        transformer=StandardScaler()
                    )
                    models["MLPReg"] = TransformedTargetRegressor(
                        regressor=MLPRegressor(max_iter=500),
                        transformer=StandardScaler()
                    )
                    
                metrics = ["rmse", "r2", "mae"]

        self.models = models
        return {"type": task, "models": models, "metrics": metrics}

    def _prepare_preprocessor(self):
        """
        Crée le pipeline de preprocessing.
        Gère différemment les DataFrames (Dense) et les Matrices (Sparse).
        """
        
        # === CAS SPARSE ===
        if self.is_sparse:
            # Pour une matrice creuse, on ne peut pas utiliser SimpleImputer(mean) facilement 
            # (car les zéros sont structurels) ni OneHotEncoder sur des colonnes inconnues.
            # On applique juste un Scaling adapté (MaxAbsScaler préserve la "sparsity", StandardScaler(with_mean=False) aussi)
            
            print(" -> Preprocessing mode Sparse activé.")
            self.preprocessor = Pipeline(steps=[
                # with_mean=False est CRUCIAL sinon la matrice devient dense et explose la RAM
                ('scaler', StandardScaler(with_mean=False)) 
            ])
            return

        # === CAS DENSE (DataFrame) ===
        numeric_features = []
        categorical_features = []
        bool_features = []

        types = self.df_probleme["Type"].values
        columns = self.df_data.columns

        for col, t in zip(columns, types):
            t_lower = t.lower()
            if "numerical" in t_lower:
                numeric_features.append(col)
            elif "categorical" in t_lower:
                categorical_features.append(col)
            elif "boolean" in t_lower:
                bool_features.append(col)

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')), # Remplacement par la moyenne
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        bool_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
                ("bool", bool_transformer, bool_features)
            ],
            remainder="drop"
        )
    
    def _fit_predict_models(self):
        for name, model in self.models.items():
            print(f"Entraînement de {name}...")
            try:
                # Si sparse, X_train_trans est une matrice sparse CSR
                model.fit(self.X_train_trans, self.y_train)
                y_pred = model.predict(self.X_test_trans)
                
                y_proba = None
                if hasattr(model, "predict_proba"):
                    try:
                        y_proba = model.predict_proba(self.X_test_trans)
                        if self.model_info["type"] == "classification_binaire" and y_proba.ndim == 2:
                            y_proba = y_proba[:, 1]
                    except:
                        pass
                
                self.results[name] = {
                    "model_obj": model,
                    "y_pred": y_pred,
                    "y_proba": y_proba,
                }
            except Exception as e:
                print(f"Echec entraînement {name}: {e}")
                # En cas d'erreur de dimension sur Sparse vs Dense pour certains algos
                import traceback
                traceback.print_exc()

    def _choisi_meilleur_model(self):
        best_model_name = None
        best_avg_score = -np.inf
        
        print("\n--- Résultats de l'évaluation ---")
        
        # Calcul de range pour normalisation métriques régression
        y_range = 1.0
        if "reg" in self.model_info["type"].lower():
             y_test_flat = np.ravel(self.y_test)
             y_range = np.max(y_test_flat) - np.min(y_test_flat)
             if y_range == 0: y_range = 1.0

        for name, data in self.results.items():
            y_p = data["y_pred"]
            y_prob = data["y_proba"]
            
            scores_norm = {}
            scores_bruts = {}
            
            for metric in self.model_info["metrics"]:
                val = 0
                try:
                    if metric == "accuracy":
                        val = accuracy_score(self.y_test, y_p)
                        scores_norm['accuracy'] = val
                    elif metric == "f1_macro":
                        val = f1_score(self.y_test, y_p, average="macro")
                        scores_norm['f1_macro'] = val
                    elif metric == "f1": # Binaire
                        val = f1_score(self.y_test, y_p, average="binary")
                        scores_norm['f1'] = val
                    elif metric == "roc_auc" and y_prob is not None:
                        if "multi-classe" in self.model_info["type"]:
                             val = roc_auc_score(self.y_test, y_prob, multi_class='ovr')
                        else:
                             val = roc_auc_score(self.y_test, y_prob)
                        scores_norm['AUC'] = val
                        scores_bruts["AUC"] = val
                    elif metric == "jaccard":
                        val = jaccard_score(self.y_test, y_p, average="samples", zero_division=0)
                        scores_norm['jaccard'] = val

                    # Régression
                    elif metric == "rmse":
                        rmse_val = root_mean_squared_error(self.y_test, y_p)
                        scores_bruts["RMSE"] = rmse_val
                        scores_norm['RMSE'] = max(0, 1 - (rmse_val / y_range))
                        
                    elif metric == "r2":
                        val = r2_score(self.y_test, y_p)
                        scores_bruts["R2"] = val
                        scores_norm['R2'] = max(0, val)

                except Exception as e:
                    pass

            vals = list(scores_norm.values())
            avg_score = np.mean(vals) if vals else -1

            details_str = ", ".join([f"{k}= {v:.3f}" for k, v in scores_bruts.items()])
            details_str = ", ".join([f"{k}= {v:.3f}" for k, v in scores_norm.items()])

            print(f"Modèle: {name:20} | Score: {avg_score:.4f} | Détails: {details_str}")
            
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                best_model_name = name

        print(f"\nMEILLEUR MODÈLE : {best_model_name} (Score: {best_avg_score:.4f})")
        return best_model_name
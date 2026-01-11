import numpy as np
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import MultinomialNB

def neg_root_mean_squared_error(y_true, y_pred):
    """Calcule le RMSE négatif pour la maximisation par Optuna."""
    return -np.sqrt(mean_squared_error(y_true, y_pred))

SCORING_MAP = {
    "accuracy": "accuracy", 
    "f1": "f1_macro", 
    "f1_macro": "f1_macro", 
    "roc_auc": "roc_auc_ovr", 
    "rmse": make_scorer(neg_root_mean_squared_error, greater_is_better=True), 
    "r2": "r2", 
    "mae": "neg_mean_absolute_error" 
}

def suggest_hyperparameters(trial, model_name):
    """
    Définit l'espace de recherche (search space) pour l'optimisation avec Optuna. 
    
    En fonction du modèle sélectionné, elle utilise l'objet 'trial' pour suggérer 
    des plages de valeurs spécifiques (numériques ou catégories) pour 
    chaque hyperparamètre.
    """
    
    if model_name in ['RandomForest', 'RandomForestReg']:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']) if model_name == 'RandomForest' else 'squared_error',
                'random_state': 42
        }

    elif model_name == 'MultinomialNB':
        return {
            'alpha': trial.suggest_float('alpha', 1e-5, 20.0, log=True),
            'fit_prior': trial.suggest_categorical('fit_prior', [True, False])
        }
    
    elif model_name == 'LogisticRegression':
        solver = trial.suggest_categorical('solver', ['lbfgs', 'saga'])
        penalty = trial.suggest_categorical('penalty', ['l2', 'l1']) if solver == 'saga' else 'l2'
        
        return {
            'penalty': penalty,
            'C': trial.suggest_float('C', 1e-5, 1e3, log=True),
            'solver': solver,
            'max_iter': 2000,
            'random_state': 42
        }
        
    elif model_name in ['SVC', 'SVR']:
        return {
            'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'degree': trial.suggest_int('degree', 2, 5) if trial.relative_params.get('kernel') == 'poly' else 3,
            'random_state': 42 
        }
        
    elif model_name in ['HistGradientBoosting', 'HistGradientBoostingReg']:
        return {
            'max_iter': trial.suggest_int('max_iter', 100, 500, step=50),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 15, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-6, 10.0, log=True),
            'random_state': 42
        }
        
    elif model_name in ['MLP', 'MLPReg']:
        return {
            'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', 
                [(100,), (100, 50), (50, 50, 50), (200, 100)]),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
            'max_iter': 1000, 
            'random_state': 42
        }
    
    elif model_name == 'SGD': 
         return {
             'loss': trial.suggest_categorical('loss', ['hinge', 'log_loss', 'modified_huber']),
             'alpha': trial.suggest_float('alpha', 1e-6, 1e-1, log=True),
             'penalty': trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet']),
             'learning_rate': trial.suggest_categorical('learning_rate', ['optimal', 'adaptive']),
             'eta0': trial.suggest_float('eta0', 1e-3, 1e-1, log=True),
             'max_iter': 2000,
             'random_state': 42
         }

    return {}

def get_base_model_class(model_name, is_classification):
    """
    Assure le mapping (faire de la correspondance) entre le nom du modèle (chaîne de caractères) 
    et sa classe réelle dans Scikit-learn. 
    
    Cette fonction est nécessaire pour l'instanciation dynamique (créer un objet à partir d'une classe) : 
    elle  permet à  l'optimiseur de récupérer le bon constructeur selon que la tâche est une classification
    ou une régression, afin de créer un nouvel objet modèle à chaque itération de recherche d'hyperparamètres.
    """
    
    if is_classification:
        if model_name == 'MultinomialNB': return MultinomialNB
        if model_name == 'RandomForest': return RandomForestClassifier
        if model_name == 'LogisticRegression': return LogisticRegression
        if model_name == 'SVC': return SVC
        if model_name == 'HistGradientBoosting': return HistGradientBoostingClassifier
        if model_name == 'MLP': return MLPClassifier
        if model_name == 'SGD': return SGDClassifier
    else: # Régression
        if model_name == 'RandomForestReg': return RandomForestRegressor
        if model_name == 'SVR': return SVR
        if model_name == 'HistGradientBoostingReg': return HistGradientBoostingRegressor
        if model_name == 'MLPReg': return MLPRegressor
        if model_name == 'SGD': return SGDRegressor 
        
    raise ValueError(f"Modèle non géré: {model_name} (Clas={is_classification})")

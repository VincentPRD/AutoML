import numpy as np
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor

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
    """ Logique de suggestion d'hyperparamètres optimisée. """
    
    if model_name in ['RandomForest', 'RandomForestReg']:
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 8, 20, log=True),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']) if model_name == 'RandomForest' else 'squared_error',
            'random_state': 42
        }
    
    elif model_name == 'LogisticRegression':
        params = {
            'penalty': trial.suggest_categorical('penalty', ['l2', None]), 
            'solver': 'lbfgs',
            'max_iter': 1000,
            'random_state': 42
        }
        if params['penalty'] == 'l2':
            params['C'] = trial.suggest_float('C', 1e-5, 1e2, log=True)
        return params
        
    elif model_name in ['SVC', 'SVR']:
        return {
            'C': trial.suggest_float('C', 1e-2, 1e2, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']), 
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'random_state': 42 
        }
        
    elif model_name in ['HistGradientBoosting', 'HistGradientBoostingReg']:
        return {
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 20, 30),
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-4, 1.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.15, log=True),
            'random_state': 42
        }
        
    elif model_name in ['MLP', 'MLPReg']:
        return {
            'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50, 50), (100,), (100, 50)]),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'alpha': trial.suggest_float('alpha', 1e-4, 1e-2, log=True),
            'max_iter': 1000, 
            'random_state': 42
        }
    
    elif model_name == 'SGD': 
         return {
             'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
             'penalty': trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet']),
             'max_iter': 1000,
             'random_state': 42
         }

    return {}

def get_base_model_class(model_name, is_classification):
    """ Renvoie la classe Scikit-learn selon le type de problème. """
    
    if is_classification:
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
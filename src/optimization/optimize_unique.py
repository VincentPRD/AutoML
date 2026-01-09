import numpy as np
import optuna
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.compose import TransformedTargetRegressor 
from sklearn.preprocessing import StandardScaler 
from sklearn.base import RegressorMixin, ClassifierMixin 

from .optimize_utils import SCORING_MAP, suggest_hyperparameters, get_base_model_class 

"""
Supprimer les warnings qui polluent l'affichage dans le notebook !
"""
warnings.filterwarnings(
    "ignore", 
    message="Setting penalty=None will ignore the C and l1_ratio parameters",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore", 
    message="The objective has been evaluated to None and no better trials are available",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore", 
    message="The max_iter parameter is deprecated and will be removed in 1.4.",
    category=FutureWarning
)
optuna.logging.set_verbosity(optuna.logging.WARN)

class AutoOptimizer:
    
    def objective(self, trial, X_train, y_train, task_type, target_metric_name, model_name):
        """
        Fonction objective pour Optuna. Elle construit le modèle avec les wrappers si nécessaire.

        Args : 
            Trial : ce qui va gérer le choix des hyperparamètres dans un intervalle
            X_train : les données d'entrainement
            y_train : les labels associés au données
            task_type : un dictionnaire contenant les informations relatives à la tâche
            target_metric_name : la métrique utilisé discriminer les itérations de l'optimisation

        Returns :
            Score du modèle pour une itération
        """
        
        is_multi_output = "multi-sortie" in task_type or "multi-label" in task_type
        is_classification = "classification" in task_type
        
        params = suggest_hyperparameters(trial, model_name)
        BaseEstimatorClass = get_base_model_class(model_name, is_classification)
        
        base_estimator = BaseEstimatorClass(**params)
        model = base_estimator
        
        if is_multi_output:
            if is_classification:
                if not hasattr(base_estimator, 'classes_'): 
                    model = MultiOutputClassifier(base_estimator)
            elif not hasattr(base_estimator, 'n_outputs_'):
                 model = MultiOutputRegressor(base_estimator)

        if not is_classification and not is_multi_output:
            if model_name in ['SVR', 'HistGradientBoostingReg', 'MLPReg']:
                model = TransformedTargetRegressor(regressor=base_estimator, transformer=StandardScaler())

        if model_name in ['HistGradientBoosting', 'HistGradientBoostingReg']:
            if hasattr(X_train, 'toarray'):
                X_train = X_train.toarray()
            elif hasattr(X_train, 'values'):
                 X_train = X_train.values
            
        scoring_metric = SCORING_MAP.get(target_metric_name, "accuracy")
        
        try:
            score = cross_val_score(model, X_train, y_train, n_jobs=-1, cv=3, scoring=scoring_metric)
            return score.mean()
        except Exception:
            return None 

    def optimize(self, X_train, y_train, task_info, model_name_to_optimize, n_trials=50):
        """
        Exécute une étude Optuna pour optimiser le modèle spécifié.

        Args : 
            X_train : les données d'entrainement
            y_train : les labels associés au données
            task_info : un dictionnaire contenant les informations relatives à la tâche (contient les métriques, ...)
            model_name_to_optimize : le nom du modèle à optimiser
            n_trials : le nombre d'itération = nombre de fois que la fonction objective sera appelé.
            
        Returns :
            Le nom du meilleur modèle et ses meilleurs paramètres avec les n_trials itérations
        """
        
        model_name = model_name_to_optimize
        task_type = task_info['type']
        target_metric_name = task_info['metrics'][0] 
        
        print(f"\nDémarrage de l'optimisation pour le modèle: {model_name} ({n_trials} irérations)")
        
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: self.objective(
                trial, X_train, y_train, task_type, target_metric_name, model_name
            ),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        best_trial = study.best_trial if study.best_trials else None
        
        if not best_trial or best_trial.state != optuna.trial.TrialState.COMPLETE:
            print("\nAucun essai n'a été complété avec succès. Retour des paramètres vides.")
            best_global_score = None
            best_params = {}
        else:
            best_global_score = study.best_value
            best_params = study.best_params
            
            print("\n=============================================")
            print(f"OPTIMISATION TERMINÉE pour: {model_name}")
            print("=============================================")
            
        return model_name, best_params
# Machine Learning Challenge – AutoML Pipeline

## Contexte
Ce projet s’inscrit dans le cadre du module **Méthodologie IA et Méthodes Classiques (M1 INFO – IA)**.  
L’objectif est de développer une **pipeline AutoML** permettant d’automatiser :
- La préparation et séparation des données (train/dev/test)  
- L’apprentissage de différents modèles avec **scikit-learn**  
- L’optimisation des hyperparamètres
- L’évaluation des modèles avec des métriques adaptées

---

## Fonctionnalités principales
- Chargement et prétraitement des jeux de données  
- Sélection automatique de modèles (classification, régression, etc.)  
- Optimisation des hyperparamètres (optuna)  
- Évaluation avec métriques adaptées (accuracy, F1-score, RMSE, etc.)
- Prédiction automatique sur de nouveaux jeux de données via ```predict()```
- Interface utilisateur :
```python
import automl

automl = automl.AutoML()
data_dest_traindev="/info/corpus/ChallengeMachineLearning/data_.../"
automl.fit(data_dest_traindev)
automl.eval()
path_to_testset = "/info/corpus/ChallengeMachineLearning/data_../"
automl.predict(path_to_testset)
```

## Installation des dépendances
```python
pip install -r requirements.txt
```

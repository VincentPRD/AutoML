# Machine Learning Challenge – AutoML Pipeline

## Contexte
Ce projet s’inscrit dans le cadre du module **Méthodologie IA et Méthodes Classiques (M1 INFO – IA)**.  
L’objectif est de développer une **pipeline AutoML** permettant d’automatiser :
- La préparation et séparation des données (train/valid/test)  
- L’apprentissage de différents modèles avec **scikit-learn**  
- L’optimisation des hyperparamètres  
- L’évaluation des modèles avec des métriques adaptées

---

## Fonctionnalités principales
- Chargement et prétraitement des jeux de données  
- Sélection automatique de modèles (classification, régression, etc.)  
- Optimisation des hyperparamètres (GridSearchCV, RandomizedSearchCV, etc.)  
- Évaluation avec métriques adaptées (accuracy, F1-score, RMSE, etc.)  
- Interface utilisateur minimale :
```python
import automl

data_dest = "/path/to/data"
automl.fit(data_dest)
automl.eval()


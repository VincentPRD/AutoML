import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split # Permet de séparer un jeu de données en un ensemble d'entrainements et un ensemble de tests.
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix # Matrice de confusion pour visualiser les performances du modèle
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import MinMaxScaler

BINAIRE = 0
MONOLABEL = 1
MULTICLASSES = 2
MULTILABELS = 3
REGRESSION = 4

class AutoML:
    
    #
    # Chargement des données.
    #
    def chargement_donnees(self, data_dest):
        
        #
        # Affecte dans un dictionnaire les différents fichiers à un mot clé.
        #

        chemin = "/info/corpus/ChallengeMachineLearning/" + data_dest + "/"
        
        fichiers = {
            'data' : chemin + os.path.basename(data_dest) + '.data', 
            'solution' : chemin + os.path.basename(data_dest) + '.solution',
            'type' : chemin + os.path.basename(data_dest) + '.type'
        }

        self.df_data = pd.read_csv(fichiers['data'], sep=' ', header=None) 
        self.df_data = self.df_data.dropna(axis=1, how='all') # Supprime les colonnes qui ne possède que des valeurs manquantes.
        
        self.df_solution = pd.read_csv(fichiers['solution'], sep=' ', header=None) 
        self.df_solution = self.df_solution.dropna(axis=1, how='all') # Supprime les colonnes qui ne possède que des valeurs manquantes.
        self.df_solution = self.df_solution.rename(columns={0 : 'A'})

    #
    # Enlève les lignes qui ne possèdent pas certaines valeurs.
    #
    def nettoyage_donnees(self):
        df = pd.concat([self.df_data, self.df_solution], axis=1)

        df = df.dropna() # Supprime les lignes vides.
        df = df.dropna(axis=1) # Supprime les colonnes vides.

        df = df.fillna(df.mean()) # Remplace les valeurs manquantes.

        df = df.drop_duplicates() # Supprime les doublons.

        #
        # Normalise les données.
        #
        df.columns = df.columns.astype(str)
        scaler = MinMaxScaler()
        df[df.columns] = scaler.fit_transform(df[df.columns])
        
        self.df_data = df.loc[:, :'A'].drop(columns='A')
        self.df_solution = df.loc[:, 'A':]

    def cherche_type_probleme(self):
        nb_colonnes = self.df_solution.shape[1]

        if((nb_colonnes > 1) and (self.df_solution.sum(axis=1) == 1).all()):
            print("Il s'agit d'un problème de classification multiclasses !\n")
            self.probleme = MULTICLASSES
        elif(nb_colonnes > 1):  
            print("Il s'agit d'un problème de classification multilabels !\n")
            self.probleme = MULTILABELS
        elif(self.df_solution.value_counts().shape[0] == 2):
            print("Il s'agit d'un problème de classification binaire !\n")
            self.probleme = BINAIRE
        elif(self.df_solution.value_counts().shape[0] <= 10):
            print("Il s'agit d'un problème de classification monolabel !\n")
            self.probleme = MONOLABEL
        else:
            print("Il s'agit d'un problème de régression !\n")
            self.probleme = REGRESSION

    def fit_binaire_monolabel(self):
        self.modele = RandomForestClassifier(random_state=42)
        self.modele.fit(self.X_train, self.y_train.squeeze())

    def fit_multiclasses(self):
        self.modele = RandomForestClassifier(random_state=42)
        self.modele.fit(self.X_train, self.y_train)

    def fit_multilabels(self):
        self.modele = RandomForestClassifier(random_state=42)
        self.modele = MultiOutputClassifier(self.modele)
        self.modele.fit(self.X_train, self.y_train)    
        
    def fit_regression(self):
        self.modele = RandomForestRegressor(random_state=42)
        self.modele.fit(self.X_train, self.y_train.squeeze())

    #
    # Préparation des données et séparation en sous ensemble train et test.
    #
    def fit(self, data_dest):
        self.chargement_donnees(data_dest)
        self.nettoyage_donnees()
        self.cherche_type_probleme()
        
        #
        # Sépare les données en sous ensemble train et test.
        #
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df_data, self.df_solution, test_size=0.3, random_state=42)

        if(self.probleme == BINAIRE or self.probleme == MONOLABEL):
            self.fit_binaire_monolabel()
        elif(self.probleme == MULTICLASSES):
            self.fit_multiclasses()
        elif(self.probleme == MULTILABELS):
            self.fit_multilabels()
        else:
            self.fit_regression()
                
        self.y_pred = self.modele.predict(self.X_test) # Prédictions sur l'ensemble de test.

    def eval(self):

        #
        # Évaluation du modèle.
        #
        if(self.probleme == REGRESSION):
            print(f"Mean Squared Error (MSE) : {mean_squared_error(self.y_test, self.y_pred)}")
            print(f"R² Score : {r2_score(self.y_test, self.y_pred)}")
        else:
            print(classification_report(self.y_test, self.y_pred, zero_division=0))
            print(f"Accuracy : {accuracy_score(self.y_test, self.y_pred)}")

            matrices_confusions = multilabel_confusion_matrix(self.y_test, self.y_pred)
    
            #
            # Visualisation des matrices de confusions.
            #
            for i, matrice_confusion in enumerate(matrices_confusions):
                sns.heatmap(matrice_confusion, annot=True, fmt="d", cmap="Blues")
                plt.title(f"Matrice de confusion pour le label {i}")
                plt.ylabel("Labels réels")
                plt.xlabel("Labels prédits")
                plt.show()





















        
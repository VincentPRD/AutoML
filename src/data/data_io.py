"""
Module de gestion des Entrées/Sorties (I/O) pour les données AutoML.

Ce module gère le chargement des datasets à partir de fichiers, en détectant
automatiquement le format des données (Dense CSV ou Sparse LibSVM) et en assurant
la cohérence entre les features (.data) et les targets (.solution).
"""

import os
import traceback
from io import BytesIO
from typing import Tuple, Optional, Union

import pandas as pd
import scipy.sparse
from sklearn.datasets import load_svmlight_file


def _chargement_sparse_non_label(filepath: str) -> scipy.sparse.spmatrix:
    """
    Charge un fichier au format LibSVM ne contenant pas de colonne cible (target) au début.

    L'implémentation standard de `sklearn.load_svmlight_file` s'attend à trouver 
    une étiquette au début de chaque ligne (ex: "1 4:0.5 ..."). Si le fichier commence
    directement par les features (ex: "4:0.5 ..."), cette fonction ajoute artificiellement
    une étiquette '0' dummy en mémoire avant de parser le fichier.

    Args:
        filepath (str): Chemin vers le fichier de données .data.

    Returns:
        scipy.sparse.spmatrix: La matrice des features au format CSR (Compressed Sparse Row).

    Raises:
        Exception: Si une erreur de lecture ou de parsing survient.
    """
    try:
        with open(filepath, 'rb') as f:
            content = f.read()
        
        # Ajout d'un dummy label '0 ' au début du fichier et après chaque saut de ligne.
        # Exemple transformation : "12:0.5 45:1\n14:0.2..." -> "0 12:0.5 45:1\n0 14:0.2..."
        new_content = b'0 ' + content.replace(b'\n', b'\n0 ')
        
        # Nettoyage : si le fichier original finissait par une ligne vide, 
        # le replace précédent a pu ajouter un "0 " orphelin à la fin qu'il faut retirer.
        if new_content.endswith(b'\n0 '):
            new_content = new_content[:-3]
            
        # Utilisation de BytesIO pour simuler un fichier en mémoire sans écriture disque
        f_obj = BytesIO(new_content)
        
        # On charge les données et on ignore le vecteur target (le dummy label) retourné en 2ème position
        X, _ = load_svmlight_file(f_obj)
        return X
        
    except Exception as e:
        print(f"Erreur lors du chargement sparse manuel : {e}")
        raise e


def _detecte_et_charge_data(filepath: str) -> Tuple[Optional[Union[pd.DataFrame, scipy.sparse.spmatrix]], bool]:
    """
    Analyse un fichier pour déterminer son format (Dense vs Sparse) et charge les données.

    L'heuristique repose sur la présence de deux points ":" dans la première ligne
    pour identifier le format LibSVM (Sparse).

    Args:
        filepath (str): Chemin complet vers le fichier .data.

    Returns:
        tuple: Un tuple contenant :
            - data (pd.DataFrame | scipy.sparse.spmatrix | None): Les données chargées.
            - is_sparse (bool): Vrai si le format est Sparse, Faux sinon.
    """
    # Lecture de la première ligne pour analyse heuristique
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
        
    if not first_line:
        print(" -> Attention : Fichier de données vide détecté.")
        return None, False

    tokens = first_line.split()
    
    # Indicateurs de format
    has_colon = ":" in first_line
    # Si le premier token contient ":", c'est que la ligne commence par une feature (ex: "1:0.5")
    # et non par un label (ex: "1 1:0.5").
    first_token_is_feature = ":" in tokens[0] if tokens else False
    
    if has_colon:
        # --- Cas SPARSE (Matrice Creuse) ---
        if first_token_is_feature:
            # Format Sparse sans label initial -> Utilisation du hack
            return _chargement_sparse_non_label(filepath), True
        else:
            # Format Sparse standard (avec label au début)
            try:
                data, _ = load_svmlight_file(filepath)
                return data, True
            except ValueError:
                # Fallback : Parfois le "label" n'est pas parsable par sklearn
                # On tente alors de charger comme si c'était du sans-label
                return _chargement_sparse_non_label(filepath), True
    else:
        # --- Cas DENSE (DataFrame Pandas) ---
        # On suppose un format séparé par des espaces
        data = pd.read_csv(filepath, sep=' ', header=None)
        
        # Nettoyage : suppression des colonnes entièrement vides (souvent dues aux espaces finaux)
        data.dropna(axis=1, how='all', inplace=True)
        
        # Conversion des noms de colonnes en str pour cohérence avec le reste du pipeline
        data.columns = data.columns.astype(str)
        return data, False


def chargement_donnees_complet(
    data_dest: str, 
    load_solution: bool = True
) -> Tuple[Optional[Union[pd.DataFrame, scipy.sparse.spmatrix]], Optional[pd.DataFrame], Optional[pd.DataFrame], bool]:
    """
    Orchestre le chargement de l'ensemble des fichiers d'un dataset (Data, Solution, Type).

    Cette fonction reconstruit les chemins des fichiers attendus (.data, .solution, .type)
    à partir du dossier racine fourni, charge les données, et vérifie la cohérence
    dimensionnelle.

    Args:
        data_dest (str): Chemin vers le dossier contenant les fichiers du dataset.
        load_solution (bool, optional): Si True, tente de charger le fichier .solution.
                                        Utile de mettre False pour la phase de prédiction.
                                        Défaut à True.

    Returns:
        tuple: (df_data, df_solution, df_probleme, is_sparse)
            - df_data : Les features d'entraînement/test.
            - df_solution : La cible (target) ou None si non chargée/trouvée.
            - df_probleme : Les métadonnées des colonnes (.type) ou None.
            - is_sparse : Booléen indiquant si df_data est une matrice creuse.

    Raises:
        Exception: 
            - Si le fichier .data ne fait pas la même taille que le fichier .solution
            - Traceback si on a pas réussi à charger les données.
    """
    df_data = None
    df_solution = None
    df_probleme = None
    is_sparse = False
    
    try:
        # Extraction du nom du dataset à partir du chemin
        # Gère les chemins finissant par '/' ou non
        name = data_dest.rstrip('/').split(sep='/')[-1]
        if name == "": 
            name = data_dest.split(sep='/')[-2]

        # Construction des chemins de fichiers attendus
        fichiers = {
            'data': os.path.join(data_dest, name + '.data'),
            'solution': os.path.join(data_dest, name + '.solution'),
            'type': os.path.join(data_dest, name + '.type')
        }

        # Chargement des données (.data)
        if not os.path.exists(fichiers['data']):
            print(f"Erreur : Fichier introuvable -> {fichiers['data']}")
            return None, None, None, False
            
        df_data, is_sparse = _detecte_et_charge_data(fichiers['data'])
        
        # Chargement des métadonnées (.type)
        if os.path.exists(fichiers['type']):
            df_probleme = pd.read_csv(fichiers['type'], sep=r'\s+', header=None, names=['Type'])
            
        # Chargement de la solution (.solution)
        if load_solution:
            if not os.path.exists(fichiers['solution']):
                # Si le fichier solution n'est pas trouvé ce n'est pas bloquant
                print(f"Info : Fichier solution absent -> {fichiers['solution']}")
                pass
            else:
                df_solution = pd.read_csv(fichiers['solution'], sep=' ', header=None)
                df_solution.dropna(axis=1, how='all', inplace=True)
                
                # Vérification de cohérence (Nombre de lignes)
                if df_data is not None and df_data.shape[0] != len(df_solution):
                    raise ValueError(
                        f"Incohérence de taille : Data a {df_data.shape[0]} lignes, "
                        f"mais Solution en a {len(df_solution)}."
                    )
            
        return df_data, df_solution, df_probleme, is_sparse

    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        traceback.print_exc()
        return None, None, None, False
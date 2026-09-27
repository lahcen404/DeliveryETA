# 🚚 DeliveryETA

Application Streamlit utilisant le Machine Learning pour prédire le temps de livraison d’une commande.

L’utilisateur renseigne les informations relatives au livreur, à la commande, à la météo et au trafic. Le modèle estime ensuite la durée de livraison en minutes.

## Fonctionnalités

- Saisie des informations de livraison.
- Prise en compte de la distance et du délai de récupération.
- Prise en compte de la météo, du trafic et du véhicule.
- Prédiction du temps de livraison.
- Affichage des métriques du modèle.
- Visualisation de la distribution des temps de livraison.

## Dataset

Le dataset utilisé est disponible dans :

```text
data/selected_features_data.csv
```

Les principales variables utilisées sont :

- Âge et note du livreur
- Nombre de livraisons simultanées
- Distance en kilomètres
- Conditions météorologiques
- Densité du trafic
- Type de véhicule
- Jour de la semaine
- Heure de commande
- Heure de récupération
- Délai de récupération
- Mois

La variable cible est :

```text
Time_taken(min)
```

## Démarche

### Préparation des données

- Sélection des variables pertinentes.
- Conversion du mois en valeur numérique.
- Normalisation des variables numériques.
- Encodage One-Hot des variables catégorielles.
- Respect de l’ordre exact des variables utilisé lors de l’entraînement.

### Modèle

Le modèle utilisé est un `Random Forest Regressor`, optimisé avec `RandomizedSearchCV`.

Les fichiers du modèle sont stockés dans :

```text
models/delivery_eta_pipeline.joblib
models/scaler.joblib
```

## Résultats et métriques

| Métrique | Résultat |
|---|---:|
| MAE | 3,57 minutes |
| RMSE | 4,55 minutes |
| R² | 76,73 % |

### Interprétation

- **MAE** : erreur moyenne d’environ 3,57 minutes.
- **RMSE** : erreur quadratique moyenne de 4,55 minutes.
- **R²** : le modèle explique 76,73 % de la variance des temps de livraison.

## Technologies utilisées

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib

## Installation

### Prérequis

- Python 3.9 ou supérieur
- Git

### Cloner le projet

```bash
git clone <URL_DU_REPOSITORY>
cd DeliveryETA
```

### Créer un environnement virtuel

Sous Linux :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Installer les dépendances

```bash
pip install streamlit pandas numpy joblib matplotlib scikit-learn
```

## Exécution

Depuis la racine du projet :

```bash
streamlit run app/app.py
```

L’application sera accessible à l’adresse :

```text
http://localhost:8501
```

## Structure du projet

```text
DeliveryETA/
├── app/
│   └── app.py
├── data/
│   └── selected_features_data.csv
├── models/
│   ├── delivery_eta_pipeline.joblib
│   └── scaler.joblib
├── screenshots/
│   ├── prediction.png
│   ├── performance.png
│   └── distribution.png
└── README.md
```

## Captures d’écran

### Formulaire de prédiction

![Formulaire de prédiction](screenshots/prediction.png)

### Performances du modèle

![Performances du modèle](screenshots/performance.png)

### Distribution des temps de livraison

![Distribution des temps de livraison](screenshots/distribution.png)

> Ajoutez les captures d’écran dans le dossier `screenshots/` avec les noms indiqués ci-dessus.
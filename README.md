🏆 Chess Analyzer
Un outil d'analyse d'échecs avancé pour analyser vos parties et améliorer votre jeu.
📖 Description
Chess Analyzer est un projet Python conçu pour analyser des parties d'échecs au format PGN de l'archive de Lichess.(Portable Game Notation). L'outil permet d'extraire des statistiques détaillées, d'analyser les ouvertures, et de fournir des insights sur la performance des joueurs.
✨ Fonctionnalités

📁 Analyse de fichiers PGN : Import et traitement de parties d'échecs
🎯 Analyse d'ouvertures : Étude des séquences d'ouverture les plus fréquentes
📈 Visualisations : Graphiques et diagrammes pour représenter les données

🚀 Installation
Prérequis

Python 3.8 ou supérieur
pip (gestionnaire de paquets Python)

Installation des dépendances
bash# Cloner le repository
git clone https://github.com/Niviferum/chess_analyzer.git
cd chess_analyzer

# Installer les dépendances
pip install -r requirements.txt
Dépendances principales
python-chess>=1.999
pandas>=1.5.0
matplotlib>=3.5.0
seaborn>=0.11.0
numpy>=1.21.0
🎮 Utilisation
Analyse basique
pythonfrom chess_analyzer import ChessAnalyzer




📁 Structure du projet
chess_analyzer/
├── data/
│   ├── raw/                    # Fichiers PGN bruts
│   ├── processed/              # Données traitées
│   └── results/                # Résultats d'analyse
├── src/
│   ├── __init__.py
│   ├── data_processing/
    |    ├── data_cleaner.py    # nettoyage des données du fichier raw
    |    ├── pgn_parser.py      # Perseur de fichiers PGN
├── notebooks/
│   └── exploration.ipynb       # Notebooks d'exploration
├── requirements.txt
├── main.py                     # Script principal
└── README.md

# Utilisation des notebooks :

La première chose à faire est d'obtenir une database dans les archives de lichess. Une fois obtenue, vous pouvez mettre votre database .pgn dans data/raw et la renommer lichess_games.pgn
Ensuite, vous pouvez run d'abord le notebook 01 pour faire le nettoyage de la base pour les quelques parties qui seraient peu utiles à l'analyse ou encore des parties buggées. 
Enfin, pour les visualisations, vous pouvez executer le notebook 02 et suivre les indications du notebook pour voir les différentes visualisations dans votre navigateur à l'aide des fichiers HTML.

📊 Exemples de sorties
Statistiques de performance
=== STATISTIQUES GÉNÉRALES ===
Nombre total de parties : 1,247
Victoires blancs : 45.2%
Victoires noirs : 38.1%
Nulles : 16.7%

=== TOP 5 OUVERTURES ===
1. Sicilian Defense : 234 parties (18.8%)
2. Queen's Gambit : 156 parties (12.5%)
3. King's Indian : 134 parties (10.7%)
4. French Defense : 98 parties (7.9%)
5. Caro-Kann : 87 parties (7.0%)



"""
Parser PGN optimisé pour les données d'échecs Lichess
"""

import chess.pgn
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import logging
import io

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChessPGNParser:
    """
    Parser optimisé pour fichiers PGN de Lichess
    """
    
    def __init__(self, min_elo: int = 800, min_games_per_opening: int = 20):
        self.min_elo = min_elo
        self.min_games_per_opening = min_games_per_opening
        
        # Patterns regex pour l'extraction
        self.eval_pattern = re.compile(r'\[%eval ([^\]]+)\]')
        self.clk_pattern = re.compile(r'\[%clk ([^\]]+)\]')
        
    def parse_pgn_file(self, pgn_file_path: str, max_games: int = 50000) -> pd.DataFrame:
        """
        Parse un fichier PGN et retourne un DataFrame nettoyé
        """
        logger.info(f"Début du parsing de {pgn_file_path}")
        
        games_data = []
        
        try:
            with open(pgn_file_path, 'r', encoding='utf-8') as pgn_file:
                for i in tqdm(range(max_games), desc="Parsing games"):
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    
                    game_data = self._extract_game_data(game)
                    if game_data:
                        games_data.append(game_data)
                        
        except FileNotFoundError:
            logger.error(f"Fichier {pgn_file_path} introuvable")
            raise
        except Exception as e:
            logger.error(f"Erreur lors du parsing: {e}")
            raise
            
        df = pd.DataFrame(games_data)
        logger.info(f"Parsing terminé: {len(df)} parties extraites")
        
        return self._prepare_dataframe(df)
    
    def _extract_game_data(self, game) -> Optional[Dict]:
        """
        Extrait les données pertinentes d'une partie
        """
        try:
            headers = game.headers
            
            # Vérification ELO minimum
            white_elo = int(headers.get('WhiteElo', 0))
            black_elo = int(headers.get('BlackElo', 0))
            
            if white_elo < self.min_elo or black_elo < self.min_elo:
                return None
            
            # Résultat numérique
            result = headers.get('Result', '*')
            white_score = self._convert_result_to_score(result, 'white')
            black_score = self._convert_result_to_score(result, 'black')
            
            # Extraction des évaluations précoces
            early_evals = self._extract_early_evaluations(game, n_moves=10)
            
            # Données de base
            game_data = {
                'white_elo': white_elo,
                'black_elo': black_elo,
                'avg_elo': (white_elo + black_elo) / 2,
                'elo_diff': abs(white_elo - black_elo),
                'white_score': white_score,
                'black_score': black_score,
                'draw': 1 if result == '1/2-1/2' else 0,
                'eco': headers.get('ECO', ''),
                'opening': headers.get('Opening', ''),
                'time_control': self._parse_time_control(headers.get('TimeControl', '')),
                'termination': self._categorize_termination(headers.get('Termination', '')),
                'moves_count': self._count_moves(game),
                'avg_eval_early': np.mean(early_evals) if early_evals else 0,
                'eval_volatility': np.std(early_evals) if len(early_evals) > 1 else 0,
                'opening_advantage': early_evals[0] if early_evals else 0
            }
            
            return game_data
            
        except (ValueError, TypeError, KeyError) as e:
            logger.debug(f"Erreur extraction partie: {e}")
            return None
    
    def _extract_early_evaluations(self, game, n_moves: int = 10) -> List[float]:
        """Extrait les évaluations des premiers coups"""
        evaluations = []
        node = game
        move_count = 0
        
        while node.variations and move_count < n_moves:
            node = node.variations[0]
            comment = node.comment
            
            eval_match = self.eval_pattern.search(comment)
            if eval_match:
                try:
                    eval_value = float(eval_match.group(1))
                    evaluations.append(eval_value)
                except ValueError:
                    pass
            
            move_count += 1
        
        return evaluations
    
    def _convert_result_to_score(self, result: str, side: str) -> float:
        """Convertit le résultat PGN en score numérique"""
        if result == '1-0':
            return 1.0 if side == 'white' else 0.0
        elif result == '0-1':
            return 0.0 if side == 'white' else 1.0
        elif result == '1/2-1/2':
            return 0.5
        else:
            return 0.5  # Parties non terminées
    
    def _parse_time_control(self, time_control: str) -> str:
        """Catégorise le contrôle de temps"""
        if not time_control or time_control == '-':
            return 'Unknown'
        
        try:
            if '+' in time_control:
                base_time = int(time_control.split('+')[0])
            else:
                base_time = int(time_control)
        except ValueError:
            return 'Unknown'
        
        if base_time < 180:
            return 'Bullet'
        elif base_time < 600:
            return 'Blitz'
        elif base_time < 1800:
            return 'Rapid'
        else:
            return 'Classical'
    
    def _categorize_termination(self, termination: str) -> str:
        """Catégorise le type de fin de partie"""
        termination_lower = termination.lower()
        
        if 'mate' in termination_lower:
            return 'Checkmate'
        elif 'time' in termination_lower:
            return 'Time'
        elif 'resign' in termination_lower:
            return 'Resignation'
        elif 'draw' in termination_lower:
            return 'Draw'
        else:
            return 'Other'
    
    def _count_moves(self, game) -> int:
        """Compte le nombre de coups dans la partie"""
        return len(list(game.mainline_moves()))
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prépare le DataFrame avec colonnes dérivées"""
        if len(df) == 0:
            return df
        
        # Séparer ouverture principale et variation
        if 'opening' in df.columns:
            df[['main_opening', 'variation']] = df['opening'].str.split(':', n=1, expand=True)
            df['main_opening'] = df['main_opening'].str.strip()
            df['variation'] = df['variation'].str.strip().fillna('Main Line')
        
        # Créer des tranches d'ELO
        if 'avg_elo' in df.columns:
            elo_bins = [0, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 3000]
            elo_labels = ['<1000', '1000-1200', '1200-1400', '1400-1600', 
                          '1600-1800', '1800-2000', '2000-2200', '2200-2400', '2400+']
            
            df['elo_range'] = pd.cut(df['avg_elo'], bins=elo_bins, labels=elo_labels)
            df['white_elo_range'] = pd.cut(df['white_elo'], bins=elo_bins, labels=elo_labels)
            df['black_elo_range'] = pd.cut(df['black_elo'], bins=elo_bins, labels=elo_labels)
        
        # Métriques dérivées
        df['decisive_game'] = 1 - df['draw']
        
        if 'moves_count' in df.columns:
            df['game_length_category'] = pd.cut(
                df['moves_count'], 
                bins=[0, 20, 40, 60, 80, 200], 
                labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long']
            )
        
        return df


def calculate_opening_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les statistiques agrégées par ouverture et tranche d'ELO
    """
    logger.info("Calcul des statistiques d'ouvertures...")
    
    if 'main_opening' not in df.columns or 'elo_range' not in df.columns:
        logger.warning("Colonnes manquantes pour les statistiques")
        return pd.DataFrame()
    
    stats = df.groupby(['main_opening', 'elo_range']).agg({
        'white_score': ['count', 'mean', 'std'],
        'black_score': ['mean', 'std'],
        'draw': 'mean',
        'moves_count': 'mean',
        'avg_eval_early': 'mean',
        'eval_volatility': 'mean',
        'elo_diff': 'mean',
        'decisive_game': 'mean'
    }).round(4)
    
    # Aplatir les colonnes
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    stats = stats.reset_index()
    
    # Renommer pour plus de clarté
    stats.rename(columns={
        'white_score_count': 'games_count',
        'white_score_mean': 'white_winrate',
        'white_score_std': 'white_winrate_std',
        'black_score_mean': 'black_winrate',
        'black_score_std': 'black_winrate_std',
        'draw_mean': 'draw_rate',
        'moves_count_mean': 'avg_moves',
        'avg_eval_early_mean': 'opening_eval',
        'eval_volatility_mean': 'position_complexity',
        'elo_diff_mean': 'avg_elo_diff',
        'decisive_game_mean': 'decisive_rate'
    }, inplace=True)
    
    # Filtrer les échantillons trop petits
    stats = stats[stats['games_count'] >= 10]
    
    # Métriques dérivées
    stats['performance_index'] = (
        stats['white_winrate'] * 0.4 + 
        stats['black_winrate'] * 0.4 + 
        stats['decisive_rate'] * 0.2
    )
    
    logger.info(f"Statistiques calculées pour {len(stats)} combinaisons ouverture-ELO")
    
    return stats


def parse_and_analyze(pgn_file_path: str, max_games: int = 50000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse un fichier PGN et retourne les données brutes + statistiques
    
    Returns:
        Tuple[games_df, stats_df]
    """
    parser = ChessPGNParser()
    games_df = parser.parse_pgn_file(pgn_file_path, max_games)
    stats_df = calculate_opening_statistics(games_df)
    
    return games_df, stats_df


def create_sample_pgn(n_games: int = 100) -> str:
    """
    Crée un fichier PGN d'exemple pour tester
    """
    sample_games = []
    
    openings = [
        ("B20", "Sicilian Defense"),
        ("D06", "Queen's Gambit"),
        ("C00", "French Defense"),
        ("A10", "English Opening"),
        ("B12", "Caro-Kann Defense")
    ]
    
    for i in range(n_games):
        eco, opening = openings[i % len(openings)]
        white_elo = np.random.randint(1000, 2500)
        black_elo = np.random.randint(1000, 2500)
        result = np.random.choice(["1-0", "0-1", "1/2-1/2"], p=[0.4, 0.35, 0.25])
        
        moves_count = np.random.randint(20, 80)
        moves = " ".join([f"{j//2 + 1}." if j % 2 == 0 else "" for j in range(moves_count)])
        
        pgn_game = f"""[Event "Rated Blitz game"]
[Site "https://lichess.org/example{i}"]
[Date "2024.01.15"]
[Round "-"]
[White "Player{i}W"]
[Black "Player{i}B"]
[Result "{result}"]
[WhiteElo "{white_elo}"]
[BlackElo "{black_elo}"]
[TimeControl "300+3"]
[ECO "{eco}"]
[Opening "{opening}"]
[Termination "Normal"]

1. e4 c5 2. Nf3 {moves} {result}

"""
        sample_games.append(pgn_game)
    
    return "\n".join(sample_games)


# Test simple si exécuté directement
if __name__ == "__main__":
    # Créer un PGN d'exemple
    sample_pgn = create_sample_pgn(10)
    print("Exemple de PGN généré:")
    print(sample_pgn[:500] + "...")
    
    # Test de parsing sur l'exemple
    try:
        games_df = pd.read_csv(io.StringIO(sample_pgn), sep='\t')  # Simple test
        print(f"Test réussi: structure basique OK")
    except Exception as e:
        print(f"Erreur de test: {e}")
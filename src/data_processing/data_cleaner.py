# src/data_processing/data_cleaner.py

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChessDataCleaner:
    """
    Nettoyeur de données d'échecs avec validation et standardisation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise le nettoyeur avec configuration personnalisable
        
        Args:
            config: Configuration personnalisée pour le nettoyage
        """
        # Configuration par défaut
        self.config = {
            'min_elo': 700,
            'max_elo': 3500,
            'min_games_per_opening': 5,
            'min_moves': 5,
            'max_moves': 200,
            'valid_results': ['1-0', '0-1', '1/2-1/2'],
            'time_controls': {
                'bullet': (0, 180),
                'blitz': (180, 600),
                'rapid': (600, 1800),
                'classical': (1800, float('inf'))
            },
            'remove_bots': True,
            'remove_variants': True,
            'standardize_openings': True
        }
        
        if config:
            self.config.update(config)
        
        # Statistiques de nettoyage
        self.cleaning_stats = {
            'initial_games': 0,
            'invalid_elo': 0,
            'invalid_result': 0,
            'invalid_moves': 0,
            'bot_games': 0,
            'variant_games': 0,
            'duplicate_games': 0,
            'rare_openings': 0,
            'final_games': 0
        }
    
    def clean_dataset(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Pipeline complet de nettoyage
        
        Args:
            df: DataFrame brut des parties
            verbose: Afficher les statistiques de nettoyage
            
        Returns:
            DataFrame nettoyé
        """
        if verbose:
            logger.info("🧹 DÉBUT DU NETTOYAGE DES DONNÉES")
            logger.info("=" * 50)
        
        self.cleaning_stats['initial_games'] = len(df)
        
        # Pipeline de nettoyage
        df_clean = df.copy()
        
        # 1. Nettoyage des colonnes essentielles
        df_clean = self._clean_basic_columns(df_clean)
        
        # 2. Validation des ELO
        df_clean = self._validate_elo(df_clean)
        
        # 3. Validation des résultats
        df_clean = self._validate_results(df_clean)
        
        # 4. Validation du nombre de coups
        df_clean = self._validate_moves(df_clean)
        
        # 5. Suppression des bots (si activé)
        if self.config['remove_bots']:
            df_clean = self._remove_bots(df_clean)
        
        # 6. Suppression des variantes (si activé)
        if self.config['remove_variants']:
            df_clean = self._remove_variants(df_clean)
        
        # 7. Suppression des doublons
        # df_clean = self._remove_duplicates(df_clean)
        
        # 8. Standardisation des ouvertures
        if self.config['standardize_openings']:
            df_clean = self._standardize_openings(df_clean)
        
        # 9. Filtrage des ouvertures rares
        df_clean = self._filter_rare_openings(df_clean)
        
        # 10. Enrichissement des données
        df_clean = self._enrich_data(df_clean)
        
        # 11. Validation finale
        df_clean = self._final_validation(df_clean)
        
        self.cleaning_stats['final_games'] = len(df_clean)
        
        if verbose:
            self._print_cleaning_stats()
        
        return df_clean
    
    def _clean_basic_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoyage des colonnes de base"""
        logger.info("🔧 Nettoyage des colonnes de base...")
        
        # Supprimer les espaces dans les noms de colonnes
        df.columns = df.columns.str.strip()
        
        # Nettoyer les valeurs textuelles
        text_columns = ['opening', 'main_opening', 'variation', 'termination', 'eco']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(['nan', 'None', ''], np.nan)
        
        # Convertir les types numériques
        numeric_columns = ['white_elo', 'black_elo', 'moves_count', 'avg_elo']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _validate_elo(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validation des ELO"""
        logger.info("🎯 Validation des ELO...")
        
        initial_count = len(df)
        
        # Supprimer les ELO manquants ou invalides
        df = df.dropna(subset=['white_elo', 'black_elo'])
        
        # Filtrer les ELO dans la plage valide
        valid_elo_mask = (
            (df['white_elo'] >= self.config['min_elo']) &
            (df['white_elo'] <= self.config['max_elo']) &
            (df['black_elo'] >= self.config['min_elo']) &
            (df['black_elo'] <= self.config['max_elo'])
        )
        
        df = df[valid_elo_mask]
        
        removed = initial_count - len(df)
        self.cleaning_stats['invalid_elo'] = removed
        
        logger.info(f"   ❌ Supprimé {removed:,} parties avec ELO invalide")
        
        return df
    
    def _validate_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validation des résultats"""
        logger.info("🏆 Validation des résultats...")
        
        initial_count = len(df)
        
        # Nettoyer les résultats
        if 'Result' in df.columns:
            df['Result'] = df['Result'].str.strip()
        
        # Filtrer les résultats valides
        valid_results = self.config['valid_results']
        result_column = 'Result' if 'Result' in df.columns else 'result'
        
        if result_column in df.columns:
            df = df[df[result_column].isin(valid_results)]
        
        removed = initial_count - len(df)
        self.cleaning_stats['invalid_result'] = removed
        
        logger.info(f"   ❌ Supprimé {removed:,} parties avec résultat invalide")
        
        return df
    
    def _validate_moves(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validation du nombre de coups"""
        logger.info("♟️ Validation du nombre de coups...")
        
        initial_count = len(df)
        
        if 'moves_count' in df.columns:
            valid_moves_mask = (
                (df['moves_count'] >= self.config['min_moves']) &
                (df['moves_count'] <= self.config['max_moves'])
            )
            df = df[valid_moves_mask]
        
        removed = initial_count - len(df)
        self.cleaning_stats['invalid_moves'] = removed
        
        logger.info(f"   ❌ Supprimé {removed:,} parties avec nombre de coups invalide")
        
        return df
    
    def _remove_bots(self, df: pd.DataFrame) -> pd.DataFrame:
        """Suppression des parties de bots"""
        logger.info("🤖 Suppression des parties de bots...")
        
        initial_count = len(df)
        
        # Patterns pour identifier les bots (ajustez selon vos données)
        bot_patterns = [
            r'.*[Bb]ot.*',
            r'.*[Aa]I.*',
            r'.*[Cc]omputer.*',
            r'.*[Ee]ngine.*',
            r'.*[Ss]tockfish.*',
            r'.*[Kk]omodo.*',
            r'.*[Ll]eela.*'
        ]
        
        if 'White' in df.columns and 'Black' in df.columns:
            for pattern in bot_patterns:
                bot_mask = (
                    df['White'].str.match(pattern, na=False) |
                    df['Black'].str.match(pattern, na=False)
                )
                df = df[~bot_mask]
        
        removed = initial_count - len(df)
        self.cleaning_stats['bot_games'] = removed
        
        logger.info(f"   ❌ Supprimé {removed:,} parties de bots")
        
        return df
    
    def _remove_variants(self, df: pd.DataFrame) -> pd.DataFrame:
        """Suppression des variantes d'échecs"""
        logger.info("♜ Suppression des variantes...")
        
        initial_count = len(df)
        
        # Variantes à exclure (garder seulement les parties standard)
        variant_patterns = [
            r'.*[Cc]hess960.*',
            r'.*[Ff]ischer.*',
            r'.*[Rr]andom.*',
            r'.*[Hh]orde.*',
            r'.*[Aa]tomic.*',
            r'.*[Aa]ntichess.*',
            r'.*[Ss]uicide.*',
            r'.*[Kk]ing.*[Hh]ill.*',
            r'.*[Tt]hree.*[Cc]heck.*'
        ]
        
        if 'Event' in df.columns:
            for pattern in variant_patterns:
                variant_mask = df['Event'].str.match(pattern, na=False)
                df = df[~variant_mask]
        
        removed = initial_count - len(df)
        self.cleaning_stats['variant_games'] = removed
        
        logger.info(f"   ❌ Supprimé {removed:,} parties de variantes")
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Suppression des doublons"""
        logger.info("🔄 Suppression des doublons...")
        
        initial_count = len(df)
        
        # Colonnes pour identifier les doublons
        duplicate_columns = []
        possible_columns = ['White', 'Black', 'Date', 'moves_count', 'Result']
        
        for col in possible_columns:
            if col in df.columns:
                duplicate_columns.append(col)
        
        if duplicate_columns:
            df = df.drop_duplicates(subset=duplicate_columns, keep='first')
        
        removed = initial_count - len(df)
        self.cleaning_stats['duplicate_games'] = removed
        
        logger.info(f"   ❌ Supprimé {removed:,} doublons")
        
        return df
    
    def _standardize_openings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardisation des noms d'ouvertures"""
        logger.info("📚 Standardisation des ouvertures...")
        
        if 'opening' not in df.columns:
            return df
        
        # Dictionnaire de standardisation
        opening_mappings = {
            # Défense Sicilienne
            r'.*[Ss]icilian.*[Dd]efense.*': 'Sicilian Defense',
            r'.*[Ss]icilian.*[Dd]ragon.*': 'Sicilian Defense: Dragon Variation',
            r'.*[Ss]icilian.*[Nn]ajdorf.*': 'Sicilian Defense: Najdorf Variation',
            r'.*[Ss]icilian.*[Aa]ccelerated.*': 'Sicilian Defense: Accelerated Dragon',
            
            # Gambit de la Dame
            r'.*[Qq]ueen.*[Gg]ambit.*[Dd]eclined.*': "Queen's Gambit Declined",
            r'.*[Qq]ueen.*[Gg]ambit.*[Aa]ccepted.*': "Queen's Gambit Accepted",
            r'.*[Qq]ueen.*[Gg]ambit.*': "Queen's Gambit",
            
            # Défense Française
            r'.*[Ff]rench.*[Dd]efense.*': 'French Defense',
            r'.*[Ff]rench.*': 'French Defense',
            
            # Ouverture Anglaise
            r'.*[Ee]nglish.*[Oo]pening.*': 'English Opening',
            r'.*[Ee]nglish.*': 'English Opening',
            
            # Défense Caro-Kann
            r'.*[Cc]aro.*[Kk]ann.*': 'Caro-Kann Defense',
            
            # Partie Italienne
            r'.*[Ii]talian.*[Gg]ame.*': 'Italian Game',
            
            # Défense Slave
            r'.*[Ss]lav.*[Dd]efense.*': 'Slav Defense',
            
            # Ouverture Réti
            r'.*[Rr]eti.*[Oo]pening.*': 'Reti Opening',
            r'.*[Rr]éti.*': 'Reti Opening',
        }
        
        # Appliquer les mappings
        for pattern, standard_name in opening_mappings.items():
            mask = df['opening'].str.match(pattern, na=False)
            df.loc[mask, 'opening'] = standard_name
        
        
        
        return df
    
    def _filter_rare_openings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtrage des ouvertures rares"""
        logger.info("🔍 Filtrage des ouvertures rares...")
        
        initial_count = len(df)
        
        if 'main_opening' in df.columns:
            opening_counts = df['main_opening'].value_counts()
            popular_openings = opening_counts[
                opening_counts >= self.config['min_games_per_opening']
            ].index
            
            df = df[df['main_opening'].isin(popular_openings)]
        
        removed = initial_count - len(df)
        self.cleaning_stats['rare_openings'] = removed
        
        logger.info(f"   ❌ Supprimé {removed:,} parties d'ouvertures rares")
        logger.info(f"   ✅ Conservé {df['main_opening'].nunique()} ouvertures populaires")
        
        return df
    
    def _enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrichissement des données avec nouvelles colonnes"""
        logger.info("✨ Enrichissement des données...")
        
        # Écart d'ELO absolu
        if 'white_elo' in df.columns and 'black_elo' in df.columns:
            df['elo_diff_abs'] = abs(df['white_elo'] - df['black_elo'])
            df['stronger_player'] = np.where(
                df['white_elo'] > df['black_elo'], 'White', 'Black'
            )
        
        # Catégories d'ELO plus fines
        if 'avg_elo' in df.columns:
            df['elo_category'] = pd.cut(
                df['avg_elo'],
                bins=[0, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 3500],
                labels=['<1000', '1000-1200', '1200-1400', '1400-1600', '1600-1800',
                       '1800-2000', '2000-2200', '2200-2400', '2400-2600', '2600+']
            )
        
        # Catégories de durée de partie
        if 'moves_count' in df.columns:
            df['game_length_category'] = pd.cut(
                df['moves_count'],
                bins=[0, 15, 25, 40, 60, 80, 200],
                labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long', 'Epic']
            )
        
        # Indicateur de partie équilibrée
        if 'elo_diff_abs' in df.columns:
            df['balanced_game'] = df['elo_diff_abs'] <= 100
        
        # Performance relative au niveau
        if 'white_score' in df.columns and 'avg_elo' in df.columns:
            # Calculer la performance attendue basée sur l'ELO
            expected_performance = df.groupby('elo_category')['white_score'].transform('mean')
            df['performance_vs_expected'] = df['white_score'] - expected_performance
        
        return df
    
    def _final_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validation finale et vérifications"""
        logger.info("🔍 Validation finale...")
        
        # Vérifier qu'il reste suffisamment de données
        if len(df) < 1000:
            logger.warning("⚠️ Attention: Moins de 1000 parties après nettoyage!")
        
        # Vérifier la distribution des ELO
        if 'avg_elo' in df.columns:
            elo_mean = df['avg_elo'].mean()
            elo_std = df['avg_elo'].std()
            
            if elo_mean < 1200 or elo_mean > 2200:
                logger.warning(f"⚠️ ELO moyen inhabituel: {elo_mean:.0f}")
            
            if elo_std < 100 or elo_std > 500:
                logger.warning(f"⚠️ Dispersion d'ELO inhabituelle: {elo_std:.0f}")
        
        # Vérifier l'équilibre blanc/noir
        if 'white_score' in df.columns:
            white_winrate = df['white_score'].mean()
            
            if white_winrate < 0.45 or white_winrate > 0.65:
                logger.warning(f"⚠️ Winrate des blancs inhabituel: {white_winrate:.1%}")
        
        return df
    
    def _print_cleaning_stats(self):
        """Affiche les statistiques de nettoyage"""
        stats = self.cleaning_stats
        
        logger.info("\n📊 STATISTIQUES DE NETTOYAGE")
        logger.info("=" * 50)
        logger.info(f"📥 Parties initiales:     {stats['initial_games']:,}")
        logger.info(f"❌ ELO invalides:        {stats['invalid_elo']:,}")
        logger.info(f"❌ Résultats invalides:  {stats['invalid_result']:,}")
        logger.info(f"❌ Coups invalides:      {stats['invalid_moves']:,}")
        logger.info(f"❌ Parties de bots:      {stats['bot_games']:,}")
        logger.info(f"❌ Variantes:            {stats['variant_games']:,}")
        logger.info(f"❌ Doublons:             {stats['duplicate_games']:,}")
        logger.info(f"❌ Ouvertures rares:     {stats['rare_openings']:,}")
        logger.info(f"✅ Parties finales:      {stats['final_games']:,}")
        
        retention_rate = (stats['final_games'] / stats['initial_games']) * 100
        logger.info(f"📈 Taux de rétention:    {retention_rate:.1f}%")
        
        total_removed = stats['initial_games'] - stats['final_games']
        logger.info(f"🗑️ Total supprimé:       {total_removed:,}")
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Génère un rapport de qualité des données
        """
        report = {
            'total_games': len(df),
            'unique_players': 0,
            'unique_openings': 0,
            'elo_distribution': {},
            'time_control_distribution': {},
            'result_distribution': {},
            'data_completeness': {},
            'potential_issues': []
        }
        
        # Joueurs uniques
        if 'White' in df.columns and 'Black' in df.columns:
            all_players = set(df['White'].unique()) | set(df['Black'].unique())
            report['unique_players'] = len(all_players)
        
        # Ouvertures uniques
        if 'main_opening' in df.columns:
            report['unique_openings'] = df['main_opening'].nunique()
        
        # Distribution des ELO
        if 'avg_elo' in df.columns:
            report['elo_distribution'] = {
                'mean': df['avg_elo'].mean(),
                'median': df['avg_elo'].median(),
                'std': df['avg_elo'].std(),
                'min': df['avg_elo'].min(),
                'max': df['avg_elo'].max()
            }
        
        # Distribution des contrôles de temps
        if 'time_control' in df.columns:
            report['time_control_distribution'] = df['time_control'].value_counts().to_dict()
        
        # Distribution des résultats
        if 'white_score' in df.columns:
            report['result_distribution'] = {
                'white_wins': (df['white_score'] == 1.0).sum(),
                'black_wins': (df['white_score'] == 0.0).sum(),
                'draws': (df['white_score'] == 0.5).sum()
            }
        
        # Complétude des données
        report['data_completeness'] = {
            col: (1 - df[col].isna().mean()) * 100
            for col in df.columns
        }
        
        # Issues potentielles
        issues = []
        
        if report['total_games'] < 10000:
            issues.append("Dataset relativement petit (< 10k parties)")
        
        if 'avg_elo' in df.columns:
            if df['avg_elo'].std() > 400:
                issues.append("Grande dispersion d'ELO (> 400 points)")
        
        missing_data = [col for col, completeness in report['data_completeness'].items() 
                       if completeness < 90]
        if missing_data:
            issues.append(f"Données manquantes importantes: {missing_data}")
        
        report['potential_issues'] = issues
        
        return report


# Fonctions utilitaires
def quick_clean(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Nettoyage rapide avec configuration par défaut
    """
    cleaner = ChessDataCleaner(config)
    return cleaner.clean_dataset(df)


def validate_pgn_data(df: pd.DataFrame) -> bool:
    """
    Validation rapide d'un DataFrame PGN
    """
    required_columns = ['white_elo', 'black_elo', 'opening']
    
    if not all(col in df.columns for col in required_columns):
        return False
    
    if len(df) == 0:
        return False
    
    if df['white_elo'].isna().all() or df['black_elo'].isna().all():
        return False
    
    return True


def create_cleaning_config(
    min_elo: int = 800,
    min_games_per_opening: int = 20,
    remove_bots: bool = True,
    **kwargs
) -> Dict:
    """
    Crée une configuration de nettoyage personnalisée
    """
    config = {
        'min_elo': min_elo,
        'min_games_per_opening': min_games_per_opening,
        'remove_bots': remove_bots
    }
    config.update(kwargs)
    return config
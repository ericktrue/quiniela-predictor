from dataclasses import dataclass

from data.providers.csv_historical_source import CsvHistoricalSource
from models.elo import EloCalculator
from models.probability_calibrator import ProbabilityCalibrator
from models.poisson_model import PoissonGoalModel


@dataclass
class HybridPrediction:
    """Resultado final combinado, listo para mostrar en la UI de Streamlit."""
    home_team: str
    away_team: str

    # Probabilidades finales (combinadas)
    home_win_prob: float
    draw_prob: float
    away_win_prob: float

    # Detalle de cada modelo por separado, para transparencia
    elo_home_win: float
    elo_draw: float
    elo_away_win: float
    poisson_home_win: float
    poisson_draw: float
    poisson_away_win: float

    # Goles esperados y marcadores mas probables (solo Poisson sabe de esto)
    expected_goals_home: float
    expected_goals_away: float
    most_likely_scores: list[tuple[tuple[int, int], float]]


class HybridPredictor:
    """
    Combina EloCalculator + ProbabilityCalibrator (fuerza historica de largo
    plazo) con PoissonGoalModel (forma reciente y marcador exacto) en una
    sola prediccion final.
    """

    def __init__(self, csv_path: str, elo_weight: float = 0.6, recent_n: int = 12):
        if not 0.0 <= elo_weight <= 1.0:
            raise ValueError("elo_weight debe estar entre 0.0 y 1.0")

        self.elo_weight = elo_weight
        self.poisson_weight = 1.0 - elo_weight

        source = CsvHistoricalSource(csv_path)
        matches = source.load_matches()

        self._elo = EloCalculator()
        self._elo.process_all(matches)

        self._calibrator = ProbabilityCalibrator()
        self._calibrator.fit(matches)

        self._poisson = PoissonGoalModel(matches, recent_n=recent_n)

    def predict(
        self, home_team: str, away_team: str, neutral: bool = False
    ) -> HybridPrediction:
        # --- Senal 1: Elo (estructural, largo plazo) ---
        rating_home = self._elo.get_rating(home_team)
        rating_away = self._elo.get_rating(away_team)
        expected_home = self._elo.expected_score(rating_home, rating_away)
        elo_probs = self._calibrator.predict_probabilities(expected_home, neutral=neutral)

        # --- Senal 2: Poisson (forma reciente) ---
        poisson_probs = self._poisson.outcome_probabilities(home_team, away_team, neutral=neutral)
        lambda_home, lambda_away = self._poisson.expected_goals(home_team, away_team, neutral=neutral)
        top_scores = self._poisson.most_likely_scores(home_team, away_team, top_n=5, neutral=neutral)

        # --- Combinacion ponderada ---
        home_win = (
            self.elo_weight * elo_probs["home_win"]
            + self.poisson_weight * poisson_probs["home_win"]
        )
        draw = (
            self.elo_weight * elo_probs["draw"]
            + self.poisson_weight * poisson_probs["draw"]
        )
        away_win = (
            self.elo_weight * elo_probs["away_win"]
            + self.poisson_weight * poisson_probs["away_win"]
        )

        return HybridPrediction(
            home_team=home_team,
            away_team=away_team,
            home_win_prob=home_win,
            draw_prob=draw,
            away_win_prob=away_win,
            elo_home_win=elo_probs["home_win"],
            elo_draw=elo_probs["draw"],
            elo_away_win=elo_probs["away_win"],
            poisson_home_win=poisson_probs["home_win"],
            poisson_draw=poisson_probs["draw"],
            poisson_away_win=poisson_probs["away_win"],
            expected_goals_home=lambda_home,
            expected_goals_away=lambda_away,
            most_likely_scores=top_scores,
        )

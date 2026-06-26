from dataclasses import dataclass

from data.providers.csv_historical_source import CsvHistoricalSource
from models.elo import EloCalculator
from models.probability_calibrator import ProbabilityCalibrator


@dataclass
class MatchPrediction:
    """Resultado limpio de una predicción, listo para mostrar en la UI."""
    home_team: str
    away_team: str
    home_rating: float
    away_rating: float
    home_win_prob: float
    draw_prob: float
    away_win_prob: float


class MatchPredictor:
    """
    Punto de entrada único para predecir partidos. Esconde los detalles
    de Elo y calibración de probabilidades -- el resto de la app (ej. Streamlit)
    solo necesita llamar a predict(equipo_local, equipo_visita).
    """

    def __init__(self, csv_path: str, bucket_size: float = 0.05):
        source = CsvHistoricalSource(csv_path)
        matches = source.load_matches()

        # El EloCalculator que usaremos para consultas se entrena con TODO el historial
        self._elo = EloCalculator()
        self._elo.process_all(matches)

        # El calibrador necesita su propio recorrido (entrena su propio EloCalculator interno)
        self._calibrator = ProbabilityCalibrator(bucket_size=bucket_size)
        self._calibrator.fit(matches)

    def predict(self, home_team: str, away_team: str) -> MatchPrediction:
        rating_home = self._elo.get_rating(home_team)
        rating_away = self._elo.get_rating(away_team)

        expected_home = self._elo.expected_score(rating_home, rating_away)
        probs = self._calibrator.predict_probabilities(expected_home)

        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            home_rating=rating_home,
            away_rating=rating_away,
            home_win_prob=probs["home_win"],
            draw_prob=probs["draw"],
            away_win_prob=probs["away_win"],
        )

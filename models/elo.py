from dataclasses import dataclass
from data.providers.base import Match


class EloCalculator:
    """
    Procesa una lista de partidos en orden cronológico y mantiene
    el rating Elo actualizado de cada selección.
    """

    def __init__(self, initial_rating: float = 1500.0):
        self.initial_rating = initial_rating
        self._ratings: dict[str, float] = {}

    def get_rating(self, team: str) -> float:
        """Devuelve el rating actual del equipo (o el inicial si nunca jugó)."""
        return self._ratings.get(team, self.initial_rating)

    def _k_factor(self, tournament: str) -> int:
        """Determina cuánto pesa el partido según el tipo de torneo."""
        tournament_lower = tournament.lower()
        if "world cup" in tournament_lower and "qualif" not in tournament_lower:
            return 60  # Fase final de Mundial
        if "qualif" in tournament_lower or "cup" in tournament_lower:
            return 40  # Eliminatorias o copas continentales
        return 20  # Amistosos y el resto

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Devuelve el valor esperado de puntos (0-1) para el equipo A frente al B,
        segun la diferencia de rating. Es publico porque otras clases (como
        ProbabilityCalibrator) lo necesitan para calibrar probabilidades reales.
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def process_match(self, match: Match) -> None:
        """Actualiza los ratings de ambos equipos tras un partido."""
        rating_home = self.get_rating(match.home_team)
        rating_away = self.get_rating(match.away_team)

        expected_home = self.expected_score(rating_home, rating_away)
        expected_away = 1 - expected_home

        if match.home_goals > match.away_goals:
            actual_home, actual_away = 1.0, 0.0
        elif match.home_goals < match.away_goals:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5

        k = self._k_factor(match.tournament)

        self._ratings[match.home_team] = rating_home + k * (actual_home - expected_home)
        self._ratings[match.away_team] = rating_away + k * (actual_away - expected_away)

    def process_all(self, matches: list[Match]) -> None:
        """Procesa una lista completa de partidos en orden cronológico."""
        for match in sorted(matches, key=lambda m: m.match_date):
            self.process_match(match)

import math
from data.providers.base import Match


def poisson_pmf(lam: float, k: int) -> float:
    """
    Probabilidad de que ocurran exactamente k eventos, dado un promedio
    esperado de lam eventos. Esta es la formula de la distribucion de
    Poisson, implementada directamente (sin librerias externas).
    """
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


class PoissonGoalModel:
    """
    Estima goles esperados (lambda) para cada equipo en un partido,
    usando fuerza de ataque/defensa derivada de partidos recientes,
    y genera la matriz completa de probabilidades de marcador.
    """

    def __init__(self, matches: list[Match], recent_n: int = 12):
        self.matches = sorted(matches, key=lambda m: m.match_date)
        self.recent_n = recent_n

        # --- Promedios globales, calculados una sola vez ---
        total_home_goals = sum(m.home_goals for m in self.matches)
        total_away_goals = sum(m.away_goals for m in self.matches)
        n = len(self.matches)

        self.avg_home_goals = total_home_goals / n
        self.avg_away_goals = total_away_goals / n
        # Promedio global de goles por equipo por partido (ambos lados combinados)
        self.league_avg_goals = (total_home_goals + total_away_goals) / (n * 2)

        # Factor de ventaja de casa, calculado SEPARADO segun si la cancha es neutral.
        # En cancha neutral no hay ventaja real, por lo que este factor debe ser mucho mas bajo.
        self.home_advantage_factor = self._compute_home_advantage(neutral=False)
        self.neutral_advantage_factor = self._compute_home_advantage(neutral=True)

    def _compute_home_advantage(self, neutral: bool) -> float:
        """Calcula el factor de ventaja de casa solo con partidos del tipo indicado."""
        subset = [m for m in self.matches if m.neutral == neutral]
        if not subset:
            return 1.0
        avg_home = sum(m.home_goals for m in subset) / len(subset)
        avg_away = sum(m.away_goals for m in subset) / len(subset)
        if avg_away == 0:
            return 1.0
        return avg_home / avg_away

    def _recent_matches_for_team(self, team: str) -> list[Match]:
        """Devuelve los ultimos recent_n partidos (de cualquier rival) de un equipo."""
        team_matches = [
            m for m in self.matches if m.home_team == team or m.away_team == team
        ]
        # self.matches ya esta ordenado por fecha ascendente, tomamos los ultimos N
        return team_matches[-self.recent_n:]

    def attack_defense_strength(self, team: str) -> tuple[float, float]:
        """
        Devuelve (fuerza_ataque, debilidad_defensa) del equipo, basado en
        sus partidos recientes. Un valor de 1.0 significa "promedio de la liga".
        """
        recent = self._recent_matches_for_team(team)

        if not recent:
            # Sin historial reciente: asumimos que es un equipo promedio
            return 1.0, 1.0

        goals_scored = []
        goals_conceded = []
        for m in recent:
            if m.home_team == team:
                goals_scored.append(m.home_goals)
                goals_conceded.append(m.away_goals)
            else:
                goals_scored.append(m.away_goals)
                goals_conceded.append(m.home_goals)

        avg_scored = sum(goals_scored) / len(goals_scored)
        avg_conceded = sum(goals_conceded) / len(goals_conceded)

        attack_strength = avg_scored / self.league_avg_goals
        defense_weakness = avg_conceded / self.league_avg_goals

        return attack_strength, defense_weakness

    def expected_goals(
        self, home_team: str, away_team: str, neutral: bool = False
    ) -> tuple[float, float]:
        """Devuelve (lambda_local, lambda_visita) para el partido."""
        attack_home, defense_home = self.attack_defense_strength(home_team)
        attack_away, defense_away = self.attack_defense_strength(away_team)

        advantage = self.neutral_advantage_factor if neutral else self.home_advantage_factor

        lambda_home = attack_home * defense_away * self.league_avg_goals * advantage
        lambda_away = attack_away * defense_home * self.league_avg_goals

        return lambda_home, lambda_away

    def score_matrix(
        self, home_team: str, away_team: str, max_goals: int = 6, neutral: bool = False
    ) -> dict[tuple[int, int], float]:
        """
        Devuelve un diccionario {(goles_local, goles_visita): probabilidad}
        para todos los marcadores posibles hasta max_goals por lado.
        """
        lambda_home, lambda_away = self.expected_goals(home_team, away_team, neutral=neutral)

        matrix: dict[tuple[int, int], float] = {}
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p_home = poisson_pmf(lambda_home, i)
                p_away = poisson_pmf(lambda_away, j)
                matrix[(i, j)] = p_home * p_away  # Asumimos independencia

        return matrix

    def most_likely_scores(
        self, home_team: str, away_team: str, top_n: int = 5, neutral: bool = False
    ) -> list[tuple[tuple[int, int], float]]:
        """Devuelve los top_n marcadores mas probables, ordenados de mayor a menor."""
        matrix = self.score_matrix(home_team, away_team, neutral=neutral)
        return sorted(matrix.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

    def outcome_probabilities(
        self, home_team: str, away_team: str, neutral: bool = False
    ) -> dict[str, float]:
        """
        Suma la matriz de marcadores para obtener probabilidad de
        victoria local / empate / victoria visita.
        """
        matrix = self.score_matrix(home_team, away_team, neutral=neutral)
        home_win = sum(p for (i, j), p in matrix.items() if i > j)
        draw = sum(p for (i, j), p in matrix.items() if i == j)
        away_win = sum(p for (i, j), p in matrix.items() if i < j)
        return {"home_win": home_win, "draw": draw, "away_win": away_win}

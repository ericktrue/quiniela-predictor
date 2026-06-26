from collections import defaultdict
from data.providers.base import Match
from models.elo import EloCalculator


class ProbabilityCalibrator:
    """
    Convierte el 'expected_score' de Elo (un valor 0-1 de puntos esperados)
    en probabilidades reales de victoria/empate/derrota, calibradas con
    el historial real de partidos.

    Mantiene DOS tablas de calibracion separadas: una para partidos con
    local real (donde existe ventaja de jugar en casa) y otra para partidos
    en cancha neutral (donde esa ventaja no aplica, o aplica mucho menos).
    """

    def __init__(self, bucket_size: float = 0.05):
        self.bucket_size = bucket_size
        self._buckets: dict[int, dict[str, int]] = defaultdict(
            lambda: {"win": 0, "draw": 0, "loss": 0}
        )
        self._neutral_buckets: dict[int, dict[str, int]] = defaultdict(
            lambda: {"win": 0, "draw": 0, "loss": 0}
        )

    def _bucket_index(self, expected_score: float) -> int:
        """Determina a que caja pertenece un expected_score dado."""
        index = int(expected_score / self.bucket_size)
        max_index = int(1.0 / self.bucket_size) - 1
        return min(index, max_index)

    def fit(self, matches: list[Match], initial_rating: float = 1500.0) -> None:
        """
        Recorre el historial completo en orden cronologico. Para cada partido,
        registra en que caja caia el expected_score del local ANTES de jugarse,
        separando segun si el partido fue en cancha neutral o no.
        """
        calc = EloCalculator(initial_rating=initial_rating)

        for match in sorted(matches, key=lambda m: m.match_date):
            rating_home = calc.get_rating(match.home_team)
            rating_away = calc.get_rating(match.away_team)
            expected_home = calc.expected_score(rating_home, rating_away)

            bucket = self._bucket_index(expected_home)
            target = self._neutral_buckets if match.neutral else self._buckets

            if match.home_goals > match.away_goals:
                target[bucket]["win"] += 1
            elif match.home_goals < match.away_goals:
                target[bucket]["loss"] += 1
            else:
                target[bucket]["draw"] += 1

            calc.process_match(match)

    def predict_probabilities(
        self, expected_score_home: float, neutral: bool = False
    ) -> dict[str, float]:
        """
        Dado un expected_score (0-1) para el equipo local, devuelve las
        probabilidades reales calibradas: {'home_win', 'draw', 'away_win'}.
        Usa la tabla de cancha neutral si neutral=True.
        """
        bucket = self._bucket_index(expected_score_home)
        buckets = self._neutral_buckets if neutral else self._buckets
        counts = buckets[bucket]
        total = counts["win"] + counts["draw"] + counts["loss"]

        if total == 0:
            return {"home_win": expected_score_home, "draw": 0.0, "away_win": 1 - expected_score_home}

        return {
            "home_win": counts["win"] / total,
            "draw": counts["draw"] / total,
            "away_win": counts["loss"] / total,
        }

    def calibration_table(self, neutral: bool = False) -> list[tuple[str, int, dict[str, float]]]:
        """Devuelve la tabla completa de calibracion (neutral o no), util para depurar."""
        buckets = self._neutral_buckets if neutral else self._buckets
        rows = []
        for bucket_idx in sorted(buckets.keys()):
            counts = buckets[bucket_idx]
            total = sum(counts.values())
            rango = f"{bucket_idx * self.bucket_size:.0%}-{(bucket_idx + 1) * self.bucket_size:.0%}"
            probs = {k: v / total for k, v in counts.items()} if total > 0 else {}
            rows.append((rango, total, probs))
        return rows

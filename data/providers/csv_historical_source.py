import csv
from datetime import datetime
from pathlib import Path

from data.providers.base import Match


class CsvHistoricalSource:
    """
    Adapter que lee el historial de partidos internacionales desde
    el CSV de martj42/international_results y lo convierte en una
    lista de objetos Match, que es el formato que entiende EloCalculator.

    No implementamos la interfaz DataProvider porque este adapter
    no necesita exponer get_team_rating() -- ese cálculo ahora
    vive en EloCalculator, no en la fuente de datos.
    """

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)

    def load_matches(self) -> list[Match]:
        """
        Lee el CSV completo y devuelve una lista de Match.
        Descarta filas sin resultado real (partidos futuros, marcados NA).
        """
        matches: list[Match] = []

        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Saltar partidos que aún no se han jugado (resultado NA)
                if row["home_score"] == "NA" or row["away_score"] == "NA":
                    continue

                matches.append(
                    Match(
                        home_team=row["home_team"],
                        away_team=row["away_team"],
                        home_goals=int(row["home_score"]),
                        away_goals=int(row["away_score"]),
                        match_date=datetime.strptime(row["date"], "%Y-%m-%d").date(),
                        tournament=row["tournament"],
                        neutral=row["neutral"] == "TRUE",
                    )
                )

        return matches

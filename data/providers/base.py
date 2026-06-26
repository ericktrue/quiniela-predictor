from dataclasses import dataclass
from datetime import date


@dataclass
class Match:
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    match_date: date
    tournament: str = "Friendly"
    neutral: bool = False

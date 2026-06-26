from datetime import date, timedelta

import streamlit as st

from data.providers.csv_historical_source import CsvHistoricalSource
from models.hybrid_predictor import HybridPredictor

CSV_PATH = "international_results/results.csv"


@st.cache_resource
def load_predictor(elo_weight: float, recent_n: int) -> HybridPredictor:
    """
    Carga y entrena el predictor una sola vez por configuracion.
    st.cache_resource evita recalcular Elo + Poisson en cada interaccion
    del usuario (cargar 49 mil partidos toma ~1 segundo, pero sin cache
    se repetiria en cada click).
    """
    return HybridPredictor(CSV_PATH, elo_weight=elo_weight, recent_n=recent_n)


@st.cache_data
def get_active_teams() -> list[str]:
    """Devuelve los equipos que han jugado en los ultimos 2 anos."""
    source = CsvHistoricalSource(CSV_PATH)
    matches = source.load_matches()

    corte = date.today() - timedelta(days=730)
    equipos = set()
    for m in matches:
        if m.match_date >= corte:
            equipos.add(m.home_team)
            equipos.add(m.away_team)

    return sorted(equipos)


st.set_page_config(page_title="Predictor de Quiniela", page_icon="⚽", layout="centered")

st.title("⚽ Predictor de Partidos - Quiniela")
st.caption("Modelo hibrido Elo + Poisson, entrenado con historial real de partidos internacionales")

with st.sidebar:
    st.header("Configuracion del modelo")
    elo_weight = st.slider(
        "Peso de Elo (historico) vs. Poisson (forma reciente)",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="1.0 = solo importa la historia de largo plazo. 0.0 = solo importa la forma reciente.",
    )
    recent_n = st.slider(
        "Partidos recientes a considerar (forma)",
        min_value=5,
        max_value=25,
        value=12,
        step=1,
    )

predictor = load_predictor(elo_weight, recent_n)
equipos = get_active_teams()

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Equipo local", equipos, index=equipos.index("Mexico") if "Mexico" in equipos else 0)
with col2:
    away_team_options = [e for e in equipos if e != home_team]
    away_team = st.selectbox("Equipo visitante", away_team_options, index=0)

neutral = st.checkbox(
    "Cancha neutral (ej. fase de grupos de Mundial, partido en sede sin local real)",
    value=False,
)

if st.button("Predecir partido", type="primary", use_container_width=True):
    with st.spinner("Calculando..."):
        pred = predictor.predict(home_team, away_team, neutral=neutral)

    st.subheader(f"{pred.home_team} vs {pred.away_team}")

    # --- Probabilidades finales ---
    st.markdown("### Probabilidad de resultado")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Gana {pred.home_team}", f"{pred.home_win_prob*100:.1f}%")
    c2.metric("Empate", f"{pred.draw_prob*100:.1f}%")
    c3.metric(f"Gana {pred.away_team}", f"{pred.away_win_prob*100:.1f}%")

    st.progress(pred.home_win_prob, text=f"Local: {pred.home_win_prob*100:.1f}%")
    st.progress(pred.draw_prob, text=f"Empate: {pred.draw_prob*100:.1f}%")
    st.progress(pred.away_win_prob, text=f"Visita: {pred.away_win_prob*100:.1f}%")

    # --- Detalle por modelo (transparencia) ---
    with st.expander("Ver detalle: que dice cada modelo por separado"):
        st.markdown("**Elo (fuerza historica de largo plazo)**")
        st.write(
            f"Local: {pred.elo_home_win*100:.1f}% | "
            f"Empate: {pred.elo_draw*100:.1f}% | "
            f"Visita: {pred.elo_away_win*100:.1f}%"
        )
        st.markdown("**Poisson (forma reciente)**")
        st.write(
            f"Local: {pred.poisson_home_win*100:.1f}% | "
            f"Empate: {pred.poisson_draw*100:.1f}% | "
            f"Visita: {pred.poisson_away_win*100:.1f}%"
        )

    # --- Marcador esperado ---
    st.markdown("### Marcador esperado")
    st.write(f"Goles esperados: **{pred.expected_goals_home:.2f} - {pred.expected_goals_away:.2f}**")

    st.markdown("**Marcadores mas probables:**")
    for (gh, ga), prob in pred.most_likely_scores:
        st.write(f"- {gh} - {ga}  ({prob*100:.1f}%)")
else:
    st.info("Selecciona dos equipos y presiona 'Predecir partido'.")
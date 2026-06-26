# ⚽ Predictor de Quiniela — Selecciones Nacionales

Herramienta en Python para predecir resultados de partidos de selecciones
nacionales (Mundial, Eliminatorias, Copas) usando un modelo híbrido
**Elo + Poisson**, con interfaz web en Streamlit.

El objetivo es apoyar la toma de decisiones en quinielas: para cada partido,
la herramienta entrega:
- Probabilidad de victoria local / empate / victoria visitante.
- Goles esperados de cada equipo.
- Los marcadores exactos más probables.

---

## Cómo funciona el modelo

El sistema combina dos señales distintas, a propósito:

| Componente | Qué mide | Ventana de tiempo |
|---|---|---|
| **Elo** | Fuerza relativa histórica de cada selección | Todo el historial (desde 1872) |
| **Poisson** | Forma ofensiva/defensiva reciente, y estimación de marcador | Últimos *N* partidos (configurable, default 12) |

Ambas señales se combinan con un peso ajustable (`elo_weight`, default 0.6)
para dar la probabilidad final de resultado. El marcador exacto solo lo
puede estimar el componente de Poisson, ya que Elo no modela goles.

### Elo

Se calcula desde cero a partir del historial completo de partidos
(`results.csv`), usando la fórmula estándar de Elo con un factor `K`
que depende del tipo de torneo:

- Amistoso: `K = 20`
- Eliminatoria / Copa: `K = 40`
- Fase final de Mundial: `K = 60`

### Calibración de probabilidades (1X2)

El "expected score" de Elo (0-1) no es directamente una probabilidad de
victoria/empate/derrota — el empate depende de qué tan pareja sea la
diferencia de fuerza. En vez de asumir una fórmula, se **mide
empíricamente** sobre los ~49 mil partidos históricos: se agrupan los
partidos según su expected_score y se calcula qué tan seguido ocurrió
cada resultado real dentro de cada grupo.

### Poisson (marcador)

Los goles esperados (`λ`) de cada equipo se calculan con:

```
λ_local  = fuerza_ataque(local) × debilidad_defensa(visita) × promedio_liga × factor_casa
λ_visita = fuerza_ataque(visita) × debilidad_defensa(local) × promedio_liga
```

Donde `fuerza_ataque` y `debilidad_defensa` se calculan de los últimos
`N` partidos de cada equipo (cualquier rival), normalizados contra el
promedio global de goles.

### Cancha neutral

Cerca del 27% de los partidos del dataset se jugaron en sede neutral
(sin ventaja real de "casa" para ninguno de los dos equipos — frecuente
en fases de grupos de Mundial). El modelo calcula y usa un **factor de
ventaja de casa separado** para partidos neutrales vs. no neutrales, y
mantiene **tablas de calibración independientes** para cada caso. Esto
se controla con el parámetro `neutral=True/False` al predecir.

---

## Fuente de datos

[`martj42/international_results`](https://github.com/martj42/international_results) —
dataset abierto (CC0) con ~49,000 partidos internacionales desde 1872
hasta la fecha, incluyendo resultado, torneo, sede y si fue cancha neutral.
Se actualiza con frecuencia, incluyendo partidos del Mundial 2026 en curso.

Se mantiene como un `git clone` separado **dentro** de la carpeta del
proyecto (subcarpeta `international_results/`), para poder actualizarlo
fácilmente con `git pull` en vez de descargar el CSV a mano cada vez:

```bash
# Primera vez
git clone https://github.com/martj42/international_results.git

# Para actualizar con los partidos mas recientes
cd international_results
git pull
cd ..
```

> **Nota sobre Git anidado:** como `international_results/` es su propio
> repositorio Git, dentro de este proyecto (que también es un repo), se
> incluye en `.gitignore` para que tu propio repo no intente versionarlo
> como submódulo ni se confunda con su `.git` interno. Es una dependencia
> de datos externa, no código propio.

---

## Estructura del proyecto

```
quiniela-predictor/
│
├── app.py                              # Interfaz Streamlit (punto de entrada)
├── international_results/              # Clon de martj42/international_results (git pull para actualizar)
│   └── results.csv                     # Historial de partidos
├── requirements.txt
├── .gitignore                          # Excluye international_results/, venv/, etc.
│
├── data/
│   └── providers/
│       ├── base.py                     # Match (dataclass): home_team, away_team,
│       │                                #   goles, fecha, torneo, neutral
│       └── csv_historical_source.py     # Lee results.csv -> list[Match]
│
├── models/
│   ├── elo.py                          # EloCalculator: calcula rating Elo
│   ├── probability_calibrator.py        # Convierte expected_score -> prob. 1X2 real
│   ├── poisson_model.py                # PoissonGoalModel: goles esperados y marcador
│   ├── match_predictor.py              # Predictor solo-Elo (referencia/comparación)
│   └── hybrid_predictor.py             # Predictor final: combina Elo + Poisson
│
└── test_elo.py                         # Script de pruebas manuales del EloCalculator
```

---

## Instalación

```bash
git clone <este-repo>
cd quiniela-predictor

python3 -m venv venv
source venv/bin/activate          # En Windows: venv\Scripts\Activate.ps1

pip install -r requirements.txt

# Clonar el dataset de partidos (separado, se actualiza con git pull)
git clone https://github.com/martj42/international_results.git
```

## Uso

### Interfaz web (recomendado)

```bash
streamlit run app.py
```

Abre `http://localhost:8501`. Desde la barra lateral puedes ajustar:
- **Peso de Elo vs. Poisson**: qué tanto pesa la historia de largo plazo vs. la forma reciente.
- **Partidos recientes a considerar**: tamaño de la ventana de forma (default 12).

En la pantalla principal eliges los dos equipos, marcas si la cancha es
neutral, y obtienes la predicción completa.

### Uso programático

```python
from models.hybrid_predictor import HybridPredictor

predictor = HybridPredictor("international_results/results.csv", elo_weight=0.6, recent_n=12)

pred = predictor.predict("El Salvador", "Mexico", neutral=False)

print(f"Local: {pred.home_win_prob:.1%}")
print(f"Empate: {pred.draw_prob:.1%}")
print(f"Visita: {pred.away_win_prob:.1%}")
print(f"Marcador mas probable: {pred.most_likely_scores[0]}")
```

---

## Limitaciones conocidas

- **Ventana de forma pequeña (`recent_n=12` por default)**: una racha
  de goleadas (a favor o en contra) puede distorsionar bastante la
  estimación de Poisson para un equipo específico. Esto es intencional
  (queremos capturar forma *actual*), pero conviene revisar los partidos
  recientes de un equipo si su predicción se ve "extraña".
- **Independencia de goles asumida en Poisson**: el modelo asume que el
  marcador del local y del visitante son estadísticamente independientes.
  En la realidad hay correlación leve (ej. partidos abiertos tienden a
  tener más goles de ambos lados) — modelos más avanzados como
  Dixon-Coles corrigen esto, pero no se implementó aquí por simplicidad.
- **Equipos sin historial reciente**: si una selección no tiene partidos
  en la ventana de `recent_n`, el modelo le asigna fuerza de ataque/defensa
  "promedio" (1.0), lo cual puede ser poco preciso para selecciones que
  juegan con muy poca frecuencia.

## Posibles próximos pasos

- Incorporar una fuente de datos en vivo (API-Football o football-data.org)
  para complementar `results.csv` con partidos más recientes que aún no
  estén en el dataset de GitHub.
- Extender el modelo a clubes (Champions League, ligas europeas), aprovechando
  que la arquitectura ya separa fuente de datos / modelo / interfaz.
- Aplicar una corrección tipo Dixon-Coles para la correlación de marcador.

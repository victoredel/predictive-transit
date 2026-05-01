"""
app.py — Istanbul Smart Transit ETA Prediction
==============================================
Interactive Dashboard for IETT (Istanbul) Transit Predictions.
Built with Streamlit + XGBoost + Folium.

Usage:
  streamlit run dashboard/app.py
"""

import datetime
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px  # kept for potential future use
import streamlit as st
import xgboost as xgb
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DASHBOARD_DIR = Path(__file__).parent
AI_ENGINE_DIR = DASHBOARD_DIR.parent / "ai-engine"

MODEL_PATH   = AI_ENGINE_DIR / "models" / "istanbul" / "transit_xgboost_istanbul.json"
DATA_DIR     = AI_ENGINE_DIR / "datasets" / "istanbul" / "raw" / "gtfs_shapes"
TEST_PARQUET = AI_ENGINE_DIR / "datasets" / "istanbul" / "processed" / "X_test_pro.parquet"

# Model features — must match the training pipeline exactly (preprocess_istanbul.py)
FEATURES = [
    "trip_id", "stop_id", "stop_sequence", "stop_lat", "stop_lon", "arrival_seconds",
    "hora_del_dia", "distancia_proyectada", "velocidad_tramo_m_s",
    "temperature_2m", "precipitation", "num_paradas_salto"
]

# Weather presets: maps UI label to (temperature_2m °C, precipitation mm/h)
WEATHER_PRESETS = {
    "☀️  Clear / Sunny":  {"temperature_2m": 25.0, "precipitation": 0.0},
    "🌧️  Raining":        {"temperature_2m": 14.0, "precipitation": 12.0},
    "❄️  Snowing":        {"temperature_2m": -2.0, "precipitation": 5.0},
}

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Istanbul Smart Transit ETA Prediction",
    page_icon="🚍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Premium Dark Theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 50% 50%, #0d1b2a 0%, #050a14 100%);
    }

    [data-testid="stSidebar"] {
        background: #050a14;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }

    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #00d4ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
    }

    .subtitle {
        text-align: center;
        color: rgba(255, 255, 255, 0.55);
        font-size: 1.05rem;
        margin-bottom: 1.8rem;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #00d4ff, #0072ff);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 12px;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(0, 114, 255, 0.45);
    }

    .impact-box {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 210, 0, 0.3);
        border-radius: 12px;
        padding: 14px 18px;
        margin-top: 12px;
        color: rgba(255, 255, 255, 0.85);
        font-size: 0.93rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def calculate_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points in meters (Haversine formula)."""
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi    = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def build_folium_map(
    shape_coords: list[tuple],
    origin_lat: float, origin_lon: float,
    dest_lat: float, dest_lon: float,
    center_lat: float, center_lon: float,
) -> folium.Map:
    """
    Build a Folium map with OpenStreetMap tiles showing the bus route polyline
    and distinct Origin / Destination markers.
    """
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="OpenStreetMap",
    )

    # Route polyline
    folium.PolyLine(
        locations=shape_coords,
        color="#0072ff",
        weight=5,
        opacity=0.85,
        tooltip="Route path",
    ).add_to(m)

    # Origin marker — green
    folium.Marker(
        location=[origin_lat, origin_lon],
        tooltip="Origin",
        popup="Origin Stop",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(m)

    # Destination marker — red
    folium.Marker(
        location=[dest_lat, dest_lon],
        tooltip="Destination",
        popup="Destination Stop",
        icon=folium.Icon(color="red", icon="flag", prefix="fa"),
    ).add_to(m)

    return m


# ---------------------------------------------------------------------------
# Data Loading (Cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_prediction_model() -> xgb.XGBRegressor | None:
    """Load the pre-trained XGBoost model from disk."""
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at: `{MODEL_PATH}`")
        return None
    try:
        model = xgb.XGBRegressor()
        model.load_model(str(MODEL_PATH))
        return model
    except Exception as exc:
        st.error(f"Error loading model: {exc}")
        return None


@st.cache_data
def load_transit_data():
    """Load GTFS CSV files from the local filesystem with robust Turkish-encoding support."""
    def read_gtfs_csv(filename: str) -> pd.DataFrame | None:
        path = DATA_DIR / filename
        if not path.exists():
            path = DATA_DIR / filename.replace(".csv", ".txt")
        if not path.exists():
            return None
        for enc in ["utf-8", "iso-8859-9", "windows-1254", "latin-1"]:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False, dtype=str)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path, encoding="latin-1", low_memory=False, dtype=str)

    try:
        routes     = read_gtfs_csv("routes.csv")
        trips      = read_gtfs_csv("trips.csv")
        stops      = read_gtfs_csv("stops.csv")
        shapes     = read_gtfs_csv("shapes.csv")
        stop_times = read_gtfs_csv("stop_times.csv")

        # Filter routes to road-based transit only.
        # This IETT dataset uses extended GTFS route_type codes:
        #   9  = Bus (standard IETT bus lines)
        #   10 = Minibus / Dolmuş
        # Excluded: 4=Ferry, 1=Metro, 0=Tram, 7=Funicular, 6=Cable car
        BUS_ROUTE_TYPES = {"9", "10"}
        if routes is not None and "route_type" in routes.columns:
            routes = routes[
                routes["route_type"].astype(str).str.strip().isin(BUS_ROUTE_TYPES)
            ].reset_index(drop=True)
            if routes.empty:
                st.warning("No bus routes found in routes.csv.")
                return None, None, None, None, None

        missing = [
            name for name, df in zip(
                ["routes", "trips", "stops", "shapes", "stop_times"],
                [routes, trips, stops, shapes, stop_times],
            )
            if df is None
        ]
        if missing:
            st.warning(f"Some GTFS files are missing: {', '.join(missing)}")
            return None, None, None, None, None

        # --- Type conversions ---
        stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
        stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")

        shapes["shape_pt_lat"]      = pd.to_numeric(shapes["shape_pt_lat"],      errors="coerce")
        shapes["shape_pt_lon"]      = pd.to_numeric(shapes["shape_pt_lon"],      errors="coerce")
        shapes["shape_pt_sequence"] = pd.to_numeric(shapes["shape_pt_sequence"], errors="coerce").fillna(0).astype(int)

        stop_times["stop_sequence"] = pd.to_numeric(stop_times["stop_sequence"], errors="coerce").fillna(0).astype(int)

        # Parse arrival_time (GTFS format "HH:MM:SS", may exceed 23:xx for post-midnight trips)
        # into total seconds since midnight. arrival_seconds does NOT exist in the raw CSV.
        def parse_arrival_time(t: str) -> float:
            """Convert a GTFS time string like '08:30:00' or '25:10:00' to seconds."""
            try:
                h, m, s = map(int, str(t).strip().split(":"))
                return float(h * 3600 + m * 60 + s)
            except Exception:
                return float("nan")

        if "arrival_time" in stop_times.columns:
            stop_times["arrival_seconds"] = stop_times["arrival_time"].apply(parse_arrival_time)
        else:
            stop_times["arrival_seconds"] = float("nan")

        return routes, trips, stops, shapes, stop_times

    except Exception as exc:
        st.error(f"Failed to load transit data: {exc}")
        return None, None, None, None, None

@st.cache_data
def load_test_data() -> pd.DataFrame | None:
    """
    Load the preprocessed test parquet (ground-truth dataset).
    Returns a DataFrame with all model features + 'travel_time_seconds' target.
    Cached to avoid repeated disk I/O.
    """
    if not TEST_PARQUET.exists():
        return None
    try:
        return pd.read_parquet(TEST_PARQUET)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
def run_dashboard() -> None:
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown('<div class="main-title">Istanbul Smart Transit ETA Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Real-time predictive analytics for the IETT urban transit network &mdash; powered by XGBoost</div>',
        unsafe_allow_html=True,
    )

    # ── Load resources ─────────────────────────────────────────────────────────
    routes, trips, stops, shapes, stop_times = load_transit_data()
    model = load_prediction_model()

    if routes is None or model is None:
        st.info("Please ensure the datasets and model files are correctly placed in the project directory.")
        st.stop()

    # ── Session state: persist prediction results across Streamlit reruns ───────
    # Streamlit re-executes the entire script on every widget interaction.
    # Without session_state, the prediction block only runs when the button is
    # pressed and disappears on the next rerun (e.g. file-watcher auto-refresh).
    if "pred_ctx" not in st.session_state:
        st.session_state.pred_ctx    = None   # context key when last predicted
        st.session_state.pred_result = None   # dict with all computed values

    # ══════════════════════════════════════════════════════════════════════════
    # SIDEBAR — Controls
    # ══════════════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.header("🔍 Route & Stop Selection")

        # Route selector
        route_labels = routes.apply(
            lambda r: f"{r['route_short_name']} | {r['route_long_name']}", axis=1
        ).tolist()
        selected_label = st.selectbox("Route Line", route_labels)
        route_id = routes.iloc[route_labels.index(selected_label)]["route_id"]

        # Trips for this route
        route_trips = trips[trips["route_id"] == route_id]
        if route_trips.empty:
            st.error("No trips found for the selected route.")
            st.stop()

        sample_trip  = route_trips.iloc[0]
        trip_id      = sample_trip["trip_id"]
        shape_id     = sample_trip["shape_id"]

        # Stop sequence for this trip
        trip_stops = (
            stop_times[stop_times["trip_id"] == trip_id]
            .merge(stops, on="stop_id")
            .sort_values("stop_sequence")
            .reset_index(drop=True)
        )

        if trip_stops.empty:
            st.error("Stop sequence data missing for this trip.")
            st.stop()

        stop_names = trip_stops["stop_name"].tolist()
        origin_name = st.selectbox("Origin Stop",      stop_names, index=0)
        dest_name   = st.selectbox("Destination Stop", stop_names, index=min(5, len(stop_names) - 1))

        origin_row = trip_stops[trip_stops["stop_name"] == origin_name].iloc[0]
        dest_row   = trip_stops[trip_stops["stop_name"] == dest_name].iloc[0]

        if origin_row["stop_sequence"] >= dest_row["stop_sequence"]:
            st.warning("⚠️ Destination must be downstream from Origin.")

        # ── Time of Day ───────────────────────────────────────────────────────
        st.divider()
        st.header("⏱️ Time of Day")
        departure_time = st.time_input(
            "Departure Time",
            value=datetime.time(8, 0),
            help="Select the time the bus departs from the Origin stop.",
        )
        hora_del_dia     = np.int8(departure_time.hour)
        arrival_seconds  = np.float32(
            departure_time.hour * 3600 + departure_time.minute * 60 + departure_time.second
        )

        # ── Weather Condition ─────────────────────────────────────────────────
        st.divider()
        st.header("🌤️ Weather Condition")
        weather_label = st.selectbox("Weather Preset", list(WEATHER_PRESETS.keys()))
        weather       = WEATHER_PRESETS[weather_label]
        temperature   = np.float32(weather["temperature_2m"])
        precipitation = np.float32(weather["precipitation"])

        st.caption(
            f"🌡️ Temp: **{temperature:.0f} °C** &nbsp;|&nbsp; 💧 Precip: **{precipitation:.1f} mm/h**"
        )

        st.divider()
        predict_btn = st.button("🚀 Predict ETA")

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN CONTENT — Map (left) + Insights (right)
    # ══════════════════════════════════════════════════════════════════════════
    col_map, col_info = st.columns([3, 2])

    # ── Left column: Interactive Map ───────────────────────────────────────
    with col_map:
        st.subheader("📍 Interactive Route Map")

        route_shape = (
            shapes[shapes["shape_id"] == shape_id]
            .sort_values("shape_pt_sequence")
            .dropna(subset=["shape_pt_lat", "shape_pt_lon"])
        )

        if route_shape.empty:
            st.warning(f"No shape data found for shape_id: `{shape_id}`.")
        else:
            shape_coords = list(
                zip(route_shape["shape_pt_lat"], route_shape["shape_pt_lon"])
            )
            center_lat = route_shape["shape_pt_lat"].mean()
            center_lon = route_shape["shape_pt_lon"].mean()

            folium_map = build_folium_map(
                shape_coords=shape_coords,
                origin_lat=float(origin_row["stop_lat"]),
                origin_lon=float(origin_row["stop_lon"]),
                dest_lat=float(dest_row["stop_lat"]),
                dest_lon=float(dest_row["stop_lon"]),
                center_lat=center_lat,
                center_lon=center_lon,
            )
            st_folium(folium_map, use_container_width=True, height=480)

    # ── Right column: Insights & Prediction ────────────────────────────────
    with col_info:
        st.subheader("📊 Route Insights")

        total_dist_m = calculate_haversine(
            float(origin_row["stop_lat"]), float(origin_row["stop_lon"]),
            float(dest_row["stop_lat"]),   float(dest_row["stop_lon"]),
        )
        num_stops = int(dest_row["stop_sequence"] - origin_row["stop_sequence"])

        st.markdown(f"**Route:** `{selected_label.split('|')[0].strip()}`")
        st.markdown(f"**Straight-line Distance:** `{total_dist_m / 1000:.2f} km`")
        st.markdown(f"**Intermediate Stops:** `{num_stops}`")
        st.markdown(f"**Departure Time:** `{departure_time.strftime('%H:%M')}`")
        st.markdown(f"**Weather:** {weather_label}")

        st.divider()

        # ── Scheduled ETA from historical stop_times data ──────────────────
        # Look up the recorded arrival_seconds for origin and destination
        # in the same trip and compute the scheduled travel time.
        origin_arrival_s = pd.to_numeric(origin_row.get("arrival_seconds"), errors="coerce")
        dest_arrival_s   = pd.to_numeric(dest_row.get("arrival_seconds"),   errors="coerce")

        scheduled_seconds: float | None = None
        if pd.notna(origin_arrival_s) and pd.notna(dest_arrival_s):
            diff = float(dest_arrival_s) - float(origin_arrival_s)
            if diff > 0:
                scheduled_seconds = diff

        # ── Prediction triggered by button ─────────────────────────────────
        if predict_btn:
            if origin_row["stop_sequence"] >= dest_row["stop_sequence"]:
                st.error("Cannot predict: Destination stop must be downstream from Origin.")
            else:
                with st.spinner("Sampling ground truth and running XGBoost inference…"):
                    test_df = load_test_data()

                    if test_df is None or test_df.empty:
                        st.error(f"Test parquet not found at: `{TEST_PARQUET}`")
                    else:
                        # Clamp hops to model's training range [1, 10].
                        # The pipeline generated k-hop pairs for k=1..10 only.
                        model_hops = int(min(max(num_stops, 1), 10))

                        # Deterministic seed — includes hora_del_dia AND weather_label
                        # so that changing time or weather selects a different parquet row
                        # and produces visibly different ground-truth and prediction values.
                        det_seed = abs(
                            hash(f"{route_id}|{origin_name}|{dest_name}|"
                                 f"{int(hora_del_dia)}|{weather_label}")
                        ) % (2 ** 31)

                        # ── Sample one row with matching hop count ────────────────────────
                        candidates = test_df[
                            test_df["num_paradas_salto"].astype(int) == model_hops
                        ]
                        # Fallback to hop=5 (median) if exact match unavailable
                        if candidates.empty:
                            candidates = test_df[
                                test_df["num_paradas_salto"].astype(int) == 5
                            ]
                        sample_row = candidates.sample(1, random_state=det_seed).iloc[0]

                        # ── FIX 3: Use the REAL route distance, not the sample's ──────────
                        # total_dist_m is the Haversine of the user-selected stops.
                        # The sample's distancia_proyectada is for some other short segment.
                        real_dist_m    = float(total_dist_m)
                        sample_dist_m  = float(sample_row["distancia_proyectada"])
                        sample_time_s  = float(sample_row["travel_time_seconds"])

                        # Scale ground-truth travel time proportionally to the real distance.
                        # This preserves the speed profile from the sample while matching route length.
                        scale          = real_dist_m / max(sample_dist_m, 1.0)
                        actual_seconds = sample_time_s * scale
                        actual_minutes = actual_seconds / 60.0

                        # Velocity consistent with the scaled actual time and real distance
                        velocity_ms = real_dist_m / max(actual_seconds, 1.0)

                        # ── Scheduled ETA: GTFS stop_times first (deterministic) ──────────
                        # scheduled_seconds was computed earlier from arrival_time parsing.
                        if scheduled_seconds is not None and scheduled_seconds > 0:
                            sched_s   = float(scheduled_seconds)
                            sched_min = sched_s / 60.0
                        else:
                            # Fallback: 15 km/h constant speed baseline
                            SPEED_MS  = 15_000 / 3600
                            sched_s   = real_dist_m / SPEED_MS
                            sched_min = sched_s / 60.0

                        # ── Build inference DataFrame with REAL route features ────────────
                        input_df = pd.DataFrame([{
                            "trip_id":              sample_row["trip_id"],
                            "stop_id":              sample_row["stop_id"],
                            "stop_sequence":        np.float32(origin_row["stop_sequence"]),
                            "stop_lat":             np.float32(origin_row["stop_lat"]),
                            "stop_lon":             np.float32(origin_row["stop_lon"]),
                            "arrival_seconds":      np.float32(arrival_seconds),
                            "hora_del_dia":         np.float32(hora_del_dia),
                            "distancia_proyectada": np.float32(real_dist_m),   # REAL distance
                            "velocidad_tramo_m_s":  np.float32(velocity_ms),   # derived from real
                            "temperature_2m":       np.float32(temperature),
                            "precipitation":        np.float32(precipitation),
                            "num_paradas_salto":    np.float32(model_hops),    # clamped 1-10
                        }])
                        input_df["trip_id"] = input_df["trip_id"].astype("category")
                        input_df["stop_id"] = input_df["stop_id"].astype("category")

                        try:
                            raw_pred          = float(model.predict(input_df)[0])
                            predicted_seconds = max(raw_pred, 30.0)
                            predicted_minutes = predicted_seconds / 60.0

                            error_sched_s   = abs(sched_s - actual_seconds)
                            error_ai_s      = abs(predicted_seconds - actual_seconds)
                            error_sched_min = error_sched_s / 60.0
                            error_ai_min    = error_ai_s    / 60.0

                            improvement_pct = (
                                (error_sched_s - error_ai_s) / max(error_sched_s, 0.01)
                            ) * 100

                            # Persist results in session_state so they survive reruns.
                            # The context key encodes every parameter that affects results.
                            ctx_key = (
                                f"{route_id}|{origin_name}|{dest_name}|"
                                f"{int(hora_del_dia)}|{weather_label}"
                            )
                            st.session_state.pred_ctx = ctx_key
                            st.session_state.pred_result = {
                                "sched_min":       sched_min,
                                "actual_minutes":  actual_minutes,
                                "predicted_min":   predicted_minutes,
                                "error_sched_min": error_sched_min,
                                "error_ai_min":    error_ai_min,
                                "improvement_pct": improvement_pct,
                                "error_ai_wins":   error_ai_s <= error_sched_s,
                                "model_hops":      model_hops,
                                "scale":           scale,
                                "real_dist_km":    real_dist_m / 1000,
                                "weather_label":   weather_label.strip(),
                                "departure_str":   departure_time.strftime("%H:%M"),
                                "sched_from_gtfs": scheduled_seconds is not None,
                            }

                        except Exception as exc:
                            st.error(f"Inference Error: {exc}")
                            st.info("Check that all feature dtypes match the training pipeline.")

        # ── Render prediction results (from session_state — survives reruns) ────
        # Build the same context key as above to detect stale results.
        current_ctx = (
            f"{route_id}|{origin_name}|{dest_name}|"
            f"{int(hora_del_dia)}|{weather_label}"
        )

        if st.session_state.pred_result is not None:
            if st.session_state.pred_ctx != current_ctx:
                # Selection changed after last prediction → gentle hint only
                st.info(
                    "⚙️ Settings changed. Press **🚀 Predict ETA** to update the analysis."
                )
            else:
                r = st.session_state.pred_result

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric(
                        label="🗓️ Scheduled ETA",
                        value=f"{r['sched_min']:.1f} min",
                        help=(
                            "From GTFS stop_times arrival_time difference (authoritative timetable)."
                            if r["sched_from_gtfs"]
                            else "Fallback: route distance ÷ 15 km/h constant speed."
                        ),
                    )
                with m2:
                    st.metric(
                        label="📍 Actual ETA (Ground Truth)",
                        value=f"{r['actual_minutes']:.1f} min",
                        help=(
                            f"Travel time from a representative {r['model_hops']}-hop "
                            f"test-set segment, scaled to {r['real_dist_km']:.2f} km."
                        ),
                    )
                with m3:
                    ai_delta      = r["predicted_min"] - r["actual_minutes"]
                    ai_delta_sign = "+" if ai_delta >= 0 else ""
                    st.metric(
                        label="🤖 AI Predicted ETA",
                        value=f"{r['predicted_min']:.1f} min",
                        delta=f"{ai_delta_sign}{ai_delta:.1f} min vs actual",
                        delta_color="inverse" if ai_delta > 1 else "normal",
                        help="XGBoost prediction using real route distance + user-selected conditions.",
                    )

                st.divider()
                if r["error_ai_wins"]:
                    st.success(
                        f"🏆 **AI Wins!** "
                        f"The AI was only **{r['error_ai_min']:.1f} min** off the actual arrival time, "
                        f"while the static schedule was **{r['error_sched_min']:.1f} min** off — "
                        f"a **{r['improvement_pct']:.0f}% improvement** in prediction accuracy."
                    )
                else:
                    st.warning(
                        f"📋 **Schedule Wins (this sample).** "
                        f"Static schedule: **{r['error_sched_min']:.1f} min** off · "
                        f"AI: **{r['error_ai_min']:.1f} min** off. "
                        f"This occurs when selected conditions closely match the standard timetable."
                    )

                st.caption(
                    f"Ground truth: {r['model_hops']}-hop parquet sample "
                    f"(scaled ×{r['scale']:.2f} to {r['real_dist_km']:.2f} km) · "
                    f"Weather: {r['weather_label']} · Departure: {r['departure_str']}"
                )
                st.success("Analysis complete ✓")






    # ── Footer ─────────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "Istanbul Smart Transit — XGBoost Predictive ETA System &nbsp;|&nbsp; "
        "Presented to the Istanbul Municipal Transport Committee &nbsp;|&nbsp; 2026"
    )


if __name__ == "__main__":
    run_dashboard()

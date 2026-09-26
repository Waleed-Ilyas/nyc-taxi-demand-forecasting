"""Project-wide constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"  # small committed files the app reads
FIGURES_DIR = ROOT / "reports" / "figures"

# Official NYC TLC yellow-taxi trip records (public parquet, no login) + zone lookup + zone shapes.
TLC_BASE = "https://d37ci6vzurychx.cloudfront.net"
MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07"]
DATA_START = "2026-01-01"
DATA_END = "2026-07-31 23:00"  # last hour with data (inclusive)
UNKNOWN_ZONES = (264, 265)  # TLC 'Unknown' / 'NA' zones

# Open-Meteo historical weather for Manhattan (local time incl. DST; no API key).
LAT, LON, TIMEZONE = 40.7580, -73.9855, "America/New_York"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_VARS = ["temperature_2m", "apparent_temperature", "precipitation", "rain", "snowfall",
                "wind_speed_10m", "cloud_cover", "relative_humidity_2m"]

HORIZON = 24  # forecast the next 24 hours
# Time-based split. Tuning uses the validation month; the test month is scored once.
TRAIN_END = "2026-05-31 23:00"
VALID_START, VALID_END = "2026-06-01 00:00", "2026-06-30 23:00"
TEST_START, TEST_END = "2026-07-01 00:00", "2026-07-31 23:00"
ORIGIN_STEP = 3  # use every 3rd hour as a forecast origin when building training tables
N_MODEL_ZONES = 150  # zones with real volume get a model; the sparse long tail uses a profile
SEED = 42

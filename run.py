import argparse
import asyncio
from pathlib import Path
import sqlite3
import kagglehub
import yaml
from ad_blaster.ad_blaster import AdBlaster
from ad_blaster.mute_providers import create_mute_provider

DEFAULT_KAGGLE_DATASET = "dataclusterlabs/digital-screen-television"


def load_config(config_path="config.yaml"):
    path = Path(config_path)
    if not path.exists() and config_path == "config.yaml":
        path = Path("config.sample.yaml")
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def resolve_dataset_path(dataset_arg: str) -> Path:
    """Resolve a local path or download/cache a Kaggle dataset via kagglehub."""
    local_path = Path(dataset_arg)
    if local_path.exists():
        return local_path
    return Path(kagglehub.dataset_download(dataset_arg))


def init_db(db_path):
    """Initialize the SQLite database and create the detected_categories table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detected_categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            description TEXT,
            logos TEXT
        )
    ''')
    conn.commit()
    return conn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AdBlaster with a live video feed or image dataset.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama decision model to use (default: clef-flash, e.g. clef-flash or nimble).",
    )
    parser.add_argument(
        "--dataset",
        nargs="?",
        const=DEFAULT_KAGGLE_DATASET,
        default=None,
        help=(
            f"Run detections on a dataset. Pass a Kaggle dataset handle "
            f"(defaults to '{DEFAULT_KAGGLE_DATASET}' when flag is given without a value) "
            f"or a local directory path."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of dataset images to process.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Webcam device index (default: 0).",
    )
    parser.add_argument(
        "--window",
        action="store_true",
        help="Also open a separate OpenCV GUI preview window alongside the terminal preview.",
    )
    parser.add_argument("--width", type=int, default=None, help="Resize frame width before sending to Ollama.")
    parser.add_argument("--height", type=int, default=None, help="Resize frame height before sending to Ollama.")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames per inference call.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model is not None:
        config["model"] = args.model
    else:
        config.setdefault("model", "clef-flash")
    if args.camera is not None:
        config.setdefault("webcam", {})["device_id"] = args.camera
    if args.window:
        config.setdefault("webcam", {})["show_window"] = True
    if args.width is not None:
        config.setdefault("webcam", {})["width"] = args.width
    if args.height is not None:
        config.setdefault("webcam", {})["height"] = args.height
    if args.frames is not None:
        config["video_buffer_frames"] = args.frames
    db_path = config.get("db_path", config.get("db", "app.db"))
    db = init_db(db_path)
    mute_provider = create_mute_provider(config)

    blaster = AdBlaster(config, db, mute_provider)
    if args.dataset:
        dataset_path = resolve_dataset_path(args.dataset)
        asyncio.run(blaster.run_dataset(dataset_path, limit=args.limit))
    else:
        asyncio.run(blaster.run())



import argparse
import asyncio
import sqlite3
import yaml
from ad_blaster.ad_blaster import AdBlaster
from ad_blaster.mute_providers import create_mute_provider


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


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
    parser = argparse.ArgumentParser(description="Run AdBlaster with a live video feed.")
    args = parser.parse_args()

    config = load_config()
    db_path = config.get("db", "app.db")
    db = init_db(db_path)
    mute_provider = create_mute_provider(config)

    blaster = AdBlaster(config, db, mute_provider)
    asyncio.run(blaster.run())

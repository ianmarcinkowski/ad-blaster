import argparse
import asyncio
import yaml
from base64 import b64encode
from ad_blaster.ad_blaster import AdBlaster
from ad_blaster.util import image_file_to_base64

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AdBlaster with an image.")
    args = parser.parse_args()

    config = load_config()
    blaster = AdBlaster(config)
    asyncio.run(blaster.run())

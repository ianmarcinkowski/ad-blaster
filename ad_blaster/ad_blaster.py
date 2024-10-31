""" AdBlaster """

import cv2
from ollama import Client as OllamaClient
from enum import Enum
from rich import print as print_rich
from rich.console import Console
from rich.markdown import Markdown
from ad_blaster.prompts import engineered_prompt, user_prompt
from ad_blaster.util import ns_to_ms, find_json, b64encode


class State(Enum):
    UNMUTED = "unmuted"
    MUTED = "muted"


categories = {
    "sports",
    "drama",
    "comedy",
    "advertisement",
    "blank screen",
    "talk show",
    "sci-fi",
    "unknown",
    "news",
}

bad_categories = {
    "advertisement",
    "tv station promo"
}

good_categories = categories - bad_categories


class AdBlaster:

    def __init__(self, config, db):
        self.config = config
        self.llm_hostname = self.config["ollama_uri"]
        self.ollama = OllamaClient(host=self.llm_hostname)
        self.console = Console()
        self.state = State.UNMUTED
        self.db = db

    async def update_state(self, desired_state):
        previous_state = self.state
        if previous_state != desired_state:
            if desired_state == State.UNMUTED:
                await self.send_unmute()
            elif desired_state == State.MUTED:
                await self.send_mute()
            self.state = desired_state

    async def run(self):
        cap = setup_capture(self.config["webcam"]["width"], self.config["webcam"]["height"])
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        cv2.namedWindow("preview")
        while True:
            with self.console.status(
                "[bold green]Fetching image and asking LLM...", spinner="aesthetic"
            ) as status:
                await self.tick(cap)

    async def update_preview(self, frame):
        cv2.imshow("preview", frame)
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyWindow("preview")

    async def tick(self, cap):
        _success, frame = cap.read()
        await self.update_preview(frame)
        _success, buffer = cv2.imencode(".png", frame)
        frame_b64 = b64encode(buffer)
        length = len(frame_b64)
        response = await self.ask(frame_b64)
        output = ""
        if response["json"]:
            output = response["json"]
            decision = self.decide(response["json"])
            self.console.print(f"[yellow] decision is {decision}")
            if decision == State.MUTED:
                await self.update_state(State.MUTED)
            elif decision == State.UNMUTED:
                await self.update_state(State.UNMUTED)
            else:
                self.console.print("[yellow]Not enough signal to determine whether to mute or not")
        else:
            try:
                output = Markdown(response["raw_message"])
            except:
                output = response["raw_message"]
        self.console.print(f"[blue]LLM: {output}")

    def decide(self, response):
        category = response.get("category")
        description = response.get("description", "")
        logos = response.get("logos", "")
        cursor = self.db.cursor()
        cursor.execute('''
            INSERT OR IGNORE INTO detected_categories (category, description, logos) VALUES (?, ?, ?)
        ''', (category, description, logos))
        self.db.commit()

        decision = None
        if category is not None:
            if category in bad_categories:
                decision = State.MUTED
            elif category in good_categories:
                decision = State.UNMUTED
        return decision

    async def ask(self, image_content):
        response = self.ollama.chat(
            model="x/llama3.2-vision",
            messages=[
                {"role": "system", "content": engineered_prompt},
                {"role": "user", "content": user_prompt, "images": [image_content]},
            ],
        )
        found_json = find_json(response["message"]["content"])
        return {
            "duration": ns_to_ms(response["total_duration"]),
            "raw_message": response["message"],
            "json": found_json,
        }

    async def send_unmute(self):
        self.console.print("[red]PLACEHOLDER UNMUTE COMMAND SENT")

    async def send_mute(self):
        self.console.print("[red]PLACEHOLDER MUTE COMMAND SENT")


def setup_capture(x_size, y_size, device_id=0):
    cap = cv2.VideoCapture(device_id)
    cap.set(3, x_size)
    cap.set(4, y_size)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # enable autofocus in case it wasn't
    return cap

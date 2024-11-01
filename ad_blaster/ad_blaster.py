""" AdBlaster """

import cv2
from ollama import Client as OllamaClient
from enum import Enum
from rich import print as print_rich
from rich.console import Console
from rich.markdown import Markdown
from ad_blaster.prompts import engineered_prompt, user_prompt, open_category_prompt
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

neutral_categories = {
    "unknown",
    "blank screen"
}

bad_categories = {
    "advertisement",
    "tv station promo"
}

good_categories = categories - bad_categories - neutral_categories


class AdBlaster:

    def __init__(self, config, db):
        self.config = config
        self.llm_hostname = self.config["ollama_uri"]
        self.ollama = OllamaClient(host=self.llm_hostname)
        self.console = Console()
        self.state = State.UNMUTED
        self.db = db

    def update_state(self, desired_state):
        if self.state != desired_state:
            self.state = desired_state
            if desired_state == State.UNMUTED:
                self.send_unmute()
            elif desired_state == State.MUTED:
                self.send_mute()

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
        _, frame = cap.read()
        await self.update_preview(frame)
        _, buffer = cv2.imencode(".png", frame)
        frame_b64 = b64encode(buffer)
        response = await self.ask(frame_b64)
        if response["json"]:
            category = response.get("category")
            description = response.get("description", "")
            logos = str(response.get("logos", ""))
            self.log_category(category, description, logos)

            decision = self.decide(response["json"])
            self.console.print(f"[yellow] decision is {decision}")
            if decision == State.MUTED:
                self.update_state(State.MUTED)
            elif decision == State.UNMUTED:
                self.update_state(State.UNMUTED)
            else:
                self.console.print("[yellow]Not enough signal to determine whether to mute or not")
            console_output = response["json"]
        else:
            try:
                console_output = Markdown(response["raw_message"])
            except:
                console_output = response["raw_message"]
        self.console.print(f"[blue]LLM: {console_output}")

    def log_category(self, category, description, logos):
        cursor = self.db.cursor()
        cursor.execute('''
            INSERT OR IGNORE INTO detected_categories (category, description, logos) VALUES (?, ?, ?)
        ''', (category, description, logos))
        self.db.commit()


    async def ask(self, image_content):
        response = self.ollama.chat(
            model="x/llama3.2-vision",
            messages=[
                {"role": "system", "content": open_category_prompt},
                {"role": "user", "content": user_prompt, "images": [image_content]},
            ],
        )
        found_json = find_json(response["message"]["content"])
        return {
            "duration": ns_to_ms(response["total_duration"]),
            "raw_message": response["message"],
            "json": found_json,
        }

    def send_unmute(self):
        self.console.print("[red]PLACEHOLDER UNMUTE COMMAND SENT")

    def send_mute(self):
        self.console.print("[red]PLACEHOLDER MUTE COMMAND SENT")


def setup_capture(x_size, y_size, device_id=0):
    cap = cv2.VideoCapture(device_id)
    cap.set(3, x_size)
    cap.set(4, y_size)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # enable autofocus in case it wasn't
    return cap


def decide(category):
    decision = None
    if category:
        if category in bad_categories:
            decision = State.MUTED
        elif category in neutral_categories:
            return None
        elif category in good_categories:
            decision = State.UNMUTED
    return decision

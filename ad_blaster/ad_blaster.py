""" AdBlaster """

import cv2
from ollama import Client as OllamaClient
from rich.console import Console
from rich.markdown import Markdown
from ad_blaster.prompts import engineered_prompt, user_prompt
from ad_blaster.util import ns_to_ms, find_json, b64encode


class AdBlaster:

    def __init__(self, config):
        self.config = config
        self.llm_hostname = self.config["ollama_uri"]
        self.ollama = OllamaClient(host=self.llm_hostname)
        self.console = Console()
        self.state = "init"

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
        if response["json"]:
            self.console.print(response["json"])
        else:
            try:
                output = Markdown(response["raw_message"])
            except:
                output = response["raw_message"]
            self.console.print(output)

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


def setup_capture(x_size, y_size, device_id=0):
    cap = cv2.VideoCapture(device_id)
    cap.set(3, x_size)
    cap.set(4, y_size)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # enable autofocus in case it wasn't
    return cap

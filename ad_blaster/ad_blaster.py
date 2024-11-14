""" AdBlaster """

import cv2
import re
import serial
from ollama import Client as OllamaClient
from enum import Enum
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.pretty import Pretty
from ad_blaster.prompts import *
from ad_blaster.util import ns_to_ms, b64encode


class State(Enum):
    UNMUTED = "unmuted"
    MUTED = "muted"


class AdBlaster:

    def __init__(self, config, db):
        self.config = config
        self.llm_hostname = self.config["ollama_uri"]
        self.ollama = OllamaClient(host=self.llm_hostname)
        self.state = State.UNMUTED
        self.db = db
        self.metrics_csv_path = "metrics.csv"
        with open(self.metrics_csv_path, "w") as csv:
            csv.write("advertising_detected, load, prompt_eval, eval, total, reason\n")

        arduino_device = self.config.get("arduino_tty_device")
        if arduino_device:
            try:
                # Quick way of only sending the signal to mute if we can talk to the device
                self.arduino = serial.Serial(port=arduino_device, baudrate=9600, timeout=0.1)
            except:
                self.arduino = None
        else:
            self.arduino = None
        self.layout = Layout()
        self.metrics = Table("advertising_detected", "load", "prompt_eval", "eval", "total", "reason")
        self.layout.split_column(
            Layout(str(self.state), name="status", size=2),
            Layout(name="progress", size=4),
            Layout(name="llm_output", size=4),
            Layout(name="raw_llm_message", size=4),
            Layout(self.metrics, name="metrics"),
        )


    async def run(self):
        cap = setup_capture(self.config["webcam"]["width"], self.config["webcam"]["height"])
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        cv2.namedWindow("preview")
        while True:
            with Live(self.layout) as live_update:
                await self.tick(cap)

    def update_state(self, desired_state):
        self.layout["status"].update(str(self.state))
        if self.state != desired_state:
            self.state = desired_state
            if desired_state == State.UNMUTED:
                self.layout["status"].update(
                    f"[white]{self.state} [blue]SENT COMMAND TO [green]UNMUTE TV"
                )
                self.send_unmute()
            elif desired_state == State.MUTED:
                self.layout["status"].update(
                    f"[white]{self.state} [blue]SENT COMMAND TO [red]MUTE TV"
                )
                self.send_mute()

    async def tick(self, cap):
        _, frame = cap.read()
        await self.update_preview(frame)
        _, buffer = cv2.imencode(".png", frame)
        frame_b64 = b64encode(buffer)
        if True:
            self.layout["progress"].update("Fetching answer from LLM...")
            description, advertising_detected, reason, metrics = await self.combined_detection(
                frame_b64
            )
            self.layout["progress"].update("Analyzing...")
            display_output = f"""
                [yellow]llama3.2-vision thinks advertising_detected: {advertising_detected}
            """
            if reason:
                display_output += f"\n[yellow]{reason}"
            if description:
                display_output += f"[yellow]{description}"
            self.layout["llm_output"].update(display_output)

        if advertising_detected:
            self.update_state(State.MUTED)
        else:
            self.update_state(State.UNMUTED)

    async def update_preview(self, frame):
        cv2.imshow("preview", frame)
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyWindow("preview")

    def log_metrics(self, advertising_detected, metrics, reason):
        with open(self.metrics_csv_path, "a") as csv_file:
            data = ", ".join([
                    str(advertising_detected),
                    metrics["load_duration"],
                    metrics["prompt_eval_duration"],
                    metrics["eval_duration"],
                    metrics["total_duration"],
                    str(reason),
                ])
            csv_file.write(f"{data}\n")
        self.metrics.add_row(
            str(advertising_detected),
            metrics["load_duration"],
            metrics["prompt_eval_duration"],
            metrics["eval_duration"],
            metrics["total_duration"],
            reason.strip())
        self.layout["metrics"].update(self.metrics)

    def log_category(self, category, description, logos):
        cursor = self.db.cursor()
        cursor.execute(
            """
            INSERT OR IGNORE INTO detected_categories (category, description, logos) VALUES (?, ?, ?)
        """,
            (category, description, logos),
        )
        self.db.commit()

    async def combined_detection(self, image_content):
        response = self.ollama.chat(
            model="x/llama3.2-vision",
            messages=[
                {"role": "system", "content": combined_system_prompt},
                {"role": "user", "content": combined_user_prompt, "images": [image_content]},
            ],
        )
        metrics = {
            "load_duration": str(ns_to_ms(response["load_duration"])),
            "prompt_eval_duration": str(ns_to_ms(response["prompt_eval_duration"])),
            "eval_duration": str(ns_to_ms(response["eval_duration"])),
            "total_duration": str(ns_to_ms(response["total_duration"])),
        }
        message_content = response.get("message", {}).get("content")
        self.layout["raw_llm_message"].update(Pretty(response["message"]))
        message_content = message_content.lower()
        description = None
        advertising_detected = False
        reason = None
        if message_content:
            advertising_detected = find_tool_use(message_content)
            reason = self.find_reason_string(message_content)

        self.log_metrics(advertising_detected, metrics, reason)
        return description, advertising_detected, reason, metrics

    def send_unmute(self):
        if self.arduino:
            self.arduino.write(b"2")

    def send_mute(self):
        if self.arduino:
            self.arduino.write(b"1")


def setup_capture(x_size, y_size, device_id=0):
    cap = cv2.VideoCapture(device_id)
    cap.set(3, x_size)
    cap.set(4, y_size)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # enable autofocus in case it wasn't
    return cap


def find_tool_use(raw):
    """
    Tries to find a function call from the output.
    """
    function_call_regex = re.compile(r"\b(\w*)\((.*)\)")
    matches = function_call_regex.findall(raw)
    if matches:
        function_name = matches[0][0]
        function_args = matches[0][1]
        return function_name, function_args
    else:
        # TODO decide what to do when the function call is ambiguous
        block_start = "```python"
        block_start_idx = raw.find(block_start)
        if block_start_idx >= 0:
            content_start_idx = block_start_idx + len(block_start)
            remaining = raw[content_start_idx:]
            end_idx = remaining.find("```")
            if end_idx >= 0:
                tool_use = raw[content_start_idx:end_idx]
            else:
                tool_use = raw[content_start_idx:]
        return tool_use


def find_reason_string(self, raw):
    reason = None
    if "reason:" in raw:
        reason_idx = raw.find("reason:")
        reason = raw[reason_idx:]
    return reason

""" AdBlaster """

import asyncio
import collections
import cv2
import re
from ollama import AsyncClient as OllamaAsyncClient
from enum import Enum
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from ad_blaster.prompts import *
from ad_blaster.util import ns_to_ms, b64encode
from ad_blaster.mute_providers.base import MuteProvider


class State(Enum):
    UNMUTED = "unmuted"
    MUTED = "muted"


class AdBlaster:

    def __init__(self, config, db, mute_provider: MuteProvider):
        self.config = config
        self.llm_hostname = self.config["ollama_uri"]
        self.model = self.config.get("model", "gemma3")
        self.ollama = OllamaAsyncClient(host=self.llm_hostname)
        self.state = State.UNMUTED
        self.db = db
        self.mute_provider = mute_provider
        self.metrics_csv_path = "metrics.csv"

        buffer_size = self.config.get("video_buffer_frames", 4)
        self.frame_buffer: collections.deque = collections.deque(maxlen=buffer_size)

        with open(self.metrics_csv_path, "w") as csv:
            csv.write("advertising_detected, load, prompt_eval, eval, total, reason\n")

        self.layout = Layout()
        self.metrics = Table("advertising_detected", "load", "prompt_eval", "eval", "total", "reason")
        self.layout.split_column(
            Layout(str(self.state), name="status", size=2),
            Layout(name="progress", size=4),
            Layout(name="llm_output", size=4),
            Layout(name="raw_llm_message", size=8),
            Layout(self.metrics, name="metrics"),
        )

    async def run(self):
        cap = setup_capture(self.config["webcam"]["width"], self.config["webcam"]["height"])
        try:
            with Live(self.layout):
                await asyncio.gather(
                    self._capture_loop(cap),
                    self._inference_loop(),
                )
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.mute_provider.close()

    async def _capture_loop(self, cap):
        """Continuously read frames from the webcam into the sliding frame buffer.

        cap.read() is blocking so it runs in a thread executor. cv2 display calls
        happen after the await and are therefore back on the main thread (the asyncio
        event loop thread), which is required on macOS.
        """
        loop = asyncio.get_event_loop()
        while True:
            ret, frame = await loop.run_in_executor(None, cap.read)
            if not ret:
                await asyncio.sleep(0.01)
                continue
            cv2.imshow("preview", frame)
            if cv2.waitKey(1) == ord("q"):
                break
            _, buffer = cv2.imencode(".png", frame)
            self.frame_buffer.append(b64encode(buffer))

    async def _inference_loop(self):
        """Consume the current frame buffer and run ad detection in a continuous loop."""
        while True:
            if not self.frame_buffer:
                await asyncio.sleep(0.05)
                continue

            frames = list(self.frame_buffer)
            self.layout["progress"].update(f"Running inference on {len(frames)} frame(s)...")
            advertising_detected, reason, metrics = await self.detect_ads(frames)
            self.layout["progress"].update("Waiting for next frames...")

            display_output = f"\n[yellow]{self.model} → advertising_detected: {advertising_detected}"
            if reason:
                display_output += f"\n[yellow]{reason}"
            self.layout["llm_output"].update(display_output)

            if advertising_detected:
                self.update_state(State.MUTED)
            else:
                self.update_state(State.UNMUTED)

    def update_state(self, desired_state):
        self.layout["status"].update(str(self.state))
        if self.state != desired_state:
            self.state = desired_state
            if desired_state == State.UNMUTED:
                self.layout["status"].update(
                    f"[white]{self.state} [blue]SENT COMMAND TO [green]UNMUTE TV"
                )
                self.mute_provider.unmute()
            elif desired_state == State.MUTED:
                self.layout["status"].update(
                    f"[white]{self.state} [blue]SENT COMMAND TO [red]MUTE TV"
                )
                self.mute_provider.mute()

    async def detect_ads(self, frames: list[str]):
        """Send a sequence of frames to the model and stream the response back.

        Args:
            frames: list of base64-encoded PNG strings (oldest → newest)

        Returns:
            (advertising_detected, reason, metrics)
        """
        stream = await self.ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": combined_system_prompt},
                {"role": "user", "content": combined_user_prompt, "images": frames},
            ],
            stream=True,
        )

        full_content = ""
        metrics = {}
        async for chunk in stream:
            token = chunk.message.content or ""
            full_content += token
            self.layout["raw_llm_message"].update(full_content)
            if chunk.done:
                metrics = {
                    "load_duration": str(ns_to_ms(chunk.load_duration or 0)),
                    "prompt_eval_duration": str(ns_to_ms(chunk.prompt_eval_duration or 0)),
                    "eval_duration": str(ns_to_ms(chunk.eval_duration or 0)),
                    "total_duration": str(ns_to_ms(chunk.total_duration or 0)),
                }

        lowered = full_content.lower()
        advertising_detected = find_tool_use(lowered)
        reason = find_reason_string(lowered)
        self.log_metrics(advertising_detected, metrics, reason)
        return advertising_detected, reason, metrics

    def log_metrics(self, advertising_detected, metrics, reason):
        if not metrics:
            return
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
            str(reason or "").strip())
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


def setup_capture(x_size, y_size, device_id=0):
    cap = cv2.VideoCapture(device_id)
    cap.set(3, x_size)
    cap.set(4, y_size)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    return cap


def find_tool_use(raw):
    """Tries to find a function call from the output."""
    function_call_regex = re.compile(r"\b(\w*)\((.*)\)")
    matches = function_call_regex.findall(raw)
    if matches:
        function_name = matches[0][0]
        function_args = matches[0][1]
        return function_name, function_args
    else:
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


def find_reason_string(raw):
    """Extract the 'Reason: ...' line from a model response, if present."""
    if "reason:" in raw:
        reason_idx = raw.find("reason:")
        return raw[reason_idx:]
    return None

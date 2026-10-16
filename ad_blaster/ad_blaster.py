""" AdBlaster """

import asyncio
from base64 import b64decode
import collections
import cv2
import json
import numpy as np
import os
import re
import select
import sys
import termios
import time
import tty
from ollama import AsyncClient as OllamaAsyncClient, SystemOneResponse
from enum import Enum
from pathlib import Path
from rich import box
from rich.align import Align
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from ad_blaster.prompts import *
from ad_blaster.util import ns_to_ms, b64encode
from ad_blaster.mute_providers.base import MuteProvider


class State(Enum):
    UNMUTED = "unmuted"
    MUTED = "muted"


def render_Probability_bar(prob: float, width: int = 24, color: str = "cyan", threshold: float | None = None) -> Text:
    """Render a smooth Unicode horizontal probability bar with an optional threshold marker."""
    blocks = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]
    prob = max(0.0, min(1.0, float(prob)))
    total_eighths = int(round(prob * width * 8))
    full_blocks = total_eighths // 8
    remainder = total_eighths % 8

    thresh_idx = int(round(threshold * width)) if threshold is not None else None
    bar = Text()
    for i in range(width):
        if thresh_idx is not None and i == thresh_idx:
            bar.append("│", style="bold yellow")
            continue
        if i < full_blocks:
            bar.append("█", style=color)
        elif i == full_blocks and remainder > 0:
            bar.append(blocks[remainder], style=color)
        else:
            bar.append("░", style="bright_black")
    return bar


def render_frame_ansi(frame_bgr: np.ndarray, width_chars: int = 30, height_chars: int = 15) -> Text:
    """Render an OpenCV BGR image as half-block ('▀') truecolor ANSI art."""
    resized = cv2.resize(frame_bgr, (width_chars, height_chars * 2), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    canvas = Text(no_wrap=True, overflow="crop")
    for y in range(0, height_chars * 2, 2):
        for x in range(width_chars):
            r1, g1, b1 = (int(v) for v in rgb[y, x])
            r2, g2, b2 = (int(v) for v in rgb[y + 1, x])
            canvas.append(
                "▀",
                style=Style(color=f"rgb({r1},{g1},{b1})", bgcolor=f"rgb({r2},{g2},{b2})"),
            )
        if y + 2 < height_chars * 2:
            canvas.append("\n")
    return canvas


def decode_b64_to_bgr(frame_b64: str) -> np.ndarray | None:
    """Decode a base64-encoded image string into an OpenCV BGR array."""
    try:
        raw = b64decode(frame_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


class AdBlaster:

    def __init__(self, config, db, mute_provider: MuteProvider):
        self.config = config
        self.llm_hostname = self.config.get("ollama_uri", "http://localhost:11434")
        self.model = self.config.get("model", "clef-flash")
        self.ad_threshold = self.config.get("ad_detection_threshold", 0.5)
        self.ollama = OllamaAsyncClient(host=self.llm_hostname)
        self.state = State.UNMUTED
        self.db = db
        self.mute_provider = mute_provider
        self.metrics_csv_path = "metrics.csv"

        self.inference_count = 0
        self.ads_detected_count = 0
        self.current_source_label = "live"
        self.last_transition_text = "—"
        self.history_rows: collections.deque = collections.deque(maxlen=10)

        buffer_size = self.config.get("video_buffer_frames", 4)
        self.frame_buffer: collections.deque = collections.deque(maxlen=buffer_size)

        with open(self.metrics_csv_path, "w") as csv:
            csv.write("advertising_detected, load, prompt_eval, eval, total, reason\n")

        self.layout = Layout()
        self.metrics = Table("advertising_detected", "load", "prompt_eval", "eval", "total", "reason")
        self.layout.split_column(
            Layout(name="status", size=3),
            Layout(name="progress", size=3),
            Layout(name="main", size=17),
            Layout(name="metrics", minimum_size=8),
        )
        self.layout["main"].split_row(
            Layout(name="frame_preview", ratio=5),
            Layout(name="right_col", ratio=7),
        )
        self.layout["right_col"].split_column(
            Layout(name="llm_output", size=10),
            Layout(name="raw_llm_message", size=7),
        )

        self._refresh_status_panel()
        self.layout["progress"].update(
            Panel(Text("Idle — waiting for video frames...", style="dim"), title="[bold]Activity[/]", border_style="bright_black")
        )
        self.layout["frame_preview"].update(
            Panel(Align.center(Text("No frame captured yet", style="dim"), vertical="middle"), title="[bold cyan]Frame Preview[/]", border_style="cyan")
        )
        self.layout["llm_output"].update(
            Panel(Text("Waiting for first inference...", style="dim"), title="[bold yellow]Clef-Flash Decision & Probabilities[/]", border_style="yellow")
        )
        self.layout["raw_llm_message"].update(
            Panel(Text("{}", style="dim"), title="[bold magenta]/v1/systemone JSON[/]", border_style="magenta")
        )
        self.layout["metrics"].update(self._build_history_panel())

    def _refresh_status_panel(self, transition_alert: str | None = None):
        if self.state == State.MUTED:
            badge = Text(" 🔇 TV MUTED (AD ACTIVE) ", style="bold white on red")
            border = "red"
        else:
            badge = Text(" 🔊 TV UNMUTED ", style="bold black on green")
            border = "green"

        info = Text()
        info.append_text(badge)
        info.append(f"   Model: ", style="dim")
        info.append(f"{self.model}", style="bold cyan")
        info.append(f"   Threshold: ", style="dim")
        info.append(f"{self.ad_threshold:.2f}", style="bold yellow")
        info.append(f"   Inferences: ", style="dim")
        info.append(f"{self.inference_count}", style="bold white")
        info.append(f"   Ads Flagged: ", style="dim")
        info.append(f"{self.ads_detected_count}", style="bold red" if self.ads_detected_count else "bold white")
        if transition_alert:
            info.append(f"   ⚡ {transition_alert}", style="bold yellow")
        info.append("   [q] Quit", style="bold magenta")

        self.layout["status"].update(Panel(info, border_style=border, box=box.ROUNDED))

    def _update_progress(self, message: str, style: str = "bold cyan"):
        self.layout["progress"].update(
            Panel(Text(message, style=style), title="[bold]Status[/]", border_style="bright_black", box=box.ROUNDED)
        )

    def _update_frame_preview(self, frame_bgr: np.ndarray | None, label: str = ""):
        if frame_bgr is None:
            return
        art = render_frame_ansi(frame_bgr, width_chars=28, height_chars=12)
        title = f"[bold cyan]Preview[/]" + (f" [dim]{label}[/]" if label else "")
        self.layout["frame_preview"].update(
            Panel(Align.center(art, vertical="middle"), title=title, border_style="cyan", box=box.ROUNDED)
        )

    def _render_probability_panel(self, ad_prob: float, advertising_detected: bool, category_answer) -> Panel:
        header = Text()
        if advertising_detected:
            header.append(" AD DETECTED ", style="bold white on red")
            ad_color = "red"
        else:
            header.append(" REGULAR CONTENT ", style="bold black on green")
            ad_color = "green"

        header.append(f"  P(ad) = {ad_prob * 100:5.1f}% ", style=f"bold {ad_color}")
        header.append(f"(threshold {self.ad_threshold * 100:.0f}%)  ", style="dim")
        header.append_text(render_Probability_bar(ad_prob, width=22, color=ad_color, threshold=self.ad_threshold))

        cat_table = Table.grid(padding=(0, 1))
        cat_table.add_column("Category", style="white", width=15)
        cat_table.add_column("Bar", width=20)
        cat_table.add_column("Prob", justify="right", width=7)

        if category_answer is not None and hasattr(category_answer, "probabilities"):
            winning = getattr(category_answer, "choice", None)
            conf = getattr(category_answer, "confidence", 0.0)
            sorted_cats = sorted(
                category_answer.probabilities.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )
            for cat_name, prob in sorted_cats:
                is_win = cat_name == winning
                bar_color = "red" if cat_name in {"advertisement", "promo"} and is_win else ("cyan" if is_win else "bright_blue")
                label_style = f"bold {bar_color}" if is_win else "dim"
                prefix = "▸ " if is_win else "  "
                cat_table.add_row(
                    Text(f"{prefix}{cat_name}", style=label_style),
                    render_Probability_bar(prob, width=18, color=bar_color),
                    Text(f"{prob * 100:5.1f}%", style=label_style),
                )
            subtitle = f"[dim]Top Category:[/] [bold]{winning}[/] [dim](conf={conf:.2f})[/]"
        else:
            subtitle = ""

        return Panel(
            Group(header, Text(""), cat_table),
            title="[bold yellow]Clef-Flash Decision & Probabilities[/]",
            subtitle=subtitle,
            border_style="red" if advertising_detected else "yellow",
            box=box.ROUNDED,
        )

    def _build_history_panel(self) -> Panel:
        table = Table(expand=True, box=box.SIMPLE_HEAVY, show_edge=False)
        table.add_column("#", justify="right", style="dim", width=4)
        table.add_column("Source", style="cyan", no_wrap=True, ratio=3)
        table.add_column("State / Transition", no_wrap=True, ratio=3)
        table.add_column("Ad?", justify="center", width=6)
        table.add_column("P(Ad)", justify="right", width=8)
        table.add_column("Category (Conf)", ratio=3)
        table.add_column("Tokens", justify="right", width=8)
        table.add_column("Latency", justify="right", width=9)

        for row in self.history_rows:
            table.add_row(*row)

        return Panel(
            table,
            title="[bold blue]Detection History & State Transitions[/]",
            border_style="blue",
            box=box.ROUNDED,
        )

    async def _key_listener(self, stop_event: asyncio.Event | None = None):
        """Listen for a single 'q' keypress on stdin (cbreak mode) to quit cleanly."""
        if not sys.stdin.isatty():
            await asyncio.Event().wait()
            return

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                if stop_event is not None and stop_event.is_set():
                    return
                rlist, _, _ = select.select([fd], [], [], 0.0)
                if rlist:
                    ch = os.read(fd, 1).decode("utf-8", errors="ignore")
                    if ch.lower() == "q":
                        if stop_event is not None:
                            stop_event.set()
                        self._update_progress("Quitting (pressed 'q')...", style="bold magenta")
                        return
                await asyncio.sleep(0.05)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    async def run(self):
        webcam_cfg = self.config.get("webcam", {})
        device_id = webcam_cfg.get("device_id", 0)
        show_window = webcam_cfg.get("show_window", False)
        cap = setup_capture(
            webcam_cfg.get("width", 800),
            webcam_cfg.get("height", 450),
            device_id=device_id,
        )
        try:
            with Live(self.layout, refresh_per_second=8):
                capture_task = asyncio.create_task(
                    self._capture_loop(cap, show_window=show_window)
                )
                inference_task = asyncio.create_task(self._inference_loop())
                key_task = asyncio.create_task(self._key_listener())
                done, pending = await asyncio.wait(
                    {capture_task, inference_task, key_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for task in done:
                    if not task.cancelled() and task.exception():
                        raise task.exception()
        finally:
            cap.release()
            if show_window:
                cv2.destroyAllWindows()
            self.mute_provider.close()

    def _encode_frame(self, frame: np.ndarray) -> tuple[np.ndarray, str]:
        """Resize a BGR frame to configured dimensions and JPEG-encode to base64."""
        webcam_cfg = self.config.get("webcam", {})
        target_w = webcam_cfg.get("width", 448)
        target_h = webcam_cfg.get("height", 252)
        jpeg_quality = webcam_cfg.get("jpeg_quality", 75)

        h, w = frame.shape[:2]
        if (w, h) != (target_w, target_h):
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            resized = frame

        _, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        return resized, b64encode(buffer).decode("utf-8")

    async def run_dataset(self, dataset_dir: str | Path, limit: int | None = None):
        """Run ad detection across images in a dataset directory."""
        dataset_path = Path(dataset_dir)
        image_paths = sorted(
            p for p in dataset_path.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if limit is not None:
            image_paths = image_paths[:limit]

        stop_event = asyncio.Event()
        key_task = asyncio.create_task(self._key_listener(stop_event=stop_event))
        try:
            with Live(self.layout, refresh_per_second=8):
                for idx, img_path in enumerate(image_paths, start=1):
                    if stop_event.is_set():
                        break
                    frame = cv2.imread(str(img_path))
                    if frame is None:
                        continue
                    resized, frame_b64 = self._encode_frame(frame)

                    short_name = img_path.name[:16] + ("…" if len(img_path.name) > 16 else "")
                    self.current_source_label = short_name
                    self._update_progress(
                        f"[{idx}/{len(image_paths)}] Running clef-flash inference on {img_path.name} (press 'q' to quit)...",
                        style="bold yellow",
                    )
                    await self.detect_ads([frame_b64])

                if not stop_event.is_set():
                    self._update_progress(f"✓ Completed {len(image_paths)} dataset image(s).", style="bold green")
        finally:
            stop_event.set()
            key_task.cancel()
            await asyncio.gather(key_task, return_exceptions=True)
            self.mute_provider.close()

    async def _capture_loop(self, cap, show_window: bool = False):
        """Continuously read frames from the webcam into the sliding frame buffer.

        cap.read() is blocking so it runs in a thread executor. cv2 display calls
        happen after the await and are therefore back on the main thread (the asyncio
        event loop thread), which is required on macOS.
        """
        loop = asyncio.get_event_loop()
        failed_reads = 0
        while True:
            ret, frame = await loop.run_in_executor(None, cap.read)
            if not ret or frame is None:
                failed_reads += 1
                if failed_reads == 20:
                    self._update_progress(
                        "⚠️ Waiting for camera feed — check macOS System Settings → Privacy & Security → Camera permissions for your terminal app.",
                        style="bold red",
                    )
                await asyncio.sleep(0.1)
                continue
            failed_reads = 0
            resized, frame_b64 = self._encode_frame(frame)
            if show_window:
                cv2.imshow("preview", resized)
                if cv2.waitKey(1) == ord("q"):
                    break
            raw_kb = (len(frame_b64) * 3 / 4) / 1024
            h, w = resized.shape[:2]
            self._update_frame_preview(resized, label=f"webcam {w}x{h} ({raw_kb:.1f} KB)")
            self.frame_buffer.append(frame_b64)
            await asyncio.sleep(0.1)

    async def _inference_loop(self):
        """Consume the current frame buffer and run ad detection in a continuous loop."""
        while True:
            if not self.frame_buffer:
                await asyncio.sleep(0.05)
                continue

            frames = list(self.frame_buffer)
            self.current_source_label = f"webcam {len(frames)}f"
            self._update_progress(f"Running clef-flash inference on {len(frames)} frame(s)...", style="bold yellow")
            await self.detect_ads(frames)
            self._update_progress("Waiting for next frames...", style="dim")

    def update_state(self, desired_state):
        if self.state != desired_state:
            prev_state = self.state
            self.state = desired_state
            if desired_state == State.UNMUTED:
                self.last_transition_text = f"⚡ {prev_state.value.upper()} → UNMUTED"
                self._refresh_status_panel(transition_alert="SENT COMMAND TO UNMUTE TV")
                self.mute_provider.unmute()
            elif desired_state == State.MUTED:
                self.last_transition_text = f"⚡ {prev_state.value.upper()} → MUTED"
                self._refresh_status_panel(transition_alert="SENT COMMAND TO MUTE TV")
                self.mute_provider.mute()
        else:
            self.last_transition_text = f"{self.state.value}"
            self._refresh_status_panel()

    async def detect_ads(self, frames: list[str | bytes]):
        """Send a sequence of frames to the local clef-flash decision model via Ollama.

        Args:
            frames: list of base64-encoded PNG/JPG strings (oldest → newest)

        Returns:
            (advertising_detected, reason, metrics)
        """
        encoded_frames = [
            f.decode("utf-8") if isinstance(f, bytes) else f
            for f in frames
        ]

        total_bytes = sum((len(f) * 3) // 4 for f in encoded_frames)
        total_kb = total_bytes / 1024.0
        dims_str = ""
        if encoded_frames:
            decoded_bgr = decode_b64_to_bgr(encoded_frames[-1])
            if decoded_bgr is not None:
                h, w = decoded_bgr.shape[:2]
                dims_str = f"{w}x{h}, "
                self._update_frame_preview(
                    decoded_bgr,
                    label=f"{self.current_source_label} ({dims_str}{total_kb:.1f} KB)",
                )

        source_with_size = f"{self.current_source_label} ({dims_str}{total_kb:.1f} KB)"

        start_ns = time.perf_counter_ns()
        response = await self.ollama._request(
            SystemOneResponse,
            "POST",
            "/v1/systemone",
            json={
                "model": self.model,
                "state": clef_state,
                "images": encoded_frames,
                "questions": clef_questions,
            },
        )
        elapsed_ns = time.perf_counter_ns() - start_ns

        raw_json = json.dumps(response.model_dump(), indent=2)
        self.layout["raw_llm_message"].update(
            Panel(
                Syntax(raw_json, "json", theme="monokai", word_wrap=True),
                title="[bold magenta]/v1/systemone JSON[/]",
                border_style="magenta",
                box=box.ROUNDED,
            )
        )

        ad_answer = response.answers["advertising_detected"]
        ad_prob = float(ad_answer.noul)
        advertising_detected = ad_prob >= self.ad_threshold

        self.inference_count += 1
        if advertising_detected:
            self.ads_detected_count += 1
            self.update_state(State.MUTED)
        else:
            self.update_state(State.UNMUTED)

        category_answer = response.answers.get("category")
        if category_answer is not None and hasattr(category_answer, "choice"):
            reason = (
                f"p(ad)={ad_prob:.3f} | category={category_answer.choice} "
                f"(conf={category_answer.confidence:.2f})"
            )
        else:
            reason = f"p(ad)={ad_prob:.3f}"

        self.layout["llm_output"].update(
            self._render_probability_panel(ad_prob, advertising_detected, category_answer)
        )

        metrics = {
            "load_duration": "0",
            "prompt_eval_duration": str(response.usage.input_tokens),
            "eval_duration": f"{ad_prob:.4f}",
            "total_duration": str(ns_to_ms(elapsed_ns)),
        }

        self.log_metrics(
            advertising_detected,
            metrics,
            reason,
            category_answer=category_answer,
            source_display=source_with_size,
        )
        return advertising_detected, reason, metrics

    def log_metrics(self, advertising_detected, metrics, reason, category_answer=None, source_display=None):
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
            str(reason or "").strip(),
        )

        ad_badge = Text("YES", style="bold white on red") if advertising_detected else Text("NO", style="green")
        prob_val = float(metrics["eval_duration"]) if metrics.get("eval_duration") else 0.0
        prob_text = Text(f"{prob_val * 100:5.1f}%", style="bold red" if advertising_detected else "green")

        if "→ MUTED" in self.last_transition_text:
            state_cell = Text(f"🔇 {self.last_transition_text}", style="bold red")
        elif "→ UNMUTED" in self.last_transition_text:
            state_cell = Text(f"🔊 {self.last_transition_text}", style="bold green")
        elif self.state == State.MUTED:
            state_cell = Text("🔇 muted", style="red")
        else:
            state_cell = Text("🔊 unmuted", style="green")

        if category_answer is not None and hasattr(category_answer, "choice"):
            cat_cell = f"{category_answer.choice} ({category_answer.confidence:.2f})"
        else:
            cat_cell = str(reason or "").strip()

        latency_s = float(metrics["total_duration"])
        self.history_rows.appendleft((
            str(self.inference_count),
            source_display or self.current_source_label,
            state_cell,
            ad_badge,
            prob_text,
            cat_cell,
            f"{metrics['prompt_eval_duration']}",
            f"{latency_s:.2f}s",
        ))
        self.layout["metrics"].update(self._build_history_panel())

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
    import sys
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(device_id, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam device {device_id}. On macOS, ensure your terminal app "
            "has Camera permission in System Settings → Privacy & Security → Camera, "
            "or try a different camera index with --camera 1."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, x_size)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, y_size)
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

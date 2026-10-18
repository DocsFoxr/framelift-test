"""framelift's visual version: a local web page to enhance videos without the command line.

Start it with ``framelift gui`` (after ``pip install "framelift-video[gui]"``). The page is
served from this computer and the videos never leave it.

The page is a thin layer over the library: a run is :meth:`VideoEnhancer.enhance` in a
background thread, its ``on_progress`` callback feeds the progress bar and the Stop
button sets its ``stop_event``, so what's done is kept as a valid MP4. Gradio is only
imported by :func:`build_app` and :func:`launch`; the helpers above them are plain Python.
"""

# No "from __future__ import annotations" here: Gradio reads the handlers' annotations
# (gr.Request) at run time, and gr only exists inside build_app.

import contextlib
import dataclasses
import logging
import math
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from .errors import FrameliftError, MissingDependencyError
from .filters import PROFILES
from .models import MODELS
from .options import DEFAULT_MODEL, EnhanceOptions
from .planning import Progress

log = logging.getLogger(__name__)

INSTALL_GRADIO = 'pip install "framelift-video[gui]"'

DEVICE_CHOICES = [
    ("Automatic (best available)", "auto"),
    ("NVIDIA GPU (CUDA)", "cuda"),
    ("Intel GPU (OpenVINO)", "igpu"),
    ("CPU only", "cpu"),
]

STAGES = {"loading": "Loading the models", "frames": "Processing", "finishing": "Finishing the MP4"}


def import_gradio():
    """Gradio, or a clear message saying how to install it."""
    try:
        import gradio as gr
    except ImportError:
        raise MissingDependencyError(
            f"The visual version needs Gradio: {INSTALL_GRADIO} (Python 3.10 or newer)."
        ) from None
    return gr


# -- plain helpers (no Gradio) -------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """What the page's controls say, before it becomes :class:`EnhanceOptions`."""

    model: str = DEFAULT_MODEL
    ai_strength: float = 0.45
    keep_size: bool = False
    scale: float = 1.5
    profile: str = "none"
    interpolate: bool = False
    output_fps: float | None = None
    device: str = "auto"
    half: bool = True
    crf: int = 20
    start: str = ""
    end: str = ""

    def to_options(self, video_path: str | Path | None = None) -> EnhanceOptions:
        """The options for a run; ``start``/``end`` (e.g. ``"1:30"``) become frames."""
        start_frame, max_frames = 1, None
        if (self.start or "").strip() or (self.end or "").strip():
            if video_path is None:
                raise FrameliftError("Choose a video first.")
            start_frame, max_frames = frames_between(video_path, self.start, self.end)
        options = EnhanceOptions(
            model=self.model,
            ai_strength=float(self.ai_strength),
            ai=float(self.ai_strength) > 0,
            same_resolution=bool(self.keep_size),
            scale=float(self.scale),
            profile=self.profile,
            interpolate=bool(self.interpolate),
            output_fps=float(self.output_fps) if self.output_fps else None,
            device=self.device,
            half=bool(self.half),
            crf=int(self.crf),
            start_frame=start_frame,
            max_frames=max_frames,
        )
        options.validate()
        return options


def frames_between(video_path: str | Path, start: str, end: str) -> tuple[int, int]:
    """``start_frame`` and ``max_frames`` for the part between two times (empty = that
    end of the video), with the frames that are on screen at those times."""
    from .trim import parse_time, range_from_times
    from .video import probe_video

    video = probe_video(video_path)
    first = parse_time(start) if (start or "").strip() else None
    last = parse_time(end) if (end or "").strip() else None
    frames = range_from_times(video, first, last)
    return frames.start, frames.end - frames.start + 1


def describe_progress(progress: Progress) -> str:
    """One line for the status area."""
    if progress.stage != "frames":
        return f"{STAGES.get(progress.stage, progress.stage)}…"
    text = (f"{STAGES['frames']}: {progress.frames_done:,} of {progress.frames_total:,} "
            f"frames ({progress.fraction:.0%})")  # fmt: skip
    if progress.eta_seconds is not None and progress.frames_done < progress.frames_total:
        text += f", about {format_duration(progress.eta_seconds)} left"
    return text


def format_duration(seconds: float) -> str:
    """``75`` → ``"1m15s"``, ``3700`` → ``"1h01m"``."""
    seconds = max(0, round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def output_path_for(video_path: str | Path, folder: str | Path) -> Path:
    """``concert.mkv`` → ``<folder>/concert_framelift.mp4``, never replacing a file."""
    folder = Path(folder)
    stem = Path(video_path).stem
    path = folder / f"{stem}_framelift.mp4"
    number = 2
    while path.exists():
        path = folder / f"{stem}_framelift_{number}.mp4"
        number += 1
    return path


def describe_video(video_path: str | Path | None) -> tuple[str, float]:
    """A line about the chosen video, and its duration in seconds."""
    if not video_path:
        return "", 0.0
    from .video import probe_video

    try:
        video = probe_video(video_path)
    except FrameliftError as exc:
        return f"⚠️ {exc}", 0.0
    return (
        f"{video.width}×{video.height} · {video.fps:.3f} fps · "
        f"{format_duration(video.duration_seconds)} ({video.frame_count:,} frames)",
        video.duration_seconds,
    )


@dataclass
class Job:
    """One video being processed. It belongs to the server, not to a browser tab: closing
    or reloading the page, or a dropped connection, doesn't stop it."""

    video: Path
    options: EnhanceOptions
    stop: threading.Event = dataclasses.field(default_factory=threading.Event)
    done: threading.Event = dataclasses.field(default_factory=threading.Event)
    progress: Progress | None = None
    result: object = None  # EnhanceResult, once done
    error: BaseException | None = None
    updates: int = 0  # bumped on every change, so watchers know when to refresh


class Runner:
    """Runs one video at a time in the background, and previews when nothing is running.

    The models stay loaded between runs and previews with the same settings, since one
    :class:`VideoEnhancer` is reused (it reloads only what changed).
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.job: Job | None = None
        self._enhancer = None
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return self.job is not None and not self.job.done.is_set()

    def enhancer(self, options: EnhanceOptions):
        from .pipeline import VideoEnhancer

        if self._enhancer is None:
            self._enhancer = VideoEnhancer(options)
        self._enhancer.options = options
        return self._enhancer

    def start(self, video_path, options: EnhanceOptions) -> Job:
        """Start processing in a background thread; refused while another job runs."""
        with self._lock:
            if self.busy:
                raise FrameliftError(
                    "A video is already being processed. Stop it or wait for it to finish."
                )
            self.output_dir.mkdir(parents=True, exist_ok=True)
            job = self.job = Job(Path(video_path), options)
        output = output_path_for(video_path, self.output_dir)

        def on_progress(progress: Progress) -> None:
            job.progress = progress
            job.updates += 1

        def work() -> None:
            try:
                job.result = self.enhancer(options).enhance(
                    video_path, output, stop_event=job.stop, show_progress=False,
                    on_progress=on_progress,
                )  # fmt: skip
                state = "stopped early" if job.result.interrupted else "done"
                log.info("Visual version: %s, saved %s", state, job.result.output_path)
            except BaseException as exc:  # shown on the page by whoever is watching
                job.error = exc
                log.error("Visual version: the run failed: %s", exc)
            finally:
                job.updates += 1
                job.done.set()

        threading.Thread(target=work, name="framelift-gui-job", daemon=True).start()
        return job

    @staticmethod
    def watch(job: Job, every: float = 0.5):
        """Yield ``job`` whenever it changes, until it's done (the last yield). Stopping
        the iteration doesn't affect the job."""
        seen = -1
        while True:
            finished = job.done.wait(every)
            if job.updates != seen or finished:
                seen = job.updates
                yield job
            if finished:
                return

    def stop(self) -> bool:
        """Ask the running job to stop after the current frame, keeping what's done."""
        if not self.busy:
            return False
        self.job.stop.set()
        return True

    def preview(self, video_path, options: EnhanceOptions, at_seconds: float):
        """The frame at ``at_seconds``: the original enlarged normally, and framelift's
        result, both RGB and the same size."""
        import cv2

        from .video import probe_video, read_frames

        if self.busy:
            raise FrameliftError("Preview is available once the current video is done.")
        video = probe_video(video_path)
        number = min(video.frame_count, max(1, int(at_seconds * video.fps) + 1))
        single = dataclasses.replace(
            options, start_frame=number, max_frames=1, interpolate=False, output_fps=None,
            audio=False, crf=10,
        )  # fmt: skip
        with tempfile.TemporaryDirectory(prefix="framelift-preview-") as folder, _quiet():
            output = Path(folder) / "frame.mp4"
            self.enhancer(single).enhance(video_path, output, show_progress=False)
            capture = cv2.VideoCapture(str(output))
            ok, enhanced = capture.read()
            capture.release()
        if not ok:
            raise FrameliftError("Couldn't read the preview frame back.")
        frames = read_frames(video_path, [number])
        if not frames:
            raise FrameliftError(f"Couldn't read frame {number} of the video.")
        height, width = enhanced.shape[:2]
        original = cv2.resize(frames[0][1], (width, height), interpolation=cv2.INTER_LANCZOS4)
        return original[..., ::-1].copy(), enhanced[..., ::-1].copy()


@contextlib.contextmanager
def _quiet():
    """Keep a preview's plan table and messages out of the terminal (warnings still show)."""
    logger = logging.getLogger("framelift")
    level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.setLevel(level)


# -- the page ------------------------------------------------------------------------------


def build_app(output_dir: str | Path = "framelift_outputs"):
    """The Gradio page (not started: see :func:`launch`)."""
    gr = import_gradio()
    from . import __version__

    runner = Runner(output_dir)
    model_choices = [(f"{name} — {spec.summary}", name) for name, spec in MODELS.items()]
    profile_choices = [
        (f"{name} — {profile.description}", name) for name, profile in PROFILES.items()
    ]

    with gr.Blocks(title="framelift") as app:
        gr.Markdown(
            f"# framelift <small>{__version__}</small>\n"
            "Upscale, restore and smooth videos. Everything runs on this computer."
        )
        with gr.Row():
            with gr.Column(scale=5):
                video = gr.Video(label="Video", sources=["upload"], format=None)
                info = gr.Markdown()
                with gr.Accordion("Only part of the video", open=False), gr.Row():
                    start = gr.Textbox(label="From", placeholder="e.g. 1:30 (empty: start)")
                    end = gr.Textbox(label="To", placeholder="e.g. 2:45 (empty: end)")
            with gr.Column(scale=4):
                model = gr.Dropdown(
                    model_choices, value=DEFAULT_MODEL, label="AI model",
                    info="The default is fast and made for animation; realesrgan-x4plus "
                         "suits real footage, but is slower.",
                )  # fmt: skip
                ai_strength = gr.Slider(0, 1, value=0.45, step=0.05, label="AI strength",
                                        info="How much of the AI result to keep. 0 = no AI "
                                             "(fast: only resize and frame rate).")  # fmt: skip
                with gr.Row():
                    scale = gr.Slider(1, 4, value=1.5, step=0.25, label="Enlarge by")
                    keep_size = gr.Checkbox(label="Keep the original size")
                profile = gr.Dropdown(profile_choices, value="none", label="Pre-cleaning")
                with gr.Row():
                    interpolate = gr.Checkbox(label="Smoother motion (RIFE)",
                                              info="New in-between frames")  # fmt: skip
                    output_fps = gr.Number(label="Frame rate", value=None, minimum=1,
                                           info="Empty: double it")  # fmt: skip
                with gr.Accordion("Hardware and quality", open=False):
                    device = gr.Dropdown(DEVICE_CHOICES, value="auto", label="Run the AI on")
                    half = gr.Checkbox(value=True, label="FP16 on GPUs (faster)")
                    crf = gr.Slider(12, 30, value=20, step=1, label="Video quality (CRF)",
                                    info="Lower is better and bigger.")  # fmt: skip

        with gr.Accordion("Preview one frame", open=True):
            with gr.Row():
                preview_at = gr.Slider(0, 1, value=0, step=0.1, label="Frame at (seconds)")
                preview_button = gr.Button("Preview", scale=0)
            # PNG, not the default WebP: the comparison shouldn't add compression of its own.
            comparison = gr.ImageSlider(label="Original, enlarged normally  |  framelift",
                                        type="numpy", format="png", max_height=640)  # fmt: skip

        with gr.Row():
            run_button = gr.Button("Enhance video", variant="primary")
            stop_button = gr.Button("Stop and keep what's done", variant="stop")
            show_button = gr.Button("Show progress", variant="secondary")
        status = gr.Markdown()
        result = gr.Video(label="Result", interactive=False)

        settings_inputs = [model, ai_strength, keep_size, scale, profile, interpolate,
                           output_fps, device, half, crf, start, end]  # fmt: skip

        def settings_from(*values) -> Settings:
            return Settings(*values)

        # Gradio hands gr.Request only to a named parameter (not after *args), so the run
        # spells out its inputs.

        def on_video(path):
            text, duration = describe_video(path)
            return text, gr.Slider(maximum=max(duration, 0.1), value=round(duration / 2, 1))

        def on_preview(path, at, *values):
            if not path:
                raise gr.Error("Choose a video first.")
            try:
                options = settings_from(*values).to_options(path)
                return runner.preview(path, options, float(at))
            except FrameliftError as exc:
                raise gr.Error(str(exc)) from None

        def follow(job: Job, progress):
            """Show a job's progress on this page until it's done."""
            for current in runner.watch(job):
                if current.done.is_set():
                    break
                if current.progress is not None:
                    line = describe_progress(current.progress)
                    progress(current.progress.fraction, desc=line)
                    yield line, gr.skip()
            if job.error is not None:
                message = (
                    str(job.error) if isinstance(job.error, FrameliftError) else repr(job.error)
                )
                raise gr.Error(message)
            yield _summary(job.result, job.video), str(job.result.output_path)

        def on_run(path, model, ai_strength, keep_size, scale, profile, interpolate,
                   output_fps, device, half, crf, start, end,
                   progress=gr.Progress()):  # noqa: B008  # fmt: skip
            if not path:
                raise gr.Error("Choose a video first.")
            settings = Settings(model, ai_strength, keep_size, scale, profile, interpolate,
                                output_fps, device, half, crf, start, end)  # fmt: skip
            try:
                job = runner.start(path, settings.to_options(path))
            except FrameliftError as exc:
                raise gr.Error(str(exc)) from None
            yield from follow(job, progress)

        def on_show(progress=gr.Progress()):  # noqa: B008
            """Pick up the current (or last) job, e.g. after a reload or a lost connection."""
            job = runner.job
            if job is None:
                yield "No video has been processed yet.", gr.skip()
                return
            yield from follow(job, progress)

        def on_load(progress=gr.Progress()):  # noqa: B008
            if runner.busy:
                yield from follow(runner.job, progress)
            else:
                yield gr.skip(), gr.skip()

        def on_stop():
            if runner.stop():
                return "Stopping after the current frame; what's done will be kept…"
            return "Nothing is being processed."

        video.change(on_video, video, [info, preview_at])
        preview_button.click(on_preview, [video, preview_at, *settings_inputs], comparison,
                             concurrency_id="gpu", concurrency_limit=1)  # fmt: skip
        run_button.click(on_run, [video, *settings_inputs], [status, result],
                         concurrency_id="gpu", concurrency_limit=1)  # fmt: skip
        stop_button.click(on_stop, None, status, concurrency_limit=None)
        show_button.click(on_show, None, [status, result], concurrency_limit=None)
        # Opening (or reloading) the page shows the job that's running, if there is one.
        app.load(on_load, None, [status, result], concurrency_limit=None)
    return app


def _summary(result, video_path) -> str:
    done = "Stopped early, saved" if result.interrupted else "Done, saved"
    text = (f"**{done}:** `{result.output_path}`  \n"
            f"{result.frames_processed:,} frames processed, {result.frames_written:,} written, "
            f"in {format_duration(result.elapsed_seconds)}.")  # fmt: skip
    if result.resume_from is not None:
        start = resume_time(result.resume_from, video_path)
        text += f"  \nTo continue later, set **From** to `{start}` and run again; "
        text += "then join the two files with `framelift join`."
    return text


def resume_time(frame: int, video_path: str | Path) -> str:
    """The time (in seconds) that **From** needs to start exactly at ``frame``: rounded up
    to the millisecond, so it lands on that frame and not the one before."""
    from .video import probe_video

    seconds = (frame - 1) / probe_video(video_path).fps
    return f"{math.ceil(seconds * 1000) / 1000:.3f}"


class _DroppedConnections(logging.Filter):
    """Hide asyncio's tracebacks for connections the browser closed abruptly
    (``ConnectionResetError``, WinError 10054 on Windows): they're harmless noise."""

    def filter(self, record: logging.LogRecord) -> bool:
        error = record.exc_info[1] if record.exc_info else None
        return not isinstance(error, (ConnectionResetError, ConnectionAbortedError))


def launch(
    output_dir: str | Path = "framelift_outputs",
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    share: bool = False,
    open_browser: bool = True,
) -> None:
    """Build the page and serve it (blocks until stopped with Ctrl+C)."""
    gr = import_gradio()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        logging.getLogger("asyncio").addFilter(_DroppedConnections())
    app = build_app(output_dir)
    app.queue()
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=open_browser,
        allowed_paths=[str(output_dir)],
        theme=gr.themes.Soft(),
    )

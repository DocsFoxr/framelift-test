"""framelift gui: the helpers behind the page (plain Python), the runs it starts, and the
page itself (needs Gradio)."""

from pathlib import Path

import pytest

from framelift.gui import (
    Settings,
    describe_progress,
    format_duration,
    frames_between,
    output_path_for,
    resume_time,
)
from framelift.planning import Progress

from .conftest import needs_ffmpeg


@pytest.mark.parametrize(("seconds", "text"), [(0, "0s"), (42.4, "42s"), (75, "1m15s"),
                                               (3700, "1h01m")])  # fmt: skip
def test_format_duration(seconds, text):
    assert format_duration(seconds) == text


def test_describe_progress():
    assert describe_progress(Progress("loading", 0, 100, 0, 1.0)) == "Loading the models…"
    line = describe_progress(Progress("frames", 25, 100, 50, 10.0, eta_seconds=90))
    assert line == "Processing: 25 of 100 frames (25%), about 1m30s left"
    assert describe_progress(Progress("finishing", 100, 100, 200, 20.0)) == "Finishing the MP4…"


def test_output_names_never_replace_a_file(tmp_path):
    assert output_path_for("/videos/concert.mkv", tmp_path) == tmp_path / "concert_framelift.mp4"
    (tmp_path / "concert_framelift.mp4").write_bytes(b"done before")
    assert output_path_for("concert.mkv", tmp_path) == tmp_path / "concert_framelift_2.mp4"


def test_settings_become_options():
    options = Settings(ai_strength=0.8, keep_size=True, interpolate=True, output_fps=0).to_options()
    assert (options.ai_strength, options.ai, options.same_resolution) == (0.8, True, True)
    assert options.interpolate and options.output_fps is None  # 0 / empty: double the rate
    assert Settings(ai_strength=0).to_options().uses_ai is False


@needs_ffmpeg
def test_times_become_frames(make_video):
    clip = make_video(frames=48, fps=24)
    assert frames_between(clip, "0.5", "1.5") == (13, 24)
    assert frames_between(clip, "", "1") == (1, 24)
    options = Settings(start="1", end="").to_options(clip)
    assert (options.start_frame, options.max_frames) == (25, 24)


@needs_ffmpeg
@pytest.mark.parametrize("frame", [2, 13, 25, 47])
def test_the_resume_time_lands_on_the_frame(make_video, frame):
    clip = make_video(frames=48, fps=24)
    assert frames_between(clip, resume_time(frame, clip), "")[0] == frame


# -- runs (need PyTorch: the pipeline imports it) ----------------------------------------


@needs_ffmpeg
def test_a_job_reports_progress_then_the_result(make_video, tmp_path):
    pytest.importorskip("torch")
    from framelift.gui import Runner
    from framelift.planning import EnhanceResult

    runner = Runner(tmp_path / "out")
    job = runner.start(make_video(frames=20), Settings(ai_strength=0, keep_size=True).to_options())
    seen = [j.progress for j in runner.watch(job, every=0.05)]
    assert job.done.is_set() and job.error is None and not runner.busy
    assert any(isinstance(p, Progress) for p in seen)
    assert isinstance(job.result, EnhanceResult) and job.result.frames_processed == 20
    assert job.result.output_path.parent == tmp_path / "out" and job.result.output_path.exists()


@needs_ffmpeg
def test_a_job_outlives_the_page_that_started_it(make_video, tmp_path):
    """A closed tab or a dropped connection stops the watching, never the job."""
    pytest.importorskip("torch")
    from framelift.gui import Runner

    runner = Runner(tmp_path)
    job = runner.start(make_video(frames=60), Settings(ai_strength=0, keep_size=True).to_options())
    watching = runner.watch(job, every=0.01)
    next(watching)
    watching.close()  # what Gradio does when the browser goes away
    assert job.done.wait(60)
    assert not job.result.interrupted and job.result.frames_processed == 60


@needs_ffmpeg
def test_stop_keeps_what_is_done(make_video, tmp_path):
    pytest.importorskip("torch")
    from framelift.errors import FrameliftError
    from framelift.gui import Runner

    runner = Runner(tmp_path)
    options = Settings(ai_strength=0, keep_size=True).to_options()
    assert runner.stop() is False
    clip = make_video(frames=200)
    job = runner.start(clip, options)
    for current in runner.watch(job, every=0.01):
        if current.progress is not None and current.progress.frames_done >= 1:
            if not current.stop.is_set():  # running for sure: a second job is refused
                with pytest.raises(FrameliftError, match="already being processed"):
                    runner.start(clip, options)
            runner.stop()
    assert job.result.interrupted and 1 <= job.result.frames_processed < 200


@needs_ffmpeg
def test_preview_shows_the_same_frame_twice_at_the_output_size(make_video, tmp_path):
    pytest.importorskip("torch")
    from framelift.gui import Runner

    original, enhanced = Runner(tmp_path).preview(
        make_video(frames=24), Settings(ai_strength=0, scale=2).to_options(), at_seconds=0.5
    )
    assert original.shape == enhanced.shape == (96, 128, 3)


# -- the page (needs Gradio) ------------------------------------------------------------


def test_the_page_builds(tmp_path):
    pytest.importorskip("gradio")
    from framelift.gui import build_app

    app = build_app(tmp_path)
    assert app.title == "framelift"


def test_gui_command_passes_its_options(monkeypatch, tmp_path):
    import framelift.gui as gui
    from framelift.cli import main

    calls = []
    monkeypatch.setattr(gui, "launch", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert main(["gui", "--output-dir", str(tmp_path), "--port", "9000", "--no-browser"]) == 0
    ((args, kwargs),) = calls
    assert Path(args[0]) == tmp_path
    assert kwargs == {"host": "127.0.0.1", "port": 9000, "share": False, "open_browser": False}


def test_dropped_connections_are_not_reported():
    import logging

    from framelift.gui import _DroppedConnections

    def record(error):
        exc_info = (type(error), error, None) if error else None
        return logging.LogRecord("asyncio", logging.ERROR, "", 0, "x", None, exc_info)

    keep = _DroppedConnections().filter
    assert not keep(record(ConnectionResetError(10054, "reset by the remote host")))
    assert keep(record(ValueError("a real problem"))) and keep(record(None))

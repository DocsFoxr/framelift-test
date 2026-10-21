import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from framelift import FrameRange, InvalidOptionError, VideoInfo, probe_video, trim_video
from framelift.trim import default_trim_path, format_time, parse_time, range_from_times

from .conftest import needs_ffmpeg

VIDEO = VideoInfo(Path("clip.mp4"), fps=25.0, frame_count=250, width=64, height=48)


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("90", 90), ("90.5", 90.5), ("1:30", 90), ("01:02:03.25", 3723.25), (" 0:05 ", 5)],
)
def test_parse_time(text, seconds):
    assert parse_time(text) == pytest.approx(seconds)


@pytest.mark.parametrize("bad", ["", "abc", "1:2:3:4", "-5", "1,5"])
def test_parse_time_rejects_nonsense(bad):
    with pytest.raises(InvalidOptionError):
        parse_time(bad)


def test_format_time():
    assert format_time(83.5) == "1:23.500"
    assert format_time(3723) == "1:02:03.000"
    assert format_time(-1) == "0:00.000"


def test_frame_range_basics():
    frames = FrameRange(26, 75)
    assert frames.frames == 50
    assert frames.start_seconds(25) == 1.0 and frames.end_seconds(25) == 3.0
    assert frames.describe(25) == "0:01.000 → 0:03.000 (2.0 s, frames 26–75)"


@pytest.mark.parametrize("bad", [(0, 5), (10, 9)])
def test_frame_range_rejects_impossible_ranges(bad):
    with pytest.raises(InvalidOptionError):
        FrameRange(*bad)


def test_range_from_times_picks_the_frames_on_screen():
    assert range_from_times(VIDEO, 1.0, 3.0) == FrameRange(26, 75)
    assert range_from_times(VIDEO, 1.01, 2.99) == FrameRange(26, 75)
    assert range_from_times(VIDEO, None, None) == FrameRange(1, 250)
    assert range_from_times(VIDEO, 9.0, 60) == FrameRange(226, 250)  # clamped to the end


@pytest.mark.parametrize(("start", "end"), [(20.0, None), (3.0, 2.0), (3.0, 3.0)])
def test_range_from_times_rejects_empty_ranges(start, end):
    with pytest.raises(InvalidOptionError):
        range_from_times(VIDEO, start, end)


def test_default_trim_path():
    assert default_trim_path("/v/clip.mkv", FrameRange(301, 2565)) == Path(
        "/v/clip_trim_301-2565.mp4"
    )


def _frame_hashes(path):
    capture = cv2.VideoCapture(str(path))
    hashes = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        hashes.append(hashlib.md5(image.tobytes()).hexdigest())
    capture.release()
    return hashes


@pytest.fixture
def h264_with_audio(tmp_path):
    """A 2 s, 30 fps H.264 clip with keyframes far apart and an AAC beep track."""
    path = tmp_path / "source.mp4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=30",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "2",
         "-c:v", "libx264", "-g", "250", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
        check=True,
    )  # fmt: skip
    return path


@needs_ffmpeg
def test_trim_is_frame_exact_and_lossless(h264_with_audio, tmp_path):
    output = tmp_path / "cut.mp4"
    result = trim_video(h264_with_audio, output, FrameRange(17, 41), show_progress=False)

    assert result.path == output.resolve() and output.exists()
    assert result.frames == 25 and result.audio
    assert probe_video(output).frame_count == 25
    # Same pixels as frames 17–41 of the original, even though they're between keyframes.
    assert _frame_hashes(output) == _frame_hashes(h264_with_audio)[16:41]


@needs_ffmpeg
def test_trim_keeps_the_audio_in_sync(h264_with_audio, tmp_path):
    output = tmp_path / "cut.mp4"
    trim_video(h264_with_audio, output, FrameRange(31, 45), show_progress=False)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,start_time,"
         "duration", "-of", "json", str(output)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    streams = {s["codec_type"]: s for s in json.loads(probe)["streams"]}
    assert streams["audio"]["codec_name"] == "alac"
    # Video and audio start together (the video a hair later, by the seek margin) …
    assert float(streams["video"]["start_time"]) - float(streams["audio"]["start_time"]) < 0.002
    # … and the audio lasts as long as the 15 frames.
    assert float(streams["audio"]["duration"]) == pytest.approx(15 / 30, abs=0.03)


@needs_ffmpeg
def test_trim_without_audio_and_range_past_the_end(make_video, tmp_path):
    result = trim_video(make_video(frames=12), tmp_path / "c.mp4", FrameRange(10, 99),
                        show_progress=False)  # fmt: skip
    assert result.frame_range == FrameRange(10, 12)
    assert result.frames == 3 and not result.audio


@needs_ffmpeg
def test_trim_refuses_to_overwrite_its_input(make_video):
    clip = make_video(frames=4)
    with pytest.raises(InvalidOptionError):
        trim_video(clip, clip, FrameRange(1, 2), show_progress=False)


@needs_ffmpeg
def test_trim_falls_back_to_decoding_when_seeking_misses(h264_with_audio, tmp_path):
    """MPEG-TS with a single keyframe: seeking lands mid-GOP, so the cut decodes from the
    start instead, and is still exact."""
    stream = tmp_path / "capture.ts"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(h264_with_audio),
                    "-c", "copy", "-f", "mpegts", str(stream)], check=True)  # fmt: skip
    result = trim_video(stream, tmp_path / "cut.mp4", FrameRange(31, 45), show_progress=False)
    assert result.frames == 15
    assert _frame_hashes(result.path) == _frame_hashes(stream)[30:45]


@needs_ffmpeg
def test_trim_command_by_time(h264_with_audio, capsys):
    from framelift.cli import main

    assert main(["trim", "-i", str(h264_with_audio), "--start", "0:00.5", "--end", "1"]) == 0
    cut = h264_with_audio.with_name("source_trim_16-30.mp4")
    assert probe_video(cut).frame_count == 15
    out = capsys.readouterr().out
    assert "framelift tune -i" in out and str(cut) in out


@needs_ffmpeg
def test_trim_command_rejects_a_bad_time(h264_with_audio):
    from framelift.cli import main

    assert main(["trim", "-i", str(h264_with_audio), "--start", "soon"]) == 1


# -- playable cuts -----------------------------------------------------------------------


def _streams(path):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,profile,pix_fmt,width,height,start_time,duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return {s["codec_type"]: s for s in json.loads(probe)["streams"]}


def _frames(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        frames.append(image.astype(float))
    capture.release()
    return frames


@needs_ffmpeg
def test_playable_cut_is_frame_exact_and_plays_everywhere(h264_with_audio, tmp_path):
    output = tmp_path / "cut.mp4"
    result = trim_video(h264_with_audio, output, FrameRange(17, 41), show_progress=False,
                        playable=True)  # fmt: skip
    assert result.frames == 25 and result.audio

    streams = _streams(output)
    # Ordinary H.264 (not the lossless 4:4:4 profile GPUs can't decode) and AAC.
    assert streams["video"]["profile"] == "High" and streams["video"]["pix_fmt"] == "yuv420p"
    assert streams["audio"]["codec_name"] == "aac"
    # The same frames as 17–41 of the original: each is closest to its counterpart (a
    # tiny, busy test pattern at CRF 18 is off by a few levels; real video, by less).
    cut, original = _frames(output), _frames(h264_with_audio)
    for index, frame in enumerate(cut):
        errors = [np.abs(frame - other).mean() for other in original]
        assert int(np.argmin(errors)) == 16 + index and errors[16 + index] < 6


@needs_ffmpeg
def test_playable_cut_keeps_the_audio_in_sync(h264_with_audio, tmp_path):
    output = tmp_path / "cut.mp4"
    trim_video(h264_with_audio, output, FrameRange(31, 45), show_progress=False, playable=True)
    streams = _streams(output)
    assert abs(float(streams["video"]["start_time"]) - float(streams["audio"]["start_time"])) < 0.03
    assert float(streams["audio"]["duration"]) == pytest.approx(15 / 30, abs=0.03)


@needs_ffmpeg
def test_playable_cut_of_odd_sized_video(tmp_path):
    """yuv420p needs even dimensions: an odd-sized source loses its last row/column."""
    source = tmp_path / "odd.mp4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=95x63:rate=24",
         "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv444p", str(source)],
        check=True,
    )  # fmt: skip
    output = tmp_path / "cut.mp4"
    trim_video(source, output, FrameRange(1, 12), show_progress=False, playable=True)
    video = _streams(output)["video"]
    assert (video["width"], video["height"], video["pix_fmt"]) == (94, 62, "yuv420p")


def test_playable_cuts_get_their_own_name():
    frames = FrameRange(301, 2565)
    assert default_trim_path("clip.mkv", frames) == Path("clip_trim_301-2565.mp4")
    assert default_trim_path("clip.mkv", frames, playable=True) == Path(
        "clip_trim_301-2565_playable.mp4"
    )


@needs_ffmpeg
def test_trim_command_playable(h264_with_audio, capsys):
    from framelift.cli import main

    assert main(["trim", "-i", str(h264_with_audio), "--start", "0.5", "--end", "1",
                 "--playable", "-q"]) == 0  # fmt: skip
    made = list(h264_with_audio.parent.glob("source_trim_*_playable.mp4"))
    assert len(made) == 1 and _streams(made[0])["audio"]["codec_name"] == "aac"
    assert "watch or share" in capsys.readouterr().out

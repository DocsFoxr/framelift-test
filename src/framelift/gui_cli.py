"""``framelift gui``: open framelift's visual version in the browser."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from .console import setup_console
from .errors import FrameliftError

log = logging.getLogger("framelift.cli")

EXAMPLES = """\
examples:
  # Open the page in your browser (needs: pip install "framelift-video[gui]")
  framelift gui

  # Save the results somewhere else, and don't open the browser by itself
  framelift gui --output-dir D:/videos/enhanced --no-browser

The page is served from this computer only (127.0.0.1): the videos never leave it.
Press Ctrl+C in this terminal to close it.
"""


def build_gui_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="framelift gui",
        description="Open framelift's visual version: enhance videos from a page in your browser.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", default="framelift_outputs", metavar="FOLDER",
        help="Where the enhanced videos are saved. (default: %(default)s, in the current folder)",
    )  # fmt: skip
    parser.add_argument(
        "--port", type=int, default=None,
        help="Port for the page. (default: 7860, or the next free one)",
    )  # fmt: skip
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Address to serve on. 0.0.0.0 lets other devices on your network open it. "
             "(default: %(default)s, this computer only)",
    )  # fmt: skip
    parser.add_argument(
        "--share", action="store_true",
        help="Also create a temporary public link (through Gradio's servers), to open the "
             "page from anywhere. Anyone with the link can use it while it runs.",
    )  # fmt: skip
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't open the browser automatically."
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Show more detail.")
    parser.add_argument(
        "-q", "--quiet", action="count", default=0, help="Only show warnings and errors."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_gui_parser().parse_args(argv)
    setup_console(args.verbose - args.quiet)
    from .gui import launch

    try:
        launch(args.output_dir, host=args.host, port=args.port, share=args.share,
               open_browser=not args.no_browser)  # fmt: skip
    except FrameliftError as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        pass
    return 0

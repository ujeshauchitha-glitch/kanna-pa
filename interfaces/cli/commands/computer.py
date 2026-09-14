"""`kanna computer ...` — direct computer-control commands.

Like `document generate`, this talks to the underlying capability
(`tools.computer.get_computer_agent()`) directly rather than through
`ToolRegistry.invoke()`'s permission gate — a CLI invocation is already
an explicit, direct user command, the same trust level a human clicking
the mouse themselves would have. The REVIEW-gating on `computer_click`/
`computer_type_text`/etc. matters for the agent loop, where a plan step
was not something the user typed by hand this instant.
"""
from __future__ import annotations

import argparse
import sys

from core.bootstrap import Kanna
from core.errors import CapabilityUnavailable
from tools.computer import get_computer_agent
from tools.computer.base import Point


def register(subparsers: argparse._SubParsersAction) -> None:
    computer_parser = subparsers.add_parser("computer", help="Direct computer control (mouse/keyboard/screen)")
    computer_sub = computer_parser.add_subparsers(dest="computer_command", required=True)

    shot_p = computer_sub.add_parser("screenshot", help="Save a screenshot as PNG")
    shot_p.add_argument("path", help="Destination path, within the sandbox")
    shot_p.set_defaults(func=_cmd_screenshot)

    inspect_p = computer_sub.add_parser("inspect", help="Show screen dimensions and the active window")
    inspect_p.set_defaults(func=_cmd_inspect)

    click_p = computer_sub.add_parser("click", help="Click at (x, y)")
    click_p.add_argument("x", type=int)
    click_p.add_argument("y", type=int)
    click_p.add_argument("--button", default="left", choices=["left", "middle", "right"])
    click_p.set_defaults(func=_cmd_click)

    move_p = computer_sub.add_parser("move", help="Move the mouse to (x, y)")
    move_p.add_argument("x", type=int)
    move_p.add_argument("y", type=int)
    move_p.set_defaults(func=_cmd_move)

    type_p = computer_sub.add_parser("type", help="Type text at the current keyboard focus")
    type_p.add_argument("text")
    type_p.set_defaults(func=_cmd_type)

    key_p = computer_sub.add_parser("key", help="Press a key/combo (xdotool syntax, e.g. 'ctrl+c')")
    key_p.add_argument("key")
    key_p.set_defaults(func=_cmd_key)

    open_p = computer_sub.add_parser("open", help="Launch an application")
    open_p.add_argument("name")
    open_p.set_defaults(func=_cmd_open)

    close_p = computer_sub.add_parser("close", help="Close an application's window(s)")
    close_p.add_argument("name")
    close_p.set_defaults(func=_cmd_close)

    clip_p = computer_sub.add_parser("clipboard", help="Read or write the clipboard")
    clip_sub = clip_p.add_subparsers(dest="clipboard_command", required=True)
    clip_get_p = clip_sub.add_parser("get")
    clip_get_p.set_defaults(func=_cmd_clipboard_get)
    clip_set_p = clip_sub.add_parser("set")
    clip_set_p.add_argument("text")
    clip_set_p.set_defaults(func=_cmd_clipboard_set)


def _cmd_screenshot(args: argparse.Namespace, kanna: Kanna) -> int:
    try:
        resolved = kanna.sandbox.resolve(args.path)
        image_bytes = get_computer_agent().screenshot()
    except CapabilityUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_bytes(image_bytes)
    print(f"Saved screenshot to {resolved} ({len(image_bytes)} bytes).")
    return 0


def _cmd_inspect(args: argparse.Namespace, kanna: Kanna) -> int:
    try:
        info = get_computer_agent().inspect_screen()
    except CapabilityUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{info['width']}x{info['height']}, active window: {info.get('active_window') or '(none)'}")
    return 0


def _cmd_click(args: argparse.Namespace, kanna: Kanna) -> int:
    return _run(lambda: get_computer_agent().click(Point(args.x, args.y), button=args.button),
                f"Clicked ({args.x}, {args.y}).")


def _cmd_move(args: argparse.Namespace, kanna: Kanna) -> int:
    return _run(lambda: get_computer_agent().move_mouse(Point(args.x, args.y)),
                f"Moved mouse to ({args.x}, {args.y}).")


def _cmd_type(args: argparse.Namespace, kanna: Kanna) -> int:
    return _run(lambda: get_computer_agent().type_text(args.text), "Typed.")


def _cmd_key(args: argparse.Namespace, kanna: Kanna) -> int:
    return _run(lambda: get_computer_agent().key_press(args.key), f"Pressed {args.key}.")


def _cmd_open(args: argparse.Namespace, kanna: Kanna) -> int:
    return _run(lambda: get_computer_agent().open_application(args.name), f"Opened {args.name}.")


def _cmd_close(args: argparse.Namespace, kanna: Kanna) -> int:
    return _run(lambda: get_computer_agent().close_application(args.name), f"Closed {args.name}.")


def _cmd_clipboard_get(args: argparse.Namespace, kanna: Kanna) -> int:
    try:
        print(get_computer_agent().get_clipboard())
    except CapabilityUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_clipboard_set(args: argparse.Namespace, kanna: Kanna) -> int:
    return _run(lambda: get_computer_agent().set_clipboard(args.text), "Clipboard set.")


def _run(action, success_message: str) -> int:
    try:
        action()
    except CapabilityUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(success_message)
    return 0

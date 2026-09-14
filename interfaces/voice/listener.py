"""Voice interface: listen to microphone, recognize speech, send to Kanna.

Uses `sounddevice` for audio capture and `speech_recognition` for
Google's free speech-to-text API. No API key needed for the Google
web API (rate-limited but fine for personal use).

Usage:
    python main.py voice              # one-shot: listen → process → exit
    python main.py voice --loop       # continuous: keep listening
"""
from __future__ import annotations

import queue
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import speech_recognition as sr

from core.errors import CapabilityUnavailable


def _check_mic() -> None:
    """Verify a microphone is available."""
    try:
        devices = sd.query_devices()
        input_devices = [d for d in devices if d["max_input_channels"] > 0]
        if not input_devices:
            raise CapabilityUnavailable("no microphone found; voice interface requires an audio input device")
    except Exception as exc:
        raise CapabilityUnavailable(f"could not query audio devices: {exc}") from exc


def record_audio(duration: float = 5.0, sample_rate: int = 16000) -> bytes:
    """Record audio from the microphone and return WAV bytes.

    Records for `duration` seconds. Returns raw WAV file bytes
    suitable for passing to speech_recognition.
    """
    _check_mic()
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                        channels=1, dtype="int16")
    sd.wait()

    # Write to a temporary WAV file, then read it back as bytes.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with wave.open(str(tmp_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


def transcribe(wav_bytes: bytes, language: str = "en-US") -> str | None:
    """Transcribe WAV audio bytes to text using Google's free STT API.

    Returns the recognized text, or None if nothing was understood.
    """
    recognizer = sr.Recognizer()
    # Write WAV bytes to a temp file for speech_recognition
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        tmp_path.write_bytes(wav_bytes)
        with sr.AudioFile(str(tmp_path)) as source:
            audio = recognizer.record(source)
    finally:
        tmp_path.unlink(missing_ok=True)

    try:
        return recognizer.recognize_google(audio, language=language)
    except sr.UnknownValueError:
        return None
    except sr.RequestError as exc:
        raise CapabilityUnavailable(f"speech recognition API error: {exc}") from exc


def listen_once(duration: float = 5.0, language: str = "en-US") -> str | None:
    """Record one utterance and transcribe it. Returns text or None."""
    print(f"[listening for {duration}s...]", file=sys.stderr)
    wav = record_audio(duration=duration)
    text = transcribe(wav, language=language)
    if text:
        print(f"[heard: {text}]", file=sys.stderr)
    else:
        print("[no speech detected]", file=sys.stderr)
    return text


def listen_loop(callback, *, duration: float = 5.0, pause: float = 0.5,
                language: str = "en-US") -> None:
    """Continuously listen and pass each utterance to `callback(text)`.

    Press Ctrl+C to stop.
    """
    _check_mic()
    print("Voice loop active — speak a command. Ctrl+C to stop.", file=sys.stderr)
    try:
        while True:
            text = listen_once(duration=duration, language=language)
            if text:
                callback(text)
            time.sleep(pause)
    except KeyboardInterrupt:
        print("\nVoice loop stopped.", file=sys.stderr)

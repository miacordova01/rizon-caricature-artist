"""
utils/audio.py
Text-to-speech prompts using the macOS built-in 'say' command.
No external dependencies required.
"""

import logging
import subprocess
import time

log = logging.getLogger(__name__)

# macOS voices — change to any voice from: say -v '?'
_VOICE = "Samantha"


def say(text: str) -> None:
    """Speak *text* and block until finished."""
    log.debug("TTS: %r", text)
    subprocess.run(["say", "-v", _VOICE, text], check=False)


def say_async(text: str) -> subprocess.Popen:
    """Speak *text* without blocking. Returns the Popen handle."""
    log.debug("TTS (async): %r", text)
    return subprocess.Popen(["say", "-v", _VOICE, text])


def countdown(n: int = 3, gap: float = 0.9) -> None:
    """
    Speak a numeric countdown from *n* down to 1, blocking throughout.
    *gap* is the pause in seconds between each number.
    """
    for i in range(n, 0, -1):
        say(str(i))
        if i > 1:
            time.sleep(gap)

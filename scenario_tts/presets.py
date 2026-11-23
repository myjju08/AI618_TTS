"""Human-authored, inspectable speaking situations; no prompt-writing LLM."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    instruction: str
    background: str
    snr_db: float


PRESETS = {
    "documentary": Preset(
        "documentary",
        "Documentary",
        "Deliver the sentence as a documentary narrator addressing an audience. "
        "Use measured pacing, clear phrasing, and restrained emotion. "
        "Finish the sentence calmly.",
        "none",
        20.0,
    ),
    "cafe": Preset(
        "cafe",
        "Café conversation",
        "You are talking naturally to a friend sitting across a café table. "
        "Sound relaxed and conversational, with comfortable pacing and informal "
        "intonation. Keep the words clear without projecting like an announcer.",
        "cafe",
        20.0,
    ),
    "station": Preset(
        "station",
        "Station announcement",
        "You are making a station announcement to passengers. "
        "Use a clear, composed public-address delivery with deliberate phrasing "
        "and precise articulation. Emphasize important information, especially "
        "times, locations, and instructions when they occur in the sentence.",
        "station",
        10.0,
    ),
    "urgent": Preset(
        "urgent",
        "Urgent scene",
        "You are telling a friend that they need to hurry. "
        "Sound urgent but controlled. Use brisk pacing and clear articulation. "
        "Emphasize time information when present. Convey urgency without screaming.",
        "street",
        10.0,
    ),
    "reassuring": Preset(
        "reassuring",
        "Reassuring scene",
        "You are reassuring someone who feels anxious. "
        "Use a warm, gentle tone and steady, unhurried pacing. "
        "Sound confident and supportive, with soft phrase endings.",
        "room",
        20.0,
    ),
    "excited": Preset(
        "excited",
        "Excited scene",
        "You are sharing good news with a friend. "
        "Sound cheerful and excited, with bright intonation, lively pacing, "
        "and expressive changes in pitch. Keep every word understandable.",
        "none",
        20.0,
    ),
}

ACCENTS = {
    "american": "Use a General American accent.",
    "british": "Use a contemporary Southern British English accent.",
}

BACKGROUNDS = ("none", "cafe", "station", "street", "room", "crowd")
NOISE_LEVELS = {"weak": 20.0, "medium": 10.0, "strong": 5.0}


def build_instruction(preset: str, accent: str, tone: str | None = None) -> str:
    """Compose accent and situation without modifying the separately supplied text.

    A non-empty ``tone`` replaces the preset's delivery instruction. Accent is a
    request to the model, not a guarantee of an audibly distinct accent.
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset!r}")
    if accent not in ACCENTS:
        raise ValueError(f"Unknown accent: {accent!r}")
    if tone is not None and not isinstance(tone, str):
        raise ValueError("Tone must be text or None.")
    delivery = tone.strip() if tone and tone.strip() else PRESETS[preset].instruction
    return "\n".join(
        (
            ACCENTS[accent],
            delivery,
            "Speak exactly the supplied text. Do not add, omit, or paraphrase words.",
        )
    )

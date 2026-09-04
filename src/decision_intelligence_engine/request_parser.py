"""Deterministic natural-language request parsing for the CIFAR-10 assistant.

The parser decides whether a free-form user request can be served by this
application's single capability: CIFAR-10 image classification. It performs no
LLM calls and never guesses: when the intent cannot be safely determined the
request is marked ambiguous so the caller can ask for clarification.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

from .model_inference import CANONICAL_CIFAR10_CLASSES


class Intent(Enum):
    """Supported request intents for the natural-language interface."""

    CLASSIFY_IMAGE = "classify_image"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class ParsedRequest:
    """Structured result of parsing a free-form user request."""

    raw_text: str
    intent: Intent
    requires_image: bool
    target_class: str | None = None
    normalized_question: str | None = None
    validation_error: str | None = None


# Words that signal an image-classification request within this application's
# scope. The list is intentionally narrow: it reflects what the CIFAR-10
# classifier can actually answer.
_IMAGE_INTENT_WORDS = (
    "image",
    "picture",
    "photo",
    "pic",
    "img",
)

_CLASSIFY_VERBS = (
    "classify",
    "analyze",
    "analyse",
    "identify",
    "recognize",
    "recognise",
    "detect",
    "predict",
    "label",
)

# Requests that are clearly outside CIFAR-10 image classification. Kept small
# and conservative: anything not clearly out of scope falls through to
# ambiguous rather than being guessed at.
_UNSUPPORTED_PATTERN = re.compile(
    r"\b(weather|forecast|temperature|rain|snow|news|stock|sports?|score|"
    r"recipe|cook|translate|math|calculate|joke|story|poem|email|"
    r"code|program|debug|movie|song|music|book|date|time|capital|"
    r"president|person|people|who is|who are)\b",
    re.IGNORECASE,
)

# Hedges or content-free phrasing that carry no classifiable intent on their
# own, e.g. "tell me about this." With no image reference and no question
# signal, these requests cannot be safely routed.
_AMBIGUOUS_PATTERN = re.compile(
    r"^(tell me about( this| it)?|what about this|and this|this\??|"
    r"explain( this| it)?|describe( this| it)?|talk about( this| it)?|"
    r"say something( about this)?)[.!]?$",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _extract_target_class(normalized: str) -> str | None:
    for class_name in CANONICAL_CIFAR10_CLASSES:
        if re.search(rf"\b{re.escape(class_name)}\b", normalized):
            return class_name
    return None


def parse_request(raw_text: str | None) -> ParsedRequest:
    """Parse free-form user text into a structured :class:`ParsedRequest`.

    The parse is deterministic and conservative:

    - empty input -> ``Intent.AMBIGUOUS`` with a validation error
    - clearly out-of-scope topics -> ``Intent.UNSUPPORTED``
    - content-free references like "tell me about this" -> ``Intent.AMBIGUOUS``
    - image/classification wording -> ``Intent.CLASSIFY_IMAGE`` with optional
      CIFAR-10 ``target_class`` extraction
    - anything else -> ``Intent.AMBIGUOUS`` (never invent an intent)
    """

    text = raw_text or ""
    stripped = text.strip()
    if not stripped:
        return ParsedRequest(
            raw_text=text,
            intent=Intent.AMBIGUOUS,
            requires_image=False,
            validation_error="The request is empty. Ask a question about a CIFAR-10 image.",
        )

    normalized = _normalize(stripped)

    if _UNSUPPORTED_PATTERN.search(normalized):
        return ParsedRequest(
            raw_text=stripped,
            intent=Intent.UNSUPPORTED,
            requires_image=False,
            validation_error=(
                "This request is outside the application's scope. "
                "The assistant only answers questions about CIFAR-10 image classification."
            ),
        )

    if _AMBIGUOUS_PATTERN.match(normalized):
        return ParsedRequest(
            raw_text=stripped,
            intent=Intent.AMBIGUOUS,
            requires_image=False,
            validation_error=(
                "The request is too vague to route. Ask what an image contains "
                "or whether it contains a specific CIFAR-10 class."
            ),
        )

    has_image_word = any(re.search(rf"\b{word}\b", normalized) for word in _IMAGE_INTENT_WORDS)
    has_classify_verb = any(re.search(rf"\b{verb}\b", normalized) for verb in _CLASSIFY_VERBS)
    target_class = _extract_target_class(normalized)

    # Question words (what/which/does/is) combined with an image reference or a
    # CIFAR-10 class name are safe to treat as classification requests.
    is_question = normalized.endswith("?") or normalized.startswith(
        ("what", "which", "does", "is", "can", "could", "show")
    )

    if has_image_word or has_classify_verb or (is_question and target_class is not None):
        return ParsedRequest(
            raw_text=stripped,
            intent=Intent.CLASSIFY_IMAGE,
            requires_image=True,
            target_class=target_class,
            normalized_question=normalized,
        )

    return ParsedRequest(
        raw_text=stripped,
        intent=Intent.AMBIGUOUS,
        requires_image=False,
        validation_error=(
            "The request could not be safely routed. Ask what a CIFAR-10 image contains "
            "or whether it contains one of: " + ", ".join(CANONICAL_CIFAR10_CLASSES) + "."
        ),
    )

from __future__ import annotations

import pytest

from src.decision_intelligence_engine.model_inference import CANONICAL_CIFAR10_CLASSES
from src.decision_intelligence_engine.request_parser import Intent, ParsedRequest, parse_request


def test_empty_question_is_ambiguous_with_error() -> None:
    for empty in (None, "", "   "):
        request = parse_request(empty)
        assert request.intent is Intent.AMBIGUOUS
        assert request.requires_image is False
        assert request.validation_error is not None


def test_normal_classification_request() -> None:
    request = parse_request("Analyze this image and tell me what it contains.")
    assert request.intent is Intent.CLASSIFY_IMAGE
    assert request.requires_image is True
    assert request.target_class is None
    assert request.normalized_question is not None
    assert request.validation_error is None


def test_what_is_shown_question() -> None:
    request = parse_request("What is shown in this image?")
    assert request.intent is Intent.CLASSIFY_IMAGE
    assert request.requires_image is True
    assert request.target_class is None


@pytest.mark.parametrize("class_name", sorted(CANONICAL_CIFAR10_CLASSES))
def test_target_class_extraction_for_all_cifar10_classes(class_name: str) -> None:
    request = parse_request(f"Does this image contain a {class_name}?")
    assert request.intent is Intent.CLASSIFY_IMAGE
    assert request.requires_image is True
    assert request.target_class == class_name
    assert request.validation_error is None


def test_target_class_extraction_without_image_word() -> None:
    request = parse_request("Is this a frog?")
    assert request.intent is Intent.CLASSIFY_IMAGE
    assert request.target_class == "frog"
    assert request.requires_image is True


def test_clearly_unsupported_request() -> None:
    request = parse_request("What will the weather be tomorrow?")
    assert request.intent is Intent.UNSUPPORTED
    assert request.requires_image is False
    assert request.target_class is None
    assert request.validation_error is not None


def test_out_of_scope_chat_request() -> None:
    request = parse_request("Who is the president?")
    assert request.intent is Intent.UNSUPPORTED


def test_vague_request_is_ambiguous() -> None:
    request = parse_request("Tell me about this.")
    assert request.intent is Intent.AMBIGUOUS
    assert request.requires_image is False
    assert request.validation_error is not None


def test_unroutable_request_defaults_to_ambiguous_not_guess() -> None:
    request = parse_request("Take it further")
    assert request.intent is Intent.AMBIGUOUS
    assert request.validation_error is not None


def test_parser_is_deterministic() -> None:
    samples = [
        "What is shown in this image?",
        "Does this image contain a frog?",
        "What will the weather be tomorrow?",
        "Tell me about this.",
        "",
    ]
    for sample in samples:
        first = parse_request(sample)
        second = parse_request(sample)
        assert first == second


def test_parsed_request_is_frozen() -> None:
    request = parse_request("What is shown in this image?")
    with pytest.raises(AttributeError):
        request.intent = Intent.UNSUPPORTED  # type: ignore[misc]


def test_class_names_not_embedded_in_other_words() -> None:
    # "caterpillar" must not extract the "cat" class via substring matching.
    request = parse_request("What is shown in this picture?")
    assert request.target_class is None


def test_raw_text_is_preserved() -> None:
    request = parse_request("  Does this image contain a ship?  ")
    assert request.raw_text == "Does this image contain a ship?"

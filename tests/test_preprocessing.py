import pytest

from ml.preprocessing import clean_text


def test_clean_text_normalizes_case_and_spacing():
    assert clean_text("  Python   TESTING!! ") == "python testing"


def test_clean_text_rejects_empty_input():
    with pytest.raises(ValueError):
        clean_text("   ")


def test_clean_text_rejects_non_string_input():
    with pytest.raises(TypeError):
        clean_text(None)


def test_clean_text_rejects_input_without_usable_characters():
    with pytest.raises(ValueError, match="letters or numbers"):
        clean_text("!" * 20)

"""
As an experiment, this script was initially generated using ChatGPT 4 April 2023 Version.
It was subsequently modified to fix errors and better cover the concerned code.
"""

import pytest
from pretty_midi import PrettyMIDI

from midiogre.core.compositions import Compose


def test_initialization():
    """Test the initialization of the Compose class."""
    transforms = [lambda x: x, lambda x: x]
    composer = Compose(transforms)
    assert len(composer) == len(transforms)


def test_call_single_transform():
    """Test the __call__ method with a single transform."""
    transform = lambda midi: midi
    composer = Compose([transform])
    mock_midi = PrettyMIDI()

    result = composer(mock_midi)
    assert result == mock_midi


def test_call_multiple_transforms():
    """Test the __call__ method with multiple transforms."""
    transform1 = lambda midi: midi
    transform2 = lambda midi: midi
    composer = Compose([transform1, transform2])
    mock_midi = PrettyMIDI()

    result = composer(mock_midi)
    assert result == mock_midi


def test_empty_transforms():
    """Test the __call__ method with no transforms."""
    composer = Compose([])
    mock_midi = PrettyMIDI()
    result = composer(mock_midi)

    assert result == mock_midi


def test_transforms_type_check():
    """Test that an error is raised if the transforms are not a list or tuple."""
    with pytest.raises(TypeError):
        Compose("not a list or tuple")


def test_transforms_functionality():
    """Test the actual functionality of the transforms."""

    def add_instrument(midi):
        midi.instruments.append("New Instrument")
        return midi

    def double_tempo(midi):
        midi.estimate_tempo = lambda: 240
        return midi

    composer = Compose([add_instrument, double_tempo])
    mock_midi = PrettyMIDI()
    mock_midi.instruments = []
    mock_midi.estimate_tempo = lambda: 120

    result = composer(mock_midi)
    assert "New Instrument" in result.instruments
    assert result.estimate_tempo() == 240


if __name__ == '__main__':
    pytest.main()

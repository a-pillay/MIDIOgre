"""
As an experiment, this script was initially generated using ChatGPT September 25 Version (Free Research Preview).
It was subsequently modified to fix errors and better cover the concerned code.
"""

import math

import numpy as np
import pytest

from MIDIOgre.augmentations.note_add import NoteAdd
from tests.core_mocks import generate_mock_midi_data


@pytest.fixture
def note_add_instance():
    note_num_range = (60, 80)  # Adjust as needed
    note_velocity_range = (40, 80)  # Adjust as needed
    note_duration_range = (0.1, 0.5)  # Adjust as needed
    restrict_to_instrument_time = True  # Adjust as needed
    p_instruments = 1.0
    p = 0.0
    return NoteAdd(note_num_range, note_velocity_range, note_duration_range, restrict_to_instrument_time,
                   p_instruments=p_instruments, p=p)


def test_invalid_note_num_range():
    # Test with invalid note_num_range
    invalid_note_num_range = (80, 60)  # Max < Min
    with pytest.raises(ValueError):
        NoteAdd(invalid_note_num_range, (40, 80), (0.1, 0.5), True)


def test_invalid_note_velocity_range():
    # Test with invalid note_velocity_range
    invalid_note_velocity_range = (80, 40)  # Max < Min
    with pytest.raises(ValueError):
        NoteAdd((60, 80), invalid_note_velocity_range, (0.1, 0.5), True)


def test_invalid_note_duration_range():
    # Test with invalid note_duration_range
    invalid_note_duration_range = (0.5, 0.1)  # Min > Max
    with pytest.raises(ValueError):
        NoteAdd((60, 80), (40, 80), invalid_note_duration_range, True)


def test_invalid_note_num_values():
    # Test with invalid note_num_range values
    invalid_note_num_range = (60, 128)  # Max > 127
    with pytest.raises(ValueError):
        NoteAdd(invalid_note_num_range, (40, 80), (0.1, 0.5), True)

    invalid_note_num_range = (128, 129)  # Max > 127
    with pytest.raises(ValueError):
        NoteAdd(invalid_note_num_range, (40, 80), (0.1, 0.5), True)


def test_invalid_note_velocity_values():
    # Test with invalid note_velocity_range values
    invalid_note_velocity_range = (40, 128)  # Max > 127
    with pytest.raises(ValueError):
        NoteAdd((60, 80), invalid_note_velocity_range, (0.1, 0.5), True)

    invalid_note_velocity_range = (128, 129)  # Max > 127
    with pytest.raises(ValueError):
        NoteAdd((60, 80), invalid_note_velocity_range, (0.1, 0.5), True)


def test_apply_enough_notes(note_add_instance, monkeypatch):
    # Generate a MidiData object with a mock instrument
    original_num_notes = 10
    midi_data = generate_mock_midi_data(num_notes=original_num_notes)

    # Mock np.random.randint to return predefined values
    def mock_randint(low, high):
        # Mocking a note pitch within the specified range
        return 75

    # Mock np.random.uniform to return predefined values
    def mock_uniform(low, high):
        # Mocking a note duration within the specified range
        return 0.3

    monkeypatch.setattr(np.random, 'randint', mock_randint)
    monkeypatch.setattr(np.random, 'uniform', mock_uniform)

    # Apply note addition
    modified_midi_data = note_add_instance.apply(midi_data)

    # Ensure that a new note has been added to the instrument
    assert len(modified_midi_data.instruments[0].notes) == math.ceil(1.3 * original_num_notes)


if __name__ == '__main__':
    pytest.main()

'''
As an experiment, this script was initially generated using ChatGPT September 25 Version (Free Research Preview).
It was subsequently modified to fix errors and better cover the concerned code.
'''

from unittest.mock import Mock
import random

import numpy as np
import pytest

from augmentations.pitch_shift import PitchShift


@pytest.fixture
def pitch_shift_instance():
    max_shift = 12
    mode = 'both'
    p_instruments = 1.0
    p = 0.2
    return PitchShift(max_shift, mode, p_instruments, p)


def test_valid_mode(pitch_shift_instance):
    pitch_shift = PitchShift(pitch_shift_instance.max_shift, 'both', pitch_shift_instance.p_instruments,
                             pitch_shift_instance.p)
    assert pitch_shift.mode == pitch_shift._both

    pitch_shift = PitchShift(pitch_shift_instance.max_shift, 'up', pitch_shift_instance.p_instruments,
                             pitch_shift_instance.p)
    assert pitch_shift.mode == pitch_shift._up

    pitch_shift = PitchShift(pitch_shift_instance.max_shift, 'down', pitch_shift_instance.p_instruments,
                             pitch_shift_instance.p)
    assert pitch_shift.mode == pitch_shift._down


def test_invalid_mode():
    invalid_mode = 'invalid_mode'
    with pytest.raises(ValueError):
        PitchShift(12, invalid_mode, 1.0, 0.2)


def test_valid_max_shift(pitch_shift_instance):
    valid_max_shift = 42
    pitch_shift = PitchShift(valid_max_shift, 'both', pitch_shift_instance.p_instruments,
                             pitch_shift_instance.p)
    assert pitch_shift.max_shift == valid_max_shift


def test_invalid_max_shift():
    with pytest.raises(ValueError):
        PitchShift(128, 'both', 1.0, 0.2)


def test_apply_enough_notes(pitch_shift_instance, monkeypatch):
    # Create a mock MIDI data object with instruments and notes
    midi_data = Mock()
    instrument = Mock()
    note = Mock()
    note.pitch = 60
    instrument.notes = [note] * 10
    instrument.is_drum = False
    midi_data.instruments = [instrument]

    # Mock random.randint to always return 1 for predictable testing
    def mock_np_randint(low, high, size=None):
        return 1

    def mock_randsamp(modified_instruments, k):
        return [note]

    monkeypatch.setattr(np.random, 'randint', mock_np_randint)
    monkeypatch.setattr(random, 'sample', mock_randsamp)

    # Apply pitch shift
    modified_midi_data = pitch_shift_instance.apply(midi_data)

    # Ensure that the note pitch has been modified
    assert modified_midi_data.instruments[0].notes[0].pitch == 61


def test_apply_not_enough_notes(pitch_shift_instance, monkeypatch):
    # Create a mock MIDI data object with instruments and notes
    midi_data = Mock()
    instrument = Mock()
    note = Mock()
    note.pitch = 60
    instrument.notes = [note]
    instrument.is_drum = False
    midi_data.instruments = [instrument]

    # Mock random.randint to always return 1 for predictable testing
    def mock_np_randint(low, high, size=None):
        return 1

    def mock_randsamp(modified_instruments, k):
        return [note]

    monkeypatch.setattr(np.random, 'randint', mock_np_randint)
    monkeypatch.setattr(random, 'sample', mock_randsamp)

    # Apply pitch shift
    modified_midi_data = pitch_shift_instance.apply(midi_data)

    # Ensure that the note pitch has not been modified
    assert modified_midi_data.instruments[0].notes[0].pitch == 60

"""
As an experiment, this script was initially generated using ChatGPT September 25 Version (Free Research Preview).
It was subsequently modified to fix errors and better cover the concerned code.
"""

from unittest.mock import Mock
import random

import numpy as np
import pytest

from augmentations.onset_time_shift import OnsetTimeShift


@pytest.fixture
def onset_time_shift_instance():
    max_shift = 0.5  # Adjust as needed
    mode = 'both'
    p_instruments = 1.0
    p = 0.2
    return OnsetTimeShift(max_shift, mode, p_instruments, p)


def test_valid_mode(onset_time_shift_instance):
    onset_time_shift = OnsetTimeShift(onset_time_shift_instance.max_shift, 'both',
                                      onset_time_shift_instance.p_instruments, onset_time_shift_instance.p)
    assert onset_time_shift.mode == onset_time_shift._both

    onset_time_shift = OnsetTimeShift(onset_time_shift_instance.max_shift, 'left',
                                      onset_time_shift_instance.p_instruments, onset_time_shift_instance.p)
    assert onset_time_shift.mode == onset_time_shift._left

    onset_time_shift = OnsetTimeShift(onset_time_shift_instance.max_shift, 'right',
                                      onset_time_shift_instance.p_instruments, onset_time_shift_instance.p)
    assert onset_time_shift.mode == onset_time_shift._right


def test_invalid_mode():
    invalid_mode = 'invalid_mode'
    with pytest.raises(ValueError):
        OnsetTimeShift(0.5, invalid_mode, 1.0, 0.2)


def test_apply_enough_notes(onset_time_shift_instance, monkeypatch):
    # Create a mock MIDI data object with instruments and notes
    midi_data = Mock()
    instrument = Mock()
    note = Mock()
    note.start = 0.0
    note.duration = 1.0
    note.end = 1.0
    instrument.notes = [note] * 10
    instrument.is_drum = False
    midi_data.instruments = [instrument]

    # Mock np.random.uniform to always return 0.5 for predictable testing
    def mock_uniform(low, high):
        return 0.5

    # Mock random.sample to return all notes
    def mock_randsamp(modified_instruments, k):
        return [note]

    monkeypatch.setattr(np.random, 'uniform', mock_uniform)
    monkeypatch.setattr(random, 'sample', mock_randsamp)

    # Apply onset time shift
    modified_midi_data = onset_time_shift_instance.apply(midi_data)

    # Ensure that the note onset times have been modified
    assert modified_midi_data.instruments[0].notes[0].start == 0.5
    assert modified_midi_data.instruments[0].notes[0].end == 1.5


def test_apply_not_enough_notes(onset_time_shift_instance, monkeypatch):
    # Create a mock MIDI data object with instruments and notes
    midi_data = Mock()
    instrument = Mock()
    note = Mock()
    note.start = 0.0
    note.duration = 1.0
    note.end = 1.0
    instrument.notes = [note]
    instrument.is_drum = False
    midi_data.instruments = [instrument]

    # Mock np.random.uniform to always return 0.5 for predictable testing
    def mock_uniform(low, high):
        return 0.5

    # Mock random.sample to return all notes
    def mock_randsamp(modified_instruments, k):
        return [note]

    monkeypatch.setattr(np.random, 'uniform', mock_uniform)
    monkeypatch.setattr(random, 'sample', mock_randsamp)

    # Apply onset time shift
    modified_midi_data = onset_time_shift_instance.apply(midi_data)

    # Ensure that the note onset times have been modified
    assert modified_midi_data.instruments[0].notes[0].start == 0.0
    assert modified_midi_data.instruments[0].notes[0].end == 1.0


if __name__ == '__main__':
    pytest.main()

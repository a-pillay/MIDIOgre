"""
As an experiment, this script was initially generated using ChatGPT September 25 Version (Free Research Preview).
It was subsequently modified to fix errors and better cover the concerned code.
"""

import random

import numpy as np
import pytest

from midiogre.augmentations.duration_shift import DurationShift
from tests.core_mocks import generate_mock_midi_data


@pytest.fixture
def duration_shift_instance():
    max_shift = 0.5  # Adjust as needed
    mode = 'both'
    min_duration = 1e-6
    p_instruments = 1.0
    p = 0.2
    return DurationShift(max_shift, mode, min_duration, p_instruments, p)


def test_valid_mode(duration_shift_instance):
    onset_time_shift = DurationShift(duration_shift_instance.max_shift, 'both',
                                     duration_shift_instance.p_instruments, duration_shift_instance.p)
    assert onset_time_shift.mode == onset_time_shift._both

    onset_time_shift = DurationShift(duration_shift_instance.max_shift, 'shrink',
                                     duration_shift_instance.p_instruments, duration_shift_instance.p)
    assert onset_time_shift.mode == onset_time_shift._shrink

    onset_time_shift = DurationShift(duration_shift_instance.max_shift, 'extend',
                                     duration_shift_instance.p_instruments, duration_shift_instance.p)
    assert onset_time_shift.mode == onset_time_shift._extend


def test_invalid_mode():
    invalid_mode = 'invalid_mode'
    with pytest.raises(ValueError):
        DurationShift(0.5, invalid_mode, 1e-6, 1.0, 0.2)


def test_apply_enough_notes(duration_shift_instance, monkeypatch):
    midi_data = generate_mock_midi_data(num_notes=10)

    # Mock np.random.uniform to always return 0.5 for predictable testing
    def mock_uniform(low, high):
        return 0.5

    # Mock random.sample to return only the first note
    def mock_randsamp(modified_instruments, k):
        return [midi_data.instruments[0].notes[0]]

    monkeypatch.setattr(np.random, 'uniform', mock_uniform)
    monkeypatch.setattr(random, 'sample', mock_randsamp)

    # Apply duration shift
    modified_midi_data = duration_shift_instance.apply(midi_data)

    # Ensure that the first note duration has been modified
    assert modified_midi_data.instruments[0].notes[0].start == 0.0
    assert modified_midi_data.instruments[0].notes[0].end == 1.5


def test_apply_not_enough_notes(duration_shift_instance, monkeypatch):
    midi_data = generate_mock_midi_data(num_notes=1)

    # Mock np.random.uniform to always return 0.5 for predictable testing
    def mock_uniform(low, high):
        return 0.5

    # Mock random.sample to return only the first note
    def mock_randsamp(modified_instruments, k):
        return [midi_data.instruments[0].notes[0]]

    monkeypatch.setattr(np.random, 'uniform', mock_uniform)
    monkeypatch.setattr(random, 'sample', mock_randsamp)

    # Apply duration shift
    modified_midi_data = duration_shift_instance.apply(midi_data)

    # Ensure that the first note duration has been modified
    assert modified_midi_data.instruments[0].notes[0].start == 0.0
    assert modified_midi_data.instruments[0].notes[0].end == 1.0


if __name__ == '__main__':
    pytest.main()

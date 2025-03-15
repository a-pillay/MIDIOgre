"""
As an experiment, this script was initially generated using ChatGPT September 25 Version (Free Research Preview).
It was subsequently modified to fix errors and better cover the concerned code.
"""

import random
import pytest
from midiogre.augmentations.note_delete import NoteDelete
from tests.core_mocks import generate_mock_midi_data
import numpy as np


@pytest.fixture
def note_delete_instance():
    p_instruments = 1.0
    p = 0.2
    return NoteDelete(p_instruments=p_instruments, p=p)


def test_apply_enough_notes(note_delete_instance, monkeypatch):
    tot_notes = 10
    midi_data = generate_mock_midi_data(num_notes=tot_notes)

    # Mock random generation to delete exactly 20% of notes
    def mock_uniform(low, high, size=None):
        return note_delete_instance.p  # Return p to delete maximum number of notes

    def mock_choice(population, size=None, replace=False):
        if isinstance(population, int):
            population = range(population)
        return list(range(size))  # Return first 'size' indices deterministically

    monkeypatch.setattr(np.random, 'uniform', mock_uniform)
    monkeypatch.setattr(np.random, 'choice', mock_choice)

    # Apply note deletion
    modified_midi_data = note_delete_instance.apply(midi_data)

    # Ensure that the correct number of notes have been deleted
    # With p=0.2, expect 20% of notes to be deleted (2 notes), leaving 8 notes
    assert len(modified_midi_data.instruments[0].notes) == 8


def test_apply_not_enough_notes(note_delete_instance, monkeypatch):
    tot_notes = 2
    midi_data = generate_mock_midi_data(num_notes=tot_notes)

    # Mock random generation to return values that preserve all notes
    def mock_uniform(low, high, size=None):
        return 0  # Return 0 to ensure no notes are deleted

    def mock_choice(population, size=None, replace=False):
        if isinstance(population, int):
            population = range(population)
        return list(population)[:size]  # Return all indices to keep all notes

    monkeypatch.setattr(np.random, 'uniform', mock_uniform)
    monkeypatch.setattr(np.random, 'choice', mock_choice)

    # Apply note deletion
    modified_midi_data = note_delete_instance.apply(midi_data)

    # Ensure that all notes have been preserved
    assert len(modified_midi_data.instruments[0].notes) == 2


if __name__ == '__main__':
    pytest.main()

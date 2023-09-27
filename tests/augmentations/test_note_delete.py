import random
import logging
import pytest
from operator import itemgetter
from augmentations.note_delete import NoteDelete
from tests.augmentations.core_mocks import generate_mock_midi_data


@pytest.fixture
def note_delete_instance():
    p_instruments = 1.0
    p = 0.2
    return NoteDelete(p_instruments=p_instruments, p=p)


def test_apply_enough_notes(note_delete_instance, monkeypatch):
    tot_notes = 10
    midi_data = generate_mock_midi_data(num_notes=tot_notes)

    # Mock random.sample to return indices of notes to be deleted
    def mock_randsamp(modified_instruments, k):
        preserved_notes = list(range(tot_notes))
        random.shuffle(preserved_notes)
        return preserved_notes[:int(tot_notes * note_delete_instance.p)]

    monkeypatch.setattr(random, 'sample', mock_randsamp)

    # Apply note deletion
    modified_midi_data = note_delete_instance.apply(midi_data)

    # Ensure that the correct number of notes have been deleted
    assert len(modified_midi_data.instruments[0].notes) == 8


def test_apply_not_enough_notes(note_delete_instance, monkeypatch):
    tot_notes = 2
    midi_data = generate_mock_midi_data(num_notes=tot_notes)

    # Mock random.sample to return indices of notes to be deleted
    def mock_randsamp(modified_instruments, k):
        preserved_notes = list(range(tot_notes))
        random.shuffle(preserved_notes)
        return preserved_notes[:k]

    monkeypatch.setattr(random, 'sample', mock_randsamp)

    # Apply note deletion
    modified_midi_data = note_delete_instance.apply(midi_data)

    # Ensure that all notes have been deleted
    assert len(modified_midi_data.instruments[0].notes) == 2


if __name__ == '__main__':
    pytest.main()

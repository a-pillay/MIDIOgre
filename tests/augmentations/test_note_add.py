"""
As an experiment, this script was initially generated using ChatGPT September 25 Version (Free Research Preview).
It was subsequently modified to fix errors and better cover the concerned code.
"""

import math
import time

import numpy as np
import pytest
from pretty_midi import Note

from midiogre.augmentations.note_add import NoteAdd
from tests.core_mocks import generate_mock_midi_data


@pytest.fixture
def note_add_instance():
    note_num_range = (60, 80)  # Adjust as needed
    note_velocity_range = (40, 80)  # Adjust as needed
    note_duration_range = (0.1, 0.5)  # Adjust as needed
    restrict_to_instrument_time = True  # Adjust as needed
    p_instruments = 1.0
    p = 0.3  # Changed to 0.3 to add more notes for testing
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
    """Test that the correct number of notes are added with mocked random values."""
    original_num_notes = 10
    midi_data = generate_mock_midi_data(num_notes=original_num_notes)

    # Mock np.random.randint to return predefined values
    def mock_randint(low, high, size=None):
        if size is None:
            return 75
        return np.full(size, 75)

    # Mock np.random.uniform to return predefined values
    def mock_uniform(low, high, size=None):
        if size is None:
            return 0.3
        return np.full(size, 0.3)

    monkeypatch.setattr(np.random, 'randint', mock_randint)
    monkeypatch.setattr(np.random, 'uniform', mock_uniform)

    # Apply note addition
    modified_midi = note_add_instance.apply(midi_data)
    
    # With p=0.3, we expect 3 new notes (30% of 10)
    expected_notes = original_num_notes + math.ceil(0.3 * original_num_notes)
    assert len(modified_midi.instruments[0].notes) == expected_notes
    
    # Verify the mocked values
    for note in modified_midi.instruments[0].notes[original_num_notes:]:
        assert note.pitch == 75
        assert note.velocity == 75
        assert note.start == 0.3
        assert note.end == 0.6  # start + duration


def test_empty_instrument():
    """Test handling of empty instruments."""
    midi_data = generate_mock_midi_data(num_notes=0)
    note_add = NoteAdd((60, 80), (40, 80), (0.1, 0.5))
    modified_midi = note_add.apply(midi_data)
    assert len(modified_midi.instruments[0].notes) == 0


def test_note_bounds(note_add_instance):
    """Test that added notes respect the given bounds."""
    original_num_notes = 10
    midi_data = generate_mock_midi_data(num_notes=original_num_notes)
    modified_midi = note_add_instance.apply(midi_data)
    
    # Check original notes remain unchanged
    for i in range(original_num_notes):
        note = modified_midi.instruments[0].notes[i]
        assert note.pitch == 60 + i
        assert note.velocity == 70
        assert note.start == float(i)
        assert note.end == float(i + 1)
    
    # Check only the newly added notes respect the bounds
    for note in modified_midi.instruments[0].notes[original_num_notes:]:
        assert isinstance(note.pitch, (int, np.integer))
        assert isinstance(note.velocity, (int, np.integer))
        assert 60 <= note.pitch <= 80
        assert 40 <= note.velocity <= 80
        assert 0 <= note.start <= note.end
        assert 0.1 <= (note.end - note.start) <= 0.5


def test_note_count(note_add_instance):
    """Test that the correct number of notes are added."""
    original_num_notes = 10
    midi_data = generate_mock_midi_data(num_notes=original_num_notes)
    modified_midi = note_add_instance.apply(midi_data)
    
    # With p=0.3, we expect between 0 and 3 new notes (30% of 10)
    added_notes = len(modified_midi.instruments[0].notes) - original_num_notes
    assert 0 <= added_notes <= math.ceil(0.3 * original_num_notes)


def test_performance():
    """Test performance of note generation."""
    note_add = NoteAdd((60, 80), (40, 80), (0.1, 0.5))
    midi_data = generate_mock_midi_data(num_notes=1000)  # Large number of notes
    
    # Time the operation
    start_time = time.time()
    modified_midi = note_add.apply(midi_data)
    end_time = time.time()
    
    # Performance should be reasonable (adjust threshold as needed)
    assert end_time - start_time < 0.1  # Should complete in under 100ms


def test_vectorized_generation(note_add_instance):
    """Test that note generation is properly vectorized."""
    n = 1000  # Large number to test vectorization
    instrument_end_time = 10.0
    
    # Time the generation
    start_time = time.time()
    notes = note_add_instance._NoteAdd__generate_n_midi_notes(n, instrument_end_time)
    end_time = time.time()
    
    assert len(notes) == n
    # Vectorized operation should be fast
    assert end_time - start_time < 0.1  # Should complete in under 100ms


def test_note_end_time_restriction(note_add_instance):
    """Test that notes respect the instrument end time."""
    midi_data = generate_mock_midi_data(num_notes=10)
    instrument_end_time = midi_data.instruments[0].notes[-1].end
    
    modified_midi = note_add_instance.apply(midi_data)
    
    # Check that no note ends after the instrument end time
    for note in modified_midi.instruments[0].notes:
        assert note.end <= instrument_end_time


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

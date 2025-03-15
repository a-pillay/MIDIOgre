"""
As an experiment, this script was initially generated using ChatGPT September 25 Version (Free Research Preview).
It was subsequently modified to fix errors and better cover the concerned code.
"""

import random
import time

import numpy as np
import pytest

from midiogre.augmentations.duration_shift import DurationShift, VALID_MODES
from tests.core_mocks import generate_mock_midi_data


@pytest.fixture
def duration_shift_instance():
    return DurationShift(max_shift=0.5, mode='both', min_duration=0.1, p_instruments=1.0, p=0.3)


def test_invalid_max_shift():
    """Test initialization with invalid max_shift values."""
    with pytest.raises(ValueError):
        DurationShift(max_shift=-1)  # Negative shift not allowed


def test_invalid_min_duration():
    """Test initialization with invalid min_duration values."""
    with pytest.raises(ValueError):
        DurationShift(max_shift=0.5, min_duration=-1)  # Negative duration not allowed


def test_invalid_mode():
    """Test initialization with invalid mode."""
    with pytest.raises(ValueError):
        DurationShift(max_shift=0.5, mode='invalid')


def test_empty_instrument():
    """Test handling of empty instruments."""
    midi_data = generate_mock_midi_data(num_notes=0)
    duration_shift = DurationShift(max_shift=0.5)
    modified_midi = duration_shift.apply(midi_data)
    assert len(modified_midi.instruments[0].notes) == 0


def test_duration_bounds(duration_shift_instance):
    """Test that shifted durations stay within valid range."""
    original_num_notes = 10
    midi_data = generate_mock_midi_data(num_notes=original_num_notes)
    modified_midi = duration_shift_instance.apply(midi_data)
    
    # Check all notes have valid durations
    for note in modified_midi.instruments[0].notes:
        duration = note.end - note.start
        assert duration >= duration_shift_instance.min_duration
        assert note.start <= note.end  # End time should be after start time


def test_shift_modes():
    """Test all shift modes work correctly."""
    original_num_notes = 20
    max_shift = 0.5
    min_duration = 0.1
    
    for mode in VALID_MODES:
        # Create fresh MIDI data for each mode test
        midi_data = generate_mock_midi_data(num_notes=original_num_notes)
        duration_shift = DurationShift(
            max_shift=max_shift,
            mode=mode,
            min_duration=min_duration,
            p=1.0  # Shift all notes
        )
        modified_midi = duration_shift.apply(midi_data)
        
        original_durations = np.array([1.0] * original_num_notes)  # From mock data
        modified_durations = np.array([
            note.end - note.start 
            for note in modified_midi.instruments[0].notes
        ])
        
        # All durations should be valid
        assert all(modified_durations >= min_duration)
        
        if mode == 'shrink':
            # All durations should be <= original and at least one should be < original
            assert all(modified_durations <= original_durations)
            assert any(modified_durations < original_durations)
        elif mode == 'extend':
            # All durations should be >= original and at least one should be > original
            assert all(modified_durations >= original_durations)
            assert any(modified_durations > original_durations)
        else:  # both
            # Some durations should be different from original
            assert any(modified_durations != original_durations)
            # Maximum shift in either direction should be <= max_shift
            assert all(abs(modified_durations - original_durations) <= max_shift)


def test_note_selection(duration_shift_instance, monkeypatch):
    """Test that the correct number of notes are selected for shifting."""
    original_num_notes = 10
    midi_data = generate_mock_midi_data(num_notes=original_num_notes)
    
    # Mock random generation to return constant shifts
    def mock_uniform(low, high, size=None):
        if size is None:
            return 0.2
        return np.full(size, 0.2)
    
    def mock_sample(population, k):
        return list(population)[:k]  # Take first k elements deterministically
    
    monkeypatch.setattr(np.random, 'uniform', mock_uniform)
    monkeypatch.setattr(random, 'sample', mock_sample)
    
    modified_midi = duration_shift_instance.apply(midi_data)
    
    # Count how many notes were actually shifted
    shifted_notes = sum(1 for note in modified_midi.instruments[0].notes
                       if note.end - note.start != 1.0)  # Original duration is 1.0
    
    # With p=0.3, expect 30% of notes to be shifted
    expected_shifts = int(0.3 * original_num_notes)
    assert shifted_notes == expected_shifts


def test_performance():
    """Test performance of duration shifting."""
    duration_shift = DurationShift(max_shift=0.5)
    midi_data = generate_mock_midi_data(num_notes=1000)  # Large number of notes
    
    start_time = time.time()
    modified_midi = duration_shift.apply(midi_data)
    end_time = time.time()
    
    # Performance should be reasonable
    assert end_time - start_time < 0.1  # Should complete in under 100ms


def test_vectorized_generation(duration_shift_instance):
    """Test that shift generation is properly vectorized."""
    n = 1000  # Large number to test vectorization
    
    start_time = time.time()
    shifts = duration_shift_instance._generate_shifts(n)
    end_time = time.time()
    
    assert len(shifts) == n
    assert isinstance(shifts, np.ndarray)
    # Vectorized operation should be fast
    assert end_time - start_time < 0.1  # Should complete in under 100ms


def test_original_notes_unchanged(duration_shift_instance, monkeypatch):
    """Test that unselected notes remain unchanged."""
    original_num_notes = 10
    midi_data = generate_mock_midi_data(num_notes=original_num_notes)
    
    # Mock random.sample to always select the first 3 notes
    def mock_sample(population, k):
        return population[:3]
    
    monkeypatch.setattr(random, 'sample', mock_sample)
    
    # Mock random shifts to be constant
    def mock_uniform(low, high, size=None):
        if size is None:
            return 0.2
        return np.full(size, 0.2)
    
    monkeypatch.setattr(np.random, 'uniform', mock_uniform)
    
    modified_midi = duration_shift_instance.apply(midi_data)
    
    # First 3 notes should be shifted
    for i in range(3):
        assert modified_midi.instruments[0].notes[i].end == float(i) + 1.2  # Original + 0.2
    
    # Rest should be unchanged
    for i in range(3, original_num_notes):
        assert modified_midi.instruments[0].notes[i].start == float(i)
        assert modified_midi.instruments[0].notes[i].end == float(i + 1)


def test_instrument_end_time_respect():
    """Test that note durations don't exceed instrument end time."""
    original_num_notes = 10
    midi_data = generate_mock_midi_data(num_notes=original_num_notes)
    duration_shift = DurationShift(max_shift=5.0, mode='extend', p=1.0)  # Large shift to test boundary
    
    modified_midi = duration_shift.apply(midi_data)
    instrument_end_time = modified_midi.instruments[0].notes[-1].end
    
    # No note should end after the instrument's end time
    for note in modified_midi.instruments[0].notes:
        assert note.end <= instrument_end_time


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

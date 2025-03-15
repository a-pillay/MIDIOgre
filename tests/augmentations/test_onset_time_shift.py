"""Test cases for the OnsetTimeShift augmentation class."""

import random
import time

import numpy as np
import pytest

from midiogre.augmentations.onset_time_shift import OnsetTimeShift, VALID_MODES
from tests.core_mocks import generate_mock_midi_data


@pytest.fixture
def onset_shift_instance():
    return OnsetTimeShift(max_shift=0.5, mode='both', p_instruments=1.0, p=0.3)


def test_invalid_max_shift():
    """Test initialization with invalid max_shift values."""
    with pytest.raises(ValueError):
        OnsetTimeShift(max_shift=-1)  # Negative shift not allowed


def test_invalid_mode():
    """Test initialization with invalid mode."""
    with pytest.raises(ValueError):
        OnsetTimeShift(max_shift=0.5, mode='invalid')


def test_empty_instrument():
    """Test handling of empty instruments."""
    midi_data = generate_mock_midi_data(num_notes=0)
    onset_shift = OnsetTimeShift(max_shift=0.5)
    modified_midi = onset_shift.apply(midi_data)
    assert len(modified_midi.instruments[0].notes) == 0


def test_onset_bounds(onset_shift_instance):
    """Test that shifted onsets stay within valid range."""
    original_num_notes = 10
    midi_data = generate_mock_midi_data(num_notes=original_num_notes)
    modified_midi = onset_shift_instance.apply(midi_data)
    
    # Check all notes have valid onset times
    for note in modified_midi.instruments[0].notes:
        assert note.start >= 0  # Onset time should be non-negative
        assert note.start <= note.end  # Start time should be before end time


def test_duration_preservation(onset_shift_instance):
    """Test that note durations are preserved after shifting."""
    original_num_notes = 10
    midi_data = generate_mock_midi_data(num_notes=original_num_notes)
    
    # Store original durations
    original_durations = {
        id(note): note.end - note.start 
        for note in midi_data.instruments[0].notes
    }
    
    modified_midi = onset_shift_instance.apply(midi_data)
    
    # Check that durations remain unchanged
    for note in modified_midi.instruments[0].notes:
        current_duration = note.end - note.start
        assert np.isclose(current_duration, original_durations[id(note)])


def test_shift_modes():
    """Test all shift modes work correctly."""
    original_num_notes = 20
    max_shift = 0.5
    
    for mode in VALID_MODES:
        # Create fresh MIDI data for each mode test
        midi_data = generate_mock_midi_data(num_notes=original_num_notes)
        onset_shift = OnsetTimeShift(
            max_shift=max_shift,
            mode=mode,
            p=1.0  # Shift all notes
        )
        modified_midi = onset_shift.apply(midi_data)
        
        original_onsets = np.array([float(i) for i in range(original_num_notes)])  # From mock data
        modified_onsets = np.array([note.start for note in modified_midi.instruments[0].notes])
        
        # All onsets should be valid
        assert all(modified_onsets >= 0)
        
        if mode == 'left':
            # All onsets should be <= original and at least one should be < original
            assert all(modified_onsets <= original_onsets)
            assert any(modified_onsets < original_onsets)
        elif mode == 'right':
            # All onsets should be >= original and at least one should be > original
            assert all(modified_onsets >= original_onsets)
            assert any(modified_onsets > original_onsets)
        else:  # both
            # Some onsets should be different from original
            assert any(modified_onsets != original_onsets)
            # Maximum shift in either direction should be <= max_shift
            assert all(abs(modified_onsets - original_onsets) <= max_shift)


def test_note_selection(onset_shift_instance, monkeypatch):
    """Test that the correct number of notes are selected for shifting."""
    original_num_notes = 10
    midi_data = generate_mock_midi_data(num_notes=original_num_notes)
    
    # Mock random generation to return constant shifts
    def mock_uniform(low, high, size=None):
        if size is None:
            return 0.2
        return np.full(size, 0.2)
    
    monkeypatch.setattr(np.random, 'uniform', mock_uniform)
    
    modified_midi = onset_shift_instance.apply(midi_data)
    
    # Count how many notes were actually shifted
    shifted_notes = sum(1 for i, note in enumerate(modified_midi.instruments[0].notes)
                       if note.start != float(i))  # Original start time is float(i)
    
    # With p=0.3, expect 30% of notes to be shifted
    expected_shifts = int(0.3 * original_num_notes)
    assert shifted_notes == expected_shifts


def test_performance():
    """Test performance of onset shifting."""
    onset_shift = OnsetTimeShift(max_shift=0.5)
    midi_data = generate_mock_midi_data(num_notes=1000)  # Large number of notes
    
    start_time = time.time()
    modified_midi = onset_shift.apply(midi_data)
    end_time = time.time()
    
    # Performance should be reasonable
    assert end_time - start_time < 0.1  # Should complete in under 100ms


def test_vectorized_generation(onset_shift_instance):
    """Test that shift generation is properly vectorized."""
    n = 1000  # Large number to test vectorization
    
    start_time = time.time()
    shifts = onset_shift_instance._generate_shifts(n)
    end_time = time.time()
    
    assert len(shifts) == n
    assert isinstance(shifts, np.ndarray)
    # Vectorized operation should be fast
    assert end_time - start_time < 0.1  # Should complete in under 100ms


def test_original_notes_unchanged(onset_shift_instance, monkeypatch):
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
    
    modified_midi = onset_shift_instance.apply(midi_data)
    
    # First 3 notes should be shifted
    for i in range(3):
        assert modified_midi.instruments[0].notes[i].start == float(i) + 0.2  # Original + 0.2
        assert modified_midi.instruments[0].notes[i].end == float(i) + 1.2  # Original + 0.2
    
    # Rest should be unchanged
    for i in range(3, original_num_notes):
        assert modified_midi.instruments[0].notes[i].start == float(i)
        assert modified_midi.instruments[0].notes[i].end == float(i + 1)


def test_instrument_end_time_respect():
    """Test that note onsets don't exceed instrument end time."""
    original_num_notes = 10
    midi_data = generate_mock_midi_data(num_notes=original_num_notes)
    onset_shift = OnsetTimeShift(max_shift=5.0, mode='right', p=1.0)  # Large shift to test boundary
    
    modified_midi = onset_shift.apply(midi_data)
    instrument_end_time = modified_midi.instruments[0].notes[-1].end
    
    # No note should start after the instrument's end time
    for note in modified_midi.instruments[0].notes:
        assert note.start <= instrument_end_time


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
As an experiment, this script was initially generated using ChatGPT September 25 Version (Free Research Preview).
It was subsequently modified to fix errors and better cover the concerned code.
"""

import random
import time

import numpy as np
import pytest

from midiogre.augmentations.pitch_shift import PitchShift, VALID_MODES
from tests.core_mocks import generate_mock_midi_data


@pytest.fixture
def pitch_shift_instance():
    return PitchShift(max_shift=12, mode='both', p_instruments=1.0, p=0.3)


def test_invalid_max_shift():
    """Test initialization with invalid max_shift values."""
    with pytest.raises(ValueError):
        PitchShift(max_shift=128)  # Too large
    with pytest.raises(ValueError):
        PitchShift(max_shift=-1)  # Negative


def test_invalid_mode():
    """Test initialization with invalid mode."""
    with pytest.raises(ValueError):
        PitchShift(max_shift=12, mode='invalid')


def test_empty_instrument():
    """Test handling of empty instruments."""
    midi_data = generate_mock_midi_data(num_notes=0)
    pitch_shift = PitchShift(max_shift=12)
    modified_midi = pitch_shift.apply(midi_data)
    assert len(modified_midi.instruments[0].notes) == 0


def test_pitch_bounds(pitch_shift_instance):
    """Test that shifted pitches stay within valid MIDI note range."""
    original_num_notes = 10
    midi_data = generate_mock_midi_data(num_notes=original_num_notes)
    modified_midi = pitch_shift_instance.apply(midi_data)
    
    # Check all notes are within valid MIDI range
    for note in modified_midi.instruments[0].notes:
        assert isinstance(note.pitch, (int, np.integer))
        assert 0 <= note.pitch <= 127


def test_shift_modes():
    """Test all shift modes work correctly."""
    original_num_notes = 20  # Using fewer notes to avoid MIDI range issues
    max_shift = 12
    
    for mode in VALID_MODES:
        # Create fresh MIDI data for each mode test
        midi_data = generate_mock_midi_data(num_notes=original_num_notes)
        pitch_shift = PitchShift(max_shift=max_shift, mode=mode, p=1.0)  # p=1.0 to shift all notes
        modified_midi = pitch_shift.apply(midi_data)
        
        original_pitches = np.array([60 + i for i in range(original_num_notes)])
        modified_pitches = np.array([note.pitch for note in modified_midi.instruments[0].notes])
        
        # All notes should be within valid MIDI range
        assert all(0 <= modified_pitches) and all(modified_pitches <= 127)
        
        if mode == 'up':
            # All notes should be >= original and at least one should be > original
            assert all(modified_pitches >= original_pitches)
            assert any(modified_pitches > original_pitches)
        elif mode == 'down':
            # All notes should be <= original and at least one should be < original
            assert all(modified_pitches <= original_pitches)
            assert any(modified_pitches < original_pitches)
        else:  # both
            # Some notes should be different from original
            assert any(modified_pitches != original_pitches)
            # Maximum shift in either direction should be <= max_shift
            assert all(abs(modified_pitches - original_pitches) <= max_shift)


def test_note_selection(pitch_shift_instance, monkeypatch):
    """Test that the correct number of notes are selected for shifting."""
    original_num_notes = 10
    midi_data = generate_mock_midi_data(num_notes=original_num_notes)
    
    # Mock random generation to return constant shifts
    def mock_randint(low, high, size=None):
        if size is None:
            return 5
        return np.full(size, 5)
    
    monkeypatch.setattr(np.random, 'randint', mock_randint)
    
    modified_midi = pitch_shift_instance.apply(midi_data)
    
    # Count how many notes were actually shifted
    shifted_notes = sum(1 for i, note in enumerate(modified_midi.instruments[0].notes)
                       if note.pitch != (60 + i))
    
    # With p=0.3, expect 30% of notes to be shifted
    expected_shifts = int(0.3 * original_num_notes)
    assert shifted_notes == expected_shifts


def test_performance():
    """Test performance of pitch shifting."""
    pitch_shift = PitchShift(max_shift=12)
    midi_data = generate_mock_midi_data(num_notes=1000)  # Large number of notes
    
    start_time = time.time()
    modified_midi = pitch_shift.apply(midi_data)
    end_time = time.time()
    
    # Performance should be reasonable
    assert end_time - start_time < 0.1  # Should complete in under 100ms


def test_vectorized_generation(pitch_shift_instance):
    """Test that shift generation is properly vectorized."""
    n = 1000  # Large number to test vectorization
    
    start_time = time.time()
    shifts = pitch_shift_instance._generate_shifts(n)
    end_time = time.time()
    
    assert len(shifts) == n
    assert isinstance(shifts, np.ndarray)
    # Vectorized operation should be fast
    assert end_time - start_time < 0.1  # Should complete in under 100ms


def test_original_notes_unchanged(pitch_shift_instance, monkeypatch):
    """Test that unselected notes remain unchanged."""
    original_num_notes = 10
    midi_data = generate_mock_midi_data(num_notes=original_num_notes)
    
    # Mock random.sample to always select the first 3 notes
    def mock_sample(population, k):
        return population[:3]
    
    monkeypatch.setattr(random, 'sample', mock_sample)
    
    # Mock random shifts to be constant
    def mock_randint(low, high, size=None):
        if size is None:
            return 5
        return np.full(size, 5)
    
    monkeypatch.setattr(np.random, 'randint', mock_randint)
    
    modified_midi = pitch_shift_instance.apply(midi_data)
    
    # First 3 notes should be shifted
    for i in range(3):
        assert modified_midi.instruments[0].notes[i].pitch == 65 + i  # Original + 5
    
    # Rest should be unchanged
    for i in range(3, original_num_notes):
        assert modified_midi.instruments[0].notes[i].pitch == 60 + i


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
Test cases for the TempoShift augmentation class.
"""

import pytest
from mido import MidiFile, MetaMessage, Message, MidiTrack
import numpy as np

from midiogre.augmentations.tempo_shift import TempoShift, VALID_MODES


def create_mock_midi_with_tempo(tempo=500000):  # 120 BPM by default
    """Helper function to create a mock MIDI file with a tempo event."""
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    
    # Add tempo event
    track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
    # Add a note event
    track.append(Message('note_on', note=60, velocity=64, time=480))
    track.append(Message('note_off', note=60, velocity=64, time=480))
    
    return midi


def create_mock_midi_without_tempo():
    """Helper function to create a mock MIDI file without tempo events."""
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    
    # Add a note event
    track.append(Message('note_on', note=60, velocity=64, time=480))
    track.append(Message('note_off', note=60, velocity=64, time=480))
    
    return midi


def test_initialization():
    """Test initialization with valid parameters."""
    tempo_shift = TempoShift(max_shift=10)
    assert tempo_shift.max_shift == 10
    assert tempo_shift.mode == 'both'
    assert tempo_shift.tempo_range == (30.0, 200.0)
    assert tempo_shift.p == 0.2
    assert tempo_shift.respect_tempo_shifts is True


def test_invalid_mode():
    """Test initialization with invalid mode."""
    with pytest.raises(ValueError, match=r"Mode must be one of.*"):
        TempoShift(max_shift=10, mode='invalid')


def test_invalid_max_shift():
    """Test initialization with invalid max_shift."""
    with pytest.raises(ValueError, match=r"max_shift must be positive.*"):
        TempoShift(max_shift=0)
    with pytest.raises(ValueError, match=r"max_shift must be positive.*"):
        TempoShift(max_shift=-10)


def test_invalid_tempo_range():
    """Test initialization with invalid tempo range."""
    with pytest.raises(ValueError, match=r"min_tempo must be positive.*"):
        TempoShift(max_shift=10, tempo_range=(-1, 200))
    
    with pytest.raises(ValueError, match=r"min_tempo must be less than max_tempo.*"):
        TempoShift(max_shift=10, tempo_range=(200, 200))
    
    with pytest.raises(ValueError, match=r"min_tempo must be less than max_tempo.*"):
        TempoShift(max_shift=10, tempo_range=(200, 100))
    
    with pytest.raises(ValueError, match=r"tempo_range must be a tuple or list.*"):
        TempoShift(max_shift=10, tempo_range=100)


def test_generate_shifts():
    """Test shift generation for different modes."""
    tempo_shift = TempoShift(max_shift=10)
    
    # Test empty case
    shifts = tempo_shift._generate_shifts(0)
    assert len(shifts) == 0
    
    # Test up mode
    tempo_shift.mode = 'up'
    shifts = tempo_shift._generate_shifts(100)
    assert len(shifts) == 100
    assert all(0 <= shift <= 10 for shift in shifts)
    
    # Test down mode
    tempo_shift.mode = 'down'
    shifts = tempo_shift._generate_shifts(100)
    assert len(shifts) == 100
    assert all(-10 <= shift <= 0 for shift in shifts)
    
    # Test both mode
    tempo_shift.mode = 'both'
    shifts = tempo_shift._generate_shifts(100)
    assert len(shifts) == 100
    assert all(-10 <= shift <= 10 for shift in shifts)


def test_tempo_conversion():
    """Test tempo conversion methods."""
    tempo_shift = TempoShift(max_shift=10)
    
    # Test BPM to tempo conversion
    assert tempo_shift._convert_bpm_to_tempo(120) == 500000
    assert tempo_shift._convert_bpm_to_tempo(60) == 1000000
    
    # Test tempo to BPM conversion
    assert np.isclose(tempo_shift._convert_tempo_to_bpm(500000), 120)
    assert np.isclose(tempo_shift._convert_tempo_to_bpm(1000000), 60)


def test_empty_midi():
    """Test handling of empty MIDI file."""
    midi_data = MidiFile()
    tempo_shift = TempoShift(max_shift=10)
    modified_midi = tempo_shift.apply(midi_data)
    
    assert len(modified_midi.tracks) == 0


def test_midi_without_tempo():
    """Test handling of MIDI file without tempo events."""
    midi_data = create_mock_midi_without_tempo()
    tempo_shift = TempoShift(max_shift=10, p=1.0)  # Force change
    modified_midi = tempo_shift.apply(midi_data)
    
    # Should add a tempo event
    tempo_events = [msg for msg in modified_midi.tracks[0] if msg.type == 'set_tempo']
    assert len(tempo_events) == 1
    
    # Check if tempo is within valid range
    bpm = tempo_shift._convert_tempo_to_bpm(tempo_events[0].tempo)
    assert 30 <= bpm <= 200


def test_respect_tempo_shifts_true():
    """Test that multiple tempo events are preserved when respect_tempo_shifts is True."""
    midi_data = MidiFile()
    track = MidiTrack()
    midi_data.tracks.append(track)
    
    # Add multiple tempo events with different timings
    track.append(MetaMessage('set_tempo', tempo=500000, time=0))  # 120 BPM
    track.append(Message('note_on', note=60, velocity=64, time=480))
    track.append(MetaMessage('set_tempo', tempo=400000, time=0))  # 150 BPM
    track.append(Message('note_off', note=60, velocity=64, time=480))
    
    tempo_shift = TempoShift(max_shift=10, p=1.0, respect_tempo_shifts=True)
    modified_midi = tempo_shift.apply(midi_data)
    
    # Check that all tempo events are preserved with their timings
    tempo_events = [msg for msg in modified_midi.tracks[0] if msg.type == 'set_tempo']
    assert len(tempo_events) == 2
    
    # Check that timings are preserved
    assert tempo_events[0].time == 0
    assert tempo_events[1].time == 0
    
    # Check that tempos are within valid range
    for event in tempo_events:
        bpm = tempo_shift._convert_tempo_to_bpm(event.tempo)
        assert 30 <= bpm <= 200


def test_respect_tempo_shifts_false():
    """Test that only first tempo is used when respect_tempo_shifts is False."""
    midi_data = MidiFile()
    track = MidiTrack()
    midi_data.tracks.append(track)
    
    # Add multiple tempo events
    track.append(MetaMessage('set_tempo', tempo=500000, time=0))  # 120 BPM
    track.append(MetaMessage('set_tempo', tempo=400000, time=480))  # 150 BPM
    
    tempo_shift = TempoShift(max_shift=10, p=1.0, respect_tempo_shifts=False)
    modified_midi = tempo_shift.apply(midi_data)
    
    # Check that only one tempo event exists at the start
    tempo_events = [msg for msg in modified_midi.tracks[0] if msg.type == 'set_tempo']
    assert len(tempo_events) == 1
    assert tempo_events[0].time == 0
    
    # Check that tempo is within valid range
    bpm = tempo_shift._convert_tempo_to_bpm(tempo_events[0].tempo)
    assert 30 <= bpm <= 200


def test_probability():
    """Test that changes are only applied based on probability."""
    midi_data = create_mock_midi_with_tempo(tempo=500000)  # 120 BPM
    
    # Test with p=0 (should never change)
    tempo_shift = TempoShift(max_shift=10, p=0)
    modified_midi = tempo_shift.apply(midi_data)
    tempo_events = [msg for msg in modified_midi.tracks[0] if msg.type == 'set_tempo']
    assert tempo_events[0].tempo == 500000
    
    # Test with p=1 (should always change)
    tempo_shift = TempoShift(max_shift=10, p=1)
    modified_midi = tempo_shift.apply(midi_data)
    tempo_events = [msg for msg in modified_midi.tracks[0] if msg.type == 'set_tempo']
    assert tempo_events[0].tempo != 500000


def test_tempo_range_clipping():
    """Test that tempo stays within specified range."""
    midi_data = create_mock_midi_with_tempo(tempo=500000)  # 120 BPM
    tempo_shift = TempoShift(max_shift=1000, mode='both', tempo_range=(50, 180), p=1.0)
    
    # Test multiple times to ensure clipping works
    for _ in range(10):
        modified_midi = tempo_shift.apply(midi_data)
        tempo_events = [msg for msg in modified_midi.tracks[0] if msg.type == 'set_tempo']
        bpm = tempo_shift._convert_tempo_to_bpm(tempo_events[0].tempo)
        # Allow for small floating point imprecision
        assert 49.99 <= bpm <= 180.01


def test_performance():
    """Test performance with many tempo events."""
    midi_data = MidiFile()
    track = MidiTrack()
    midi_data.tracks.append(track)
    
    # Add many tempo events
    for _ in range(1000):
        track.append(MetaMessage('set_tempo', tempo=500000, time=480))
    
    tempo_shift = TempoShift(max_shift=10, respect_tempo_shifts=True)
    
    import time
    start_time = time.time()
    modified_midi = tempo_shift.apply(midi_data)
    end_time = time.time()
    
    # Should process quickly
    assert end_time - start_time < 0.1  # Should complete in under 100ms


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 
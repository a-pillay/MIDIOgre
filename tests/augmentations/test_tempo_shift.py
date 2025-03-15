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
    track.append(Message('note_on', note=60, velocity=64, time=0))
    track.append(Message('note_off', note=60, velocity=64, time=480))
    
    return midi


def create_mock_midi_without_tempo():
    """Helper function to create a mock MIDI file without a tempo event."""
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    
    # Add a note event without tempo
    track.append(Message('note_on', note=60, velocity=64, time=0))
    track.append(Message('note_off', note=60, velocity=64, time=480))
    
    return midi


def test_valid_mode():
    """Test initialization with valid modes."""
    for mode in VALID_MODES:
        tempo_shift = TempoShift(max_shift=10, mode=mode)
        assert tempo_shift.mode.__name__ == f"_{mode}"


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
    with pytest.raises(ValueError, match=r"Lower range of tempo must be >=0.*"):
        TempoShift(max_shift=10, tempo_range=(-1, 200))
    
    with pytest.raises(ValueError, match=r"min_tempo must be less than max_tempo.*"):
        TempoShift(max_shift=10, tempo_range=(200, 200))
    
    with pytest.raises(ValueError, match=r"min_tempo must be less than max_tempo.*"):
        TempoShift(max_shift=10, tempo_range=(200, 100))
    
    with pytest.raises(ValueError, match=r"tempo_range must be a tuple or list.*"):
        TempoShift(max_shift=10, tempo_range=100)


def test_apply_with_tempo_both_mode():
    """Test applying tempo shift in 'both' mode."""
    midi_data = create_mock_midi_with_tempo()
    tempo_shift = TempoShift(max_shift=10, mode='both', p=1.0)
    
    modified_midi = tempo_shift.apply(midi_data)
    
    # Check if tempo event exists and is modified
    tempo_events = [msg for msg in modified_midi.tracks[0] if msg.type == 'set_tempo']
    assert len(tempo_events) == 1
    assert isinstance(tempo_events[0], MetaMessage)
    assert tempo_events[0].type == 'set_tempo'
    assert 30 <= (6e7 / tempo_events[0].tempo) <= 200  # Check if tempo is within range


def test_apply_with_tempo_up_mode():
    """Test applying tempo shift in 'up' mode."""
    midi_data = create_mock_midi_with_tempo(tempo=500000)  # 120 BPM
    tempo_shift = TempoShift(max_shift=10, mode='up', p=1.0)
    
    modified_midi = tempo_shift.apply(midi_data)
    
    # Check if tempo is increased
    tempo_events = [msg for msg in modified_midi.tracks[0] if msg.type == 'set_tempo']
    assert len(tempo_events) == 1
    new_tempo_bpm = 6e7 / tempo_events[0].tempo
    assert new_tempo_bpm >= 120  # Should be higher than original
    assert 30 <= new_tempo_bpm <= 200  # Within valid range


def test_apply_with_tempo_down_mode():
    """Test applying tempo shift in 'down' mode."""
    midi_data = create_mock_midi_with_tempo(tempo=500000)  # 120 BPM
    tempo_shift = TempoShift(max_shift=10, mode='down', p=1.0)
    
    modified_midi = tempo_shift.apply(midi_data)
    
    # Check if tempo is decreased
    tempo_events = [msg for msg in modified_midi.tracks[0] if msg.type == 'set_tempo']
    assert len(tempo_events) == 1
    new_tempo_bpm = 6e7 / tempo_events[0].tempo
    assert new_tempo_bpm <= 120  # Should be lower than original
    assert 30 <= new_tempo_bpm <= 200  # Within valid range


def test_apply_without_tempo():
    """Test applying tempo shift to MIDI without tempo event."""
    midi_data = create_mock_midi_without_tempo()
    tempo_shift = TempoShift(max_shift=10, mode='both', p=1.0)
    
    modified_midi = tempo_shift.apply(midi_data)
    
    # Check if default tempo (120 BPM) is used and modified
    tempo_events = [msg for msg in modified_midi.tracks[0] if msg.type == 'set_tempo']
    assert len(tempo_events) == 1
    new_tempo_bpm = 6e7 / tempo_events[0].tempo
    assert 30 <= new_tempo_bpm <= 200  # Within valid range


def test_tempo_range_clipping():
    """Test that tempo stays within specified range."""
    midi_data = create_mock_midi_with_tempo(tempo=500000)  # 120 BPM
    tempo_shift = TempoShift(max_shift=1000, mode='both', tempo_range=(50, 180), p=1.0)
    
    modified_midi = tempo_shift.apply(midi_data)
    
    # Check if tempo is clipped to range
    tempo_events = [msg for msg in modified_midi.tracks[0] if msg.type == 'set_tempo']
    assert len(tempo_events) == 1
    new_tempo_bpm = 6e7 / tempo_events[0].tempo
    # Allow for small floating-point differences (within 0.001 BPM)
    assert 49.999 <= new_tempo_bpm <= 180.001


def test_multiple_tempo_events():
    """Test handling of multiple tempo events in MIDI file when respect_tempo_shifts is False."""
    midi_data = MidiFile()
    track = MidiTrack()
    midi_data.tracks.append(track)
    
    # Add multiple tempo events
    track.append(MetaMessage('set_tempo', tempo=500000, time=0))  # 120 BPM
    track.append(Message('note_on', note=60, velocity=64, time=480))
    track.append(MetaMessage('set_tempo', tempo=400000, time=0))  # 150 BPM
    track.append(Message('note_off', note=60, velocity=64, time=480))
    
    tempo_shift = TempoShift(max_shift=10, mode='both', p=1.0, respect_tempo_shifts=False)
    modified_midi = tempo_shift.apply(midi_data)
    
    # Check that all tempo events are replaced with a single tempo event
    tempo_events = [msg for msg in modified_midi.tracks[0] if msg.type == 'set_tempo']
    assert len(tempo_events) == 1
    new_tempo_bpm = 6e7 / tempo_events[0].tempo
    assert 30 <= new_tempo_bpm <= 200


def test_no_change_when_probability_fails(monkeypatch):
    """Test that tempo remains unchanged when random probability check fails."""
    # Mock numpy's random to always return 0.3 (less than p=0.5, so no change)
    def mock_random():
        return 0.3
    monkeypatch.setattr(np.random, 'random', mock_random)
    
    original_tempo = 500000  # 120 BPM
    midi_data = create_mock_midi_with_tempo(tempo=original_tempo)
    tempo_shift = TempoShift(max_shift=10, mode='both', p=0.5)  # 50% chance of change
    
    modified_midi = tempo_shift.apply(midi_data)
    
    # Check that tempo event exists and is unchanged
    tempo_events = [msg for msg in modified_midi.tracks[0] if msg.type == 'set_tempo']
    assert len(tempo_events) == 1
    assert tempo_events[0].tempo == original_tempo  # Should be exactly the same


def test_change_when_probability_succeeds(monkeypatch):
    """Test that tempo changes when random probability check succeeds."""
    # Mock numpy's random to always return 0.7 (greater than p=0.5, so should change)
    def mock_random():
        return 0.7
    monkeypatch.setattr(np.random, 'random', mock_random)
    
    original_tempo = 500000  # 120 BPM
    midi_data = create_mock_midi_with_tempo(tempo=original_tempo)
    tempo_shift = TempoShift(max_shift=10, mode='both', p=0.5)  # 50% chance of change
    
    modified_midi = tempo_shift.apply(midi_data)
    
    # Check that tempo event exists and is changed
    tempo_events = [msg for msg in modified_midi.tracks[0] if msg.type == 'set_tempo']
    assert len(tempo_events) == 1
    assert tempo_events[0].tempo != original_tempo  # Should be different


def test_multiple_tempo_events_with_respect_true(monkeypatch):
    """Test that all tempo events are preserved and shifted when respect_tempo_shifts is True."""
    # Mock numpy's random to always return 0.7 (greater than p=0.5, so should change)
    def mock_random():
        return 0.7
    monkeypatch.setattr(np.random, 'random', mock_random)
    
    midi_data = MidiFile()
    track = MidiTrack()
    midi_data.tracks.append(track)
    
    # Add multiple tempo events with different timings
    track.append(MetaMessage('set_tempo', tempo=500000, time=0))  # 120 BPM
    track.append(Message('note_on', note=60, velocity=64, time=480))
    track.append(MetaMessage('set_tempo', tempo=400000, time=0))  # 150 BPM
    track.append(Message('note_off', note=60, velocity=64, time=480))
    track.append(MetaMessage('set_tempo', tempo=600000, time=0))  # 100 BPM
    
    tempo_shift = TempoShift(max_shift=10, mode='both', p=0.5, respect_tempo_shifts=True)
    modified_midi = tempo_shift.apply(midi_data)
    
    # Check that all tempo events are preserved with their timings
    tempo_events = [msg for msg in modified_midi.tracks[0] if msg.type == 'set_tempo']
    assert len(tempo_events) == 3
    
    # Check that all tempos have been shifted (since random > p)
    original_tempos = [500000, 400000, 600000]
    for orig_tempo, new_event in zip(original_tempos, tempo_events):
        assert new_event.tempo != orig_tempo
        new_tempo_bpm = 6e7 / new_event.tempo
        assert 30 <= new_tempo_bpm <= 200


def test_multiple_tempo_events_with_respect_false(monkeypatch):
    """Test that only first tempo is used when respect_tempo_shifts is False."""
    # Mock numpy's random to always return 0.7 (greater than p=0.5, so should change)
    def mock_random():
        return 0.7
    monkeypatch.setattr(np.random, 'random', mock_random)
    
    midi_data = MidiFile()
    track = MidiTrack()
    midi_data.tracks.append(track)
    
    # Add multiple tempo events with different timings
    track.append(MetaMessage('set_tempo', tempo=500000, time=0))  # 120 BPM
    track.append(Message('note_on', note=60, velocity=64, time=480))
    track.append(MetaMessage('set_tempo', tempo=400000, time=0))  # 150 BPM
    track.append(Message('note_off', note=60, velocity=64, time=480))
    track.append(MetaMessage('set_tempo', tempo=600000, time=0))  # 100 BPM
    
    tempo_shift = TempoShift(max_shift=10, mode='both', p=0.5, respect_tempo_shifts=False)
    modified_midi = tempo_shift.apply(midi_data)
    
    # Check that only one tempo event exists at the start
    tempo_events = [msg for msg in modified_midi.tracks[0] if msg.type == 'set_tempo']
    assert len(tempo_events) == 1
    assert tempo_events[0].time == 0


def test_tempo_at_boundaries():
    """Test tempo shifting at the boundary values of the tempo range."""
    # Test at minimum tempo (30 BPM)
    midi_data = create_mock_midi_with_tempo(tempo=int(6e7/30))  # 30 BPM
    tempo_shift = TempoShift(max_shift=10, mode='down', p=0.0)  # Force shift
    modified_midi = tempo_shift.apply(midi_data)
    tempo_events = [msg for msg in modified_midi.tracks[0] if msg.type == 'set_tempo']
    assert 6e7/tempo_events[0].tempo >= 30  # Should not go below minimum

    # Test at maximum tempo (200 BPM)
    midi_data = create_mock_midi_with_tempo(tempo=int(6e7/200))  # 200 BPM
    tempo_shift = TempoShift(max_shift=10, mode='up', p=0.0)  # Force shift
    modified_midi = tempo_shift.apply(midi_data)
    tempo_events = [msg for msg in modified_midi.tracks[0] if msg.type == 'set_tempo']
    assert 6e7/tempo_events[0].tempo <= 200  # Should not go above maximum


def test_non_zero_time_tempo_events():
    """Test handling of tempo events with non-zero time values."""
    midi_data = MidiFile()
    track = MidiTrack()
    midi_data.tracks.append(track)
    
    # Add tempo events with different time values
    track.append(MetaMessage('set_tempo', tempo=500000, time=100))
    track.append(MetaMessage('set_tempo', tempo=400000, time=200))
    
    tempo_shift = TempoShift(max_shift=10, mode='both', respect_tempo_shifts=True)
    modified_midi = tempo_shift.apply(midi_data)
    
    # Check that time values are preserved
    tempo_events = [msg for msg in modified_midi.tracks[0] if msg.type == 'set_tempo']
    assert len(tempo_events) == 2
    assert tempo_events[0].time == 100
    assert tempo_events[1].time == 200


def test_multiple_tracks_with_tempo():
    """Test handling of MIDI files with multiple tracks containing tempo events."""
    midi_data = MidiFile()
    track1 = MidiTrack()
    track2 = MidiTrack()
    midi_data.tracks.extend([track1, track2])
    
    # Add tempo events to both tracks
    track1.append(MetaMessage('set_tempo', tempo=500000, time=0))
    track2.append(MetaMessage('set_tempo', tempo=400000, time=0))
    
    tempo_shift = TempoShift(max_shift=10, mode='both')
    modified_midi = tempo_shift.apply(midi_data)
    
    # Check that tempo events in first track are modified
    track1_tempos = [msg for msg in modified_midi.tracks[0] if msg.type == 'set_tempo']
    assert len(track1_tempos) == 1
    # Tempo events in other tracks remain unchanged
    track2_tempos = [msg for msg in modified_midi.tracks[1] if msg.type == 'set_tempo']
    assert len(track2_tempos) == 1
    assert track2_tempos[0].tempo == 400000  # Original tempo preserved


def test_empty_tracks():
    """Test handling of MIDI files with empty tracks."""
    midi_data = MidiFile()
    track1 = MidiTrack()
    track2 = MidiTrack()
    midi_data.tracks.extend([track1, track2])
    
    tempo_shift = TempoShift(max_shift=10, mode='both')
    modified_midi = tempo_shift.apply(midi_data)
    
    # Check that a default tempo is added to first track
    track1_tempos = [msg for msg in modified_midi.tracks[0] if msg.type == 'set_tempo']
    assert len(track1_tempos) == 1
    assert 30 <= (6e7 / track1_tempos[0].tempo) <= 200


def test_empty_midi():
    """Test handling of empty MIDI file."""
    midi_data = MidiFile()
    tempo_shift = TempoShift(max_shift=10, mode='both')
    modified_midi = tempo_shift.apply(midi_data)
    
    # Should return unmodified MIDI file
    assert len(modified_midi.tracks) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 
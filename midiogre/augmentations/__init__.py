"""MIDI data augmentation transforms.

This module provides a collection of transforms for augmenting MIDI data. These transforms
can be used individually or composed together to create complex augmentation pipelines.

Available Transforms:
    - DurationShift: Randomly modify note durations while keeping onset times intact
    - NoteAdd: Randomly add new notes to instrument tracks
    - NoteDelete: Randomly remove notes from instrument tracks
    - OnsetTimeShift: Randomly shift note onset times while preserving durations
    - PitchShift: Randomly transpose note pitches up or down
    - TempoShift: Randomly modify tempo while preserving relative note timing

All transforms follow a consistent interface inherited from BaseMidiTransform:
    - They are callable objects that take a PrettyMIDI object as input
    - They support probabilistic application through the 'p' parameter
    - They handle multi-instrument MIDI files through the 'p_instruments' parameter

Example:
    >>> from midiogre.augmentations import PitchShift, OnsetTimeShift
    >>> from midiogre.core import Compose
    >>> import pretty_midi
    >>> 
    >>> # Create transform pipeline
    >>> transform = Compose([
    ...     PitchShift(max_shift=2, p=0.5),
    ...     OnsetTimeShift(max_shift=0.1, p=0.3)
    ... ])
    >>> 
    >>> # Load and transform MIDI file
    >>> midi_data = pretty_midi.PrettyMIDI('song.mid')
    >>> transformed = transform(midi_data)

See Also:
    midiogre.core.transforms_interface: Base class and interface for all transforms
    midiogre.core.compositions: Tools for composing multiple transforms
"""

from .duration_shift import DurationShift
from .note_add import NoteAdd
from .note_delete import NoteDelete
from .onset_time_shift import OnsetTimeShift
from .pitch_shift import PitchShift
from .tempo_shift import TempoShift

__all__ = [
    'DurationShift',
    'NoteAdd',
    'NoteDelete',
    'OnsetTimeShift',
    'PitchShift',
    'TempoShift'
]

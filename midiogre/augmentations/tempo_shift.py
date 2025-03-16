"""MIDI tempo augmentation module.

This module provides functionality to modify the tempo of MIDI files while preserving
the relative timing of notes. The tempo can be shifted up or down within specified
bounds, and the transform can either respect existing tempo changes or apply a single
global tempo.

.. image:: ../../demo/plots/temposhift.png
   :alt: Visualization of TempoShift transform
   :align: center
   :width: 100%
   :class: transform-viz

Note:
    This transform operates on Mido MidiFile objects. If you have a PrettyMIDI object,
    you must first convert it using MIDIOgre's converters:
    >>> from midiogre.core.conversions import ConvertToMido
    >>> midi_data = ConvertToMido()(pretty_midi_obj)

Example:
    >>> from midiogre.augmentations import TempoShift
    >>> from midiogre.core.conversions import ConvertToMido, ConvertToPrettyMIDI
    >>> 
    >>> # Create transform that can increase or decrease tempo by up to 20 BPM
    >>> transform = TempoShift(
    ...     max_shift=20.0,
    ...     mode='both',
    ...     tempo_range=(60.0, 180.0),
    ...     p=0.8
    ... )
    >>> 
    >>> # Load and transform MIDI file
    >>> midi_data = ConvertToMido()('song.mid')  # Convert to Mido format
    >>> transformed = transform(midi_data)  # Apply transform
    >>> pretty_midi_obj = ConvertToPrettyMIDI()(transformed)  # Convert back if needed
"""

import logging
import random

import numpy as np
from mido import MetaMessage

from midiogre.core.transforms_interface import BaseMidiTransform

VALID_MODES = ['both', 'up', 'down']


class TempoShift(BaseMidiTransform):
    """Transform for randomly modifying MIDI tempo.
    
    This transform allows for random modification of MIDI tempo while preserving
    the relative timing of notes. It can either respect all existing tempo changes
    in the file or replace them with a single global tempo.
    
    Note:
        This transform operates on Mido MidiFile objects. If you have a PrettyMIDI object,
        you must first convert it using MIDIOgre's converters:
        >>> from midiogre.core.conversions import ConvertToMido
        >>> midi_data = ConvertToMido()(pretty_midi_obj)
    
    The tempo shift is applied with probability p, and when applied, generates
    random shifts based on the specified mode and maximum shift value. The final
    tempo is always clipped to stay within the specified tempo range.
    
    Args:
        max_shift (float): Maximum value by which tempo can be randomly shifted (in BPM).
            Must be positive.
        mode (str): Direction of tempo shift. One of:
            - 'up': Only increase tempo
            - 'down': Only decrease tempo
            - 'both': Allow both increase and decrease
            Default: 'both'
        tempo_range (tuple[float, float]): Allowed tempo range in BPM as (min_tempo, max_tempo).
            The transformed tempo will be clipped to stay within this range.
            Default: (30.0, 200.0)
        p (float): Probability of applying the tempo shift. Must be in range [0, 1].
            Default: 0.2
        respect_tempo_shifts (bool): If True, preserves all tempo change events,
            shifting each while maintaining their timing. If False, replaces all
            tempo events with a single tempo at the start.
            Default: True
        eps (float, optional): Small epsilon value for numerical stability.
            Default: 1e-12
            
    Raises:
        ValueError: If any of the following conditions are met:
            - mode is not one of 'both', 'up', 'down'
            - max_shift is not positive
            - tempo_range is not a tuple/list of length 2
            - min_tempo is negative
            - min_tempo >= max_tempo
            
    Example:
        >>> from midiogre.core.conversions import ConvertToMido, ConvertToPrettyMIDI
        >>> # Create transform that increases tempo by 10-30 BPM
        >>> transform = TempoShift(
        ...     max_shift=30.0,
        ...     mode='up',
        ...     tempo_range=(60.0, 240.0),
        ...     p=1.0
        ... )
        >>> midi_data = ConvertToMido()('song.mid')  # Convert to Mido format
        >>> transformed = transform(midi_data)  # Apply transform
        >>> pretty_midi_obj = ConvertToPrettyMIDI()(transformed)  # Convert back if needed
    """

    def __init__(self, max_shift: float, mode: str = 'both', tempo_range: (float, float) = (30.0, 200.0),
                 p: float = 0.2, respect_tempo_shifts: bool = True, eps: float = 1e-12):
        """Initialize the TempoShift transform.
        
        Args:
            max_shift (float): Maximum value by which tempo can be randomly shifted (in BPM).
            mode (str, optional): Direction of tempo shift ('up', 'down', or 'both').
                Default: 'both'
            tempo_range (tuple[float, float], optional): Allowed tempo range in BPM.
                Default: (30.0, 200.0)
            p (float, optional): Probability of applying the transform.
                Default: 0.2
            respect_tempo_shifts (bool, optional): Whether to preserve multiple tempo events.
                Default: True
            eps (float, optional): Small epsilon value for numerical stability.
                Default: 1e-12
                
        Raises:
            ValueError: If parameters are invalid (see class docstring for details).
        """
        super().__init__(p_instruments=1.0, p=p, eps=eps)

        if mode not in VALID_MODES:
            raise ValueError(
                f"Mode must be one of {VALID_MODES}, got {mode}"
            )

        if max_shift <= 0:
            raise ValueError(
                f"max_shift must be positive, got {max_shift}"
            )

        if not isinstance(tempo_range, (tuple, list)) or len(tempo_range) != 2:
            raise ValueError(
                f"tempo_range must be a tuple or list of length 2, got {tempo_range}"
            )

        min_tempo, max_tempo = tempo_range
        if min_tempo < 0:
            raise ValueError(
                f"min_tempo must be positive, got {min_tempo}"
            )
        
        if min_tempo >= max_tempo:
            raise ValueError(
                f"min_tempo must be less than max_tempo, got {min_tempo} >= {max_tempo}"
            )

        self.max_shift = max_shift
        self.tempo_range = tempo_range
        self.respect_tempo_shifts = respect_tempo_shifts
        self.mode = mode

    def _generate_shifts(self, num_shifts: int) -> np.ndarray:
        """Generate random tempo shifts based on the configured mode.
        
        Args:
            num_shifts (int): Number of shifts to generate.
            
        Returns:
            np.ndarray: Array of random shifts in BPM. Shape: (num_shifts,)
            
        Note:
            The shifts are generated uniformly within the range determined by
            self.mode and self.max_shift:
            - 'up': [0, max_shift]
            - 'down': [-max_shift, 0]
            - 'both': [-max_shift, max_shift]
        """
        if num_shifts == 0:
            return np.array([])
            
        if self.mode == 'up':
            return np.random.uniform(0, self.max_shift, num_shifts)
        elif self.mode == 'down':
            return np.random.uniform(-self.max_shift, 0, num_shifts)
        else:  # both
            return np.random.uniform(-self.max_shift, self.max_shift, num_shifts)

    def _convert_tempo_to_bpm(self, tempo_microseconds_per_beat: int) -> float:
        """Convert tempo from microseconds per beat to BPM.
        
        Args:
            tempo_microseconds_per_beat (int): Tempo in microseconds per beat.
            
        Returns:
            float: Tempo in beats per minute (BPM).
            
        Note:
            The conversion formula is: BPM = 60,000,000 / microseconds_per_beat
        """
        return 6e7 / tempo_microseconds_per_beat

    def _convert_bpm_to_tempo(self, bpm: float) -> int:
        """Convert BPM to tempo in microseconds per beat.
        
        Args:
            bpm (float): Tempo in beats per minute.
            
        Returns:
            int: Tempo in microseconds per beat, rounded to nearest integer.
            
        Note:
            The conversion formula is: microseconds_per_beat = 60,000,000 / BPM
        """
        return int(round(6e7 / bpm))

    def apply(self, midi_data):
        """Apply the tempo shift transformation to the MIDI data.
        
        This method handles several cases:
        1. Empty MIDI file (returns unchanged)
        2. No tempo events (adds default 120 BPM)
        3. Single tempo event
        4. Multiple tempo events (based on respect_tempo_shifts)
        
        Args:
            midi_data (mido.MidiFile): The MIDI data to transform.
            
        Returns:
            mido.MidiFile: The transformed MIDI data with modified tempo(s).
            
        Note:
            - If no tempo events are found, a default tempo of 120 BPM is used.
            - Only tempo events in the first track are processed.
            - When respect_tempo_shifts is True, all tempo events maintain their
              relative timing but get new tempo values.
            - When respect_tempo_shifts is False, all tempo events are replaced
              with a single tempo event at the start.
            - The transform is applied with probability self.p. If not applied,
              the original tempo(s) are preserved.
        """
        if not midi_data.tracks:
            logging.warning("Empty MIDI file provided")
            return midi_data

        # Find all tempo events in first track
        tempo_events = []
        tempo_events_idx = []
        for idx, event in enumerate(midi_data.tracks[0]):
            if event.type == 'set_tempo':
                tempo_events.append(event)
                tempo_events_idx.append(idx)

        # Handle case with no tempo events
        if not tempo_events:
            logging.warning("No tempo metadata found in MIDI file; assuming a default value of 120 BPM.")
            default_bpm = 120.0
            should_change = np.random.random() < self.p
            
            if should_change:
                shifts = self._generate_shifts(1)
                new_bpm = np.clip(default_bpm + shifts[0], self.tempo_range[0], self.tempo_range[1])
                new_tempo = self._convert_bpm_to_tempo(new_bpm)
            else:
                new_tempo = self._convert_bpm_to_tempo(default_bpm)
                
            midi_data.tracks[0].insert(0, MetaMessage(type="set_tempo", tempo=new_tempo, time=0))
            return midi_data

        # Remove existing tempo events (in reverse order to maintain indices)
        for idx in reversed(tempo_events_idx):
            midi_data.tracks[0].pop(idx)

        # Determine if we should apply changes
        should_change = np.random.random() < self.p

        if self.respect_tempo_shifts:
            # Process all tempo events while maintaining timing
            if should_change:
                shifts = self._generate_shifts(len(tempo_events))
                for i, event in enumerate(tempo_events):
                    current_bpm = self._convert_tempo_to_bpm(event.tempo)
                    new_bpm = np.clip(current_bpm + shifts[i], self.tempo_range[0], self.tempo_range[1])
                    new_tempo = self._convert_bpm_to_tempo(new_bpm)
                    midi_data.tracks[0].append(
                        MetaMessage(type="set_tempo", tempo=new_tempo, time=event.time)
                    )
            else:
                # Keep original tempos
                for event in tempo_events:
                    midi_data.tracks[0].append(
                        MetaMessage(type="set_tempo", tempo=event.tempo, time=event.time)
                    )
        else:
            # Use only first tempo event and place at start
            current_bpm = self._convert_tempo_to_bpm(tempo_events[0].tempo)
            if should_change:
                shifts = self._generate_shifts(1)
                new_bpm = np.clip(current_bpm + shifts[0], self.tempo_range[0], self.tempo_range[1])
                new_tempo = self._convert_bpm_to_tempo(new_bpm)
            else:
                new_tempo = tempo_events[0].tempo
                
            midi_data.tracks[0].insert(0, MetaMessage(type="set_tempo", tempo=new_tempo, time=0))

        return midi_data

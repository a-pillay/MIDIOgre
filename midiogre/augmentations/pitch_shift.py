"""MIDI pitch augmentation module.

This module provides functionality to randomly transpose MIDI notes up or down
while preserving their timing and velocity. The pitch shifts are applied on a
per-note basis, allowing for complex harmonic variations.

Example:
    >>> from midiogre.augmentations import PitchShift
    >>> import pretty_midi
    >>> 
    >>> # Create transform that shifts pitches up or down by up to 2 semitones
    >>> transform = PitchShift(
    ...     max_shift=2,
    ...     mode='both',
    ...     p=0.5
    ... )
    >>> 
    >>> # Load and transform MIDI file
    >>> midi_data = pretty_midi.PrettyMIDI('song.mid')
    >>> transformed = transform(midi_data)
"""

import logging
import random

import numpy as np

from midiogre.core.transforms_interface import BaseMidiTransform

VALID_MODES = ['both', 'up', 'down']


class PitchShift(BaseMidiTransform):
    """Transform for randomly transposing MIDI note pitches.
    
    This transform allows for random transposition of MIDI notes while preserving
    their timing and velocity. Each selected note can be shifted up or down by
    a random amount within the specified range. The shifts are applied on a
    per-note basis, allowing for complex harmonic variations.
    
    The pitch shift is applied with probability p to each selected instrument,
    and within each instrument, to a random subset of notes determined by p.
    All shifts are automatically clipped to stay within the valid MIDI note
    range [0, 127].

    .. image:: ../../demo/plots/pitchshift.png
       :alt: Visualization of PitchShift transform
       :align: center
       :width: 100%
       :class: transform-viz
    
    Args:
        max_shift (int): Maximum number of semitones by which a note pitch can
            be shifted. Must be in range [0, 127].
        mode (str): Direction of pitch shift. One of:
            - 'up': Only shift pitches up
            - 'down': Only shift pitches down
            - 'both': Allow both up and down shifts
            Default: 'both'
        p_instruments (float): If a MIDI file has multiple instruments, this
            determines the probability of applying the transform to each
            instrument. Must be in range [0, 1].
            Default: 1.0 (apply to all instruments)
        p (float): For each selected instrument, this determines the probability
            of shifting each note. Must be in range [0, 1].
            Default: 0.2
        eps (float, optional): Small epsilon value for numerical stability.
            Default: 1e-12
            
    Raises:
        ValueError: If any of the following conditions are met:
            - max_shift is not in range [0, 127]
            - mode is not one of 'both', 'up', 'down'
            - p_instruments is not in range [0, 1]
            - p is not in range [0, 1]
            
    Example:
        >>> # Create transform that only shifts pitches up by 1-3 semitones
        >>> transform = PitchShift(
        ...     max_shift=3,
        ...     mode='up',
        ...     p_instruments=1.0,
        ...     p=0.5
        ... )
        >>> transformed = transform(midi_data)
    """

    def __init__(self, max_shift: int, mode: str = 'both', p_instruments: float = 1.0,
                 p: float = 0.2, eps: float = 1e-12):
        """Initialize the PitchShift transform.
        
        Args:
            max_shift (int): Maximum number of semitones for pitch shifts.
            mode (str, optional): Direction of pitch shift ('up', 'down', or 'both').
                Default: 'both'
            p_instruments (float, optional): Probability of applying to each instrument.
                Default: 1.0
            p (float, optional): Probability of shifting each note.
                Default: 0.2
            eps (float, optional): Small epsilon value for numerical stability.
                Default: 1e-12
                
        Raises:
            ValueError: If parameters are invalid (see class docstring for details).
        """
        super().__init__(p_instruments=p_instruments, p=p, eps=eps)

        if not 0 <= max_shift <= 127:
            raise ValueError(
                f"MIDI notes cannot be shifted by more than 127, got {max_shift}"
            )

        if mode not in VALID_MODES:
            raise ValueError(
                f"Mode must be one of {VALID_MODES}, got {mode}"
            )

        self.max_shift = max_shift
        self.mode = mode

    def _generate_shifts(self, num_shifts: int) -> np.ndarray:
        """Generate random pitch shifts based on the configured mode.
        
        Args:
            num_shifts (int): Number of shifts to generate.
            
        Returns:
            np.ndarray: Array of random shifts in semitones. Shape: (num_shifts,)
            
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
            return np.random.randint(0, self.max_shift + 1, num_shifts)
        elif self.mode == 'down':
            return np.random.randint(-self.max_shift, 1, num_shifts)
        else:  # both
            return np.random.randint(-self.max_shift, self.max_shift + 1, num_shifts)

    def apply(self, midi_data):
        """Apply the pitch shift transformation to the MIDI data.
        
        For each non-drum instrument selected based on p_instruments, this method:
        1. Randomly selects a subset of notes based on p
        2. Generates random pitch shifts based on mode and max_shift
        3. Applies the shifts while clipping to valid MIDI note range [0, 127]
        
        Args:
            midi_data (pretty_midi.PrettyMIDI): The MIDI data to transform.
            
        Returns:
            pretty_midi.PrettyMIDI: The transformed MIDI data with shifted pitches.
            
        Note:
            - Drum instruments are skipped by default
            - The transform maintains the original timing and velocity of all notes
            - Notes are shifted independently, allowing for complex harmonic variations
        """
        modified_instruments = self._get_modified_instruments_list(midi_data)
        for instrument in modified_instruments:
            if not instrument.notes:
                continue
                
            num_notes_to_shift = int(self.p * len(instrument.notes))
            if num_notes_to_shift == 0:
                logging.debug(
                    "PitchShift can't be performed on 0 notes on given non-drum instrument. Skipping.",
                )
                continue

            # Select notes to modify
            notes_to_modify = random.sample(instrument.notes, k=num_notes_to_shift)
            
            # Get current pitches
            current_pitches = np.array([note.pitch for note in notes_to_modify])
            
            # Generate and apply shifts
            shifts = self._generate_shifts(num_notes_to_shift)
            new_pitches = np.clip(current_pitches + shifts, 0, 127)
            
            # Update notes with new pitches
            for note, new_pitch in zip(notes_to_modify, new_pitches):
                note.pitch = int(new_pitch)
                
        return midi_data

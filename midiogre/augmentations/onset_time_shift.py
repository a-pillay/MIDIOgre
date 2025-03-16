"""MIDI onset time augmentation module.

This module provides functionality to randomly shift the onset times of MIDI notes
while preserving their durations. The shifts can be applied to move notes earlier
or later in time, allowing for rhythmic variations while maintaining note lengths.

.. image:: ../../demo/plots/onsettimeshift.png
   :alt: Visualization of OnsetTimeShift transform
   :align: center
   :width: 100%
   :class: transform-viz

Example:
    >>> from midiogre.augmentations import OnsetTimeShift
    >>> import pretty_midi
    >>> 
    >>> # Create transform that shifts note timings by up to 0.1 seconds
    >>> transform = OnsetTimeShift(
    ...     max_shift=0.1,
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

VALID_MODES = ['both', 'left', 'right']


class OnsetTimeShift(BaseMidiTransform):
    """Transform for randomly shifting MIDI note onset times.
    
    This transform allows for random modification of note start times while
    preserving their durations. Each selected note can be shifted earlier or
    later in time by a random amount within the specified range. The shifts
    are applied on a per-note basis, allowing for complex rhythmic variations.
    
    The onset shift is applied with probability p to each selected instrument,
    and within each instrument, to a random subset of notes determined by p.
    All shifts are automatically clipped to ensure notes stay within valid time
    ranges (no negative start times, no extending beyond track end).
    
    Args:
        max_shift (float): Maximum time in seconds by which a note onset can
            be shifted. Must be positive.
        mode (str): Direction of time shift. One of:
            - 'left': Only shift notes earlier (ie, advance) in time
            - 'right': Only shift notes later (ie, delay) in time
            - 'both': Allow both earlier and later shifts
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
            - max_shift is not positive
            - mode is not one of 'both', 'left', 'right'
            - p_instruments is not in range [0, 1]
            - p is not in range [0, 1]
            
    Example:
        >>> # Create transform that only delays notes by 0.05-0.15 seconds
        >>> transform = OnsetTimeShift(
        ...     max_shift=0.15,
        ...     mode='right',
        ...     p_instruments=1.0,
        ...     p=0.3
        ... )
        >>> transformed = transform(midi_data)
    """

    def __init__(self, max_shift: float, mode: str = 'both', p_instruments: float = 1.0,
                 p: float = 0.2, eps: float = 1e-12):
        """Initialize the OnsetTimeShift transform.
        
        Args:
            max_shift (float): Maximum time in seconds for onset shifts.
            mode (str, optional): Direction of time shift ('left', 'right', or 'both').
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

        if mode not in VALID_MODES:
            raise ValueError(
                f"Mode must be one of {VALID_MODES}, got {mode}"
            )

        if max_shift < 0:
            raise ValueError(
                f"max_shift must be positive, got {max_shift}"
            )

        self.max_shift = max_shift
        self.mode = mode

    def _generate_shifts(self, num_shifts: int) -> np.ndarray:
        """Generate random onset time shifts based on the configured mode.
        
        Args:
            num_shifts (int): Number of shifts to generate.
            
        Returns:
            np.ndarray: Array of random shifts in seconds. Shape: (num_shifts,)
            
        Note:
            The shifts are generated uniformly within the range determined by
            self.mode and self.max_shift:
            - 'left': [-max_shift, 0] (earlier in time)
            - 'right': [0, max_shift] (later in time)
            - 'both': [-max_shift, max_shift] (either direction)
        """
        if num_shifts == 0:
            return np.array([])
            
        if self.mode == 'left':
            return np.random.uniform(-self.max_shift, 0, num_shifts)
        elif self.mode == 'right':
            return np.random.uniform(0, self.max_shift, num_shifts)
        else:  # both
            return np.random.uniform(-self.max_shift, self.max_shift, num_shifts)

    def apply(self, midi_data):
        """Apply the onset time shift transformation to the MIDI data.
        
        For each non-drum instrument selected based on p_instruments, this method:
        1. Randomly selects a subset of notes based on p
        2. Generates random time shifts based on mode and max_shift
        3. Applies the shifts while maintaining note durations and ensuring valid times
        
        Args:
            midi_data (pretty_midi.PrettyMIDI): The MIDI data to transform.
            
        Returns:
            pretty_midi.PrettyMIDI: The transformed MIDI data with shifted note timings.
            
        Note:
            - Drum instruments are skipped by default
            - The transform maintains the duration of all notes
            - Notes are shifted independently, allowing for complex rhythmic variations
            - Shifts are clipped to ensure notes stay within valid time ranges:
                - No negative start times
                - No extending beyond the end of the track
        """
        modified_instruments = self._get_modified_instruments_list(midi_data)
        for instrument in modified_instruments:
            if not instrument.notes:
                continue
                
            num_notes_to_shift = int(self.p * len(instrument.notes))
            if num_notes_to_shift == 0:
                logging.debug(
                    "OnsetTimeShift can't be performed on 0 notes on given non-drum instrument. Skipping.",
                )
                continue

            instrument_end_time = instrument.notes[-1].end
            
            # Select notes to modify
            notes_to_modify = random.sample(instrument.notes, k=num_notes_to_shift)
            
            # Get current onsets and durations
            onsets = np.array([note.start for note in notes_to_modify])
            durations = np.array([note.end - note.start for note in notes_to_modify])
            
            # Generate and apply shifts
            shifts = self._generate_shifts(num_notes_to_shift)
            new_onsets = np.clip(onsets + shifts, 0, instrument_end_time)
            new_offsets = new_onsets + durations
            
            # Update notes with new times
            for note, new_onset, new_offset in zip(notes_to_modify, new_onsets, new_offsets):
                note.start = float(new_onset)
                note.end = float(new_offset)
                
        return midi_data

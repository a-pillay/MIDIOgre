"""MIDI note duration augmentation module.

This module provides functionality to randomly modify the durations of MIDI notes
while preserving their onset times. The durations can be extended or shortened,
allowing for articulation variations while maintaining rhythmic timing.

Example:
    >>> from midiogre.augmentations import DurationShift
    >>> import pretty_midi
    >>> 
    >>> # Create transform that modifies note durations by up to 0.5 seconds
    >>> transform = DurationShift(
    ...     max_shift=0.5,
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

VALID_MODES = ['both', 'shrink', 'extend']


class DurationShift(BaseMidiTransform):
    """Transform for randomly modifying MIDI note durations.
    
    This transform allows for random modification of note durations while
    preserving their onset times. Each selected note can be shortened or
    lengthened by a random amount within the specified range. The shifts
    are applied on a per-note basis, allowing for varied articulation.
    
    The duration shift is applied with probability p to each selected instrument,
    and within each instrument, to a random subset of notes determined by p.
    All shifts are automatically clipped to ensure notes maintain a minimum
    duration and don't extend beyond the track end.

    .. image:: ../../demo/plots/durationshift.png
       :alt: Visualization of DurationShift transform
       :align: center
       :width: 100%
       :class: transform-viz
    
    Args:
        max_shift (float): Maximum time in seconds by which a note duration can
            be modified. Must be positive.
        mode (str): Type of duration modification. One of:
            - 'shrink': Only shorten notes
            - 'extend': Only lengthen notes
            - 'both': Allow both shortening and lengthening
            Default: 'both'
        min_duration (float): Minimum allowed note duration in seconds. Notes
            will not be shortened below this value.
            Default: 1e-6
        p_instruments (float): If a MIDI file has multiple instruments, this
            determines the probability of applying the transform to each
            instrument. Must be in range [0, 1].
            Default: 1.0 (apply to all instruments)
        p (float): For each selected instrument, this determines the probability
            of modifying each note. Must be in range [0, 1].
            Default: 0.2
        eps (float, optional): Small epsilon value for numerical stability.
            Default: 1e-12
            
    Raises:
        ValueError: If any of the following conditions are met:
            - max_shift is not positive
            - mode is not one of 'both', 'shrink', 'extend'
            - min_duration is not positive
            - p_instruments is not in range [0, 1]
            - p is not in range [0, 1]
            
    Example:
        >>> # Create transform that only extends notes by 0.1-0.3 seconds
        >>> transform = DurationShift(
        ...     max_shift=0.3,
        ...     mode='extend',
        ...     min_duration=0.1,
        ...     p_instruments=1.0,
        ...     p=0.4
        ... )
        >>> transformed = transform(midi_data)
    """

    def __init__(self, max_shift: float, mode: str = 'both', min_duration: float = 1e-6,
                 p_instruments: float = 1.0, p: float = 0.2, eps: float = 1e-12):
        """Initialize the DurationShift transform.
        
        Args:
            max_shift (float): Maximum time in seconds for duration modifications.
            mode (str, optional): Type of duration modification ('shrink', 'extend', or 'both').
                Default: 'both'
            min_duration (float, optional): Minimum allowed note duration in seconds.
                Default: 1e-6
            p_instruments (float, optional): Probability of applying to each instrument.
                Default: 1.0
            p (float, optional): Probability of modifying each note.
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

        if min_duration < 0:
            raise ValueError(
                f"min_duration must be positive, got {min_duration}"
            )

        if max_shift < 0:
            raise ValueError(
                f"max_shift must be positive, got {max_shift}"
            )

        self.max_shift = max_shift
        self.min_duration = min_duration
        self.mode = mode

    def _generate_shifts(self, num_shifts: int) -> np.ndarray:
        """Generate random duration shifts based on the configured mode.
        
        Args:
            num_shifts (int): Number of shifts to generate.
            
        Returns:
            np.ndarray: Array of random shifts in seconds. Shape: (num_shifts,)
            
        Note:
            The shifts are generated uniformly within the range determined by
            self.mode and self.max_shift:
            - 'shrink': [-max_shift, 0] (shorter duration)
            - 'extend': [0, max_shift] (longer duration)
            - 'both': [-max_shift, max_shift] (either direction)
        """
        if num_shifts == 0:
            return np.array([])
            
        if self.mode == 'shrink':
            return np.random.uniform(-self.max_shift, 0, num_shifts)
        elif self.mode == 'extend':
            return np.random.uniform(0, self.max_shift, num_shifts)
        else:  # both
            return np.random.uniform(-self.max_shift, self.max_shift, num_shifts)

    def apply(self, midi_data):
        """Apply the duration shift transformation to the MIDI data.
        
        For each non-drum instrument selected based on p_instruments, this method:
        1. Randomly selects a subset of notes based on p
        2. Generates random duration shifts based on mode and max_shift
        3. Applies the shifts while maintaining onset times and ensuring valid durations
        
        Args:
            midi_data (pretty_midi.PrettyMIDI): The MIDI data to transform.
            
        Returns:
            pretty_midi.PrettyMIDI: The transformed MIDI data with modified note durations.
            
        Note:
            - Drum instruments are skipped by default
            - The transform maintains the onset time of all notes
            - Notes are modified independently, allowing for varied articulation
            - Duration changes are clipped to ensure:
                - Notes maintain the minimum duration
                - Notes don't extend beyond the end of the track
        """
        modified_instruments = self._get_modified_instruments_list(midi_data)
        for instrument in modified_instruments:
            if not instrument.notes:
                continue
                
            num_notes_to_shift = int(self.p * len(instrument.notes))
            if num_notes_to_shift == 0:
                logging.debug(
                    "DurationShift can't be performed on 0 notes on given non-drum instrument. Skipping.",
                )
                continue

            instrument_end_time = instrument.notes[-1].end
            
            # Select notes to modify
            notes_to_modify = random.sample(instrument.notes, k=num_notes_to_shift)
            
            # Get current onsets and offsets
            onsets = np.array([note.start for note in notes_to_modify])
            offsets = np.array([note.end for note in notes_to_modify])
            
            # Generate and apply shifts
            shifts = self._generate_shifts(num_notes_to_shift)
            new_offsets = np.clip(
                offsets + shifts,
                onsets + self.min_duration,  # Minimum allowed end time
                instrument_end_time  # Maximum allowed end time
            )
            
            # Update notes with new end times
            for note, new_offset in zip(notes_to_modify, new_offsets):
                note.end = float(new_offset)
                
        return midi_data

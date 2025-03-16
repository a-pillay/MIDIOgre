"""MIDI note addition augmentation module.

This module provides functionality to randomly add new notes to MIDI tracks.
The added notes can be configured in terms of their pitch range, velocity range,
and duration range, allowing for controlled density variations in the music.

.. image:: ../../demo/plots/noteadd.png
   :alt: Visualization of NoteAdd transform
   :align: center
   :width: 100%
   :class: transform-viz

Example:
    >>> from midiogre.augmentations import NoteAdd
    >>> import pretty_midi
    >>> 
    >>> # Create transform that adds notes in the middle register
    >>> transform = NoteAdd(
    ...     note_num_range=(48, 72),  # C3 to C5
    ...     note_velocity_range=(60, 100),
    ...     note_duration_range=(0.2, 0.8),
    ...     p=0.3
    ... )
    >>> 
    >>> # Load and transform MIDI file
    >>> midi_data = pretty_midi.PrettyMIDI('song.mid')
    >>> transformed = transform(midi_data)
"""

import math

import numpy as np
from pretty_midi import Note

from midiogre.core.transforms_interface import BaseMidiTransform


class NoteAdd(BaseMidiTransform):
    """Transform for randomly adding new MIDI notes.
    
    This transform allows for random addition of new notes to MIDI tracks.
    Each new note's properties (pitch, velocity, duration, timing) are randomly
    generated within specified ranges. The number of notes added is proportional
    to the existing number of notes in each track.
    
    The note addition is applied with probability p to each selected instrument.
    For each selected instrument, the number of notes to add is randomly chosen
    between 0 and p * (current number of notes).
    
    Args:
        note_num_range (tuple[int, int]): Range of MIDI note numbers for new notes
            as (min_note, max_note). Each value must be in range [0, 127].
            Example: (60, 72) for notes between middle C and C5.
        note_velocity_range (tuple[int, int]): Range of MIDI velocities for new
            notes as (min_velocity, max_velocity). Each value must be in range [0, 127].
            Example: (64, 100) for medium to loud notes.
        note_duration_range (tuple[float, float]): Range of note durations in seconds
            as (min_duration, max_duration). Values must be positive.
            Example: (0.1, 0.5) for short to medium notes.
        restrict_to_instrument_time (bool): If True, new notes will not extend beyond
            the end time of existing notes in the instrument.
            Default: True
        p_instruments (float): If a MIDI file has multiple instruments, this
            determines the probability of applying the transform to each
            instrument. Must be in range [0, 1].
            Default: 1.0 (apply to all instruments)
        p (float): For each selected instrument, this determines the maximum ratio
            of new notes to add relative to existing notes. Must be in range [0, 1].
            Example: If p=0.2 and an instrument has 100 notes, up to 20 new notes
            may be added.
            Default: 0.2
        eps (float, optional): Small epsilon value for numerical stability.
            Default: 1e-12
            
    Raises:
        ValueError: If any of the following conditions are met:
            - note_num_range values not in [0, 127]
            - note_velocity_range values not in [0, 127]
            - note_duration_range values not positive
            - ranges not specified as (min, max) with max > min
            - p_instruments not in range [0, 1]
            - p not in range [0, 1]
            
    Example:
        >>> # Create transform that adds short, quiet notes in the high register
        >>> transform = NoteAdd(
        ...     note_num_range=(72, 96),  # C5 to C7
        ...     note_velocity_range=(20, 40),  # Soft notes
        ...     note_duration_range=(0.05, 0.2),  # Short notes
        ...     restrict_to_instrument_time=True,
        ...     p_instruments=1.0,
        ...     p=0.15
        ... )
        >>> transformed = transform(midi_data)
    """

    def __init__(self, note_num_range: (int, int), note_velocity_range: (int, int), note_duration_range: (int, int),
                 restrict_to_instrument_time: bool = True, p_instruments: float = 1.0, p: float = 0.2,
                 eps: float = 1e-12):
        """Initialize the NoteAdd transform.
        
        Args:
            note_num_range (tuple[int, int]): Range of MIDI note numbers for new notes.
            note_velocity_range (tuple[int, int]): Range of MIDI velocities for new notes.
            note_duration_range (tuple[float, float]): Range of note durations in seconds.
            restrict_to_instrument_time (bool, optional): Whether to limit note end times.
                Default: True
            p_instruments (float, optional): Probability of applying to each instrument.
                Default: 1.0
            p (float, optional): Maximum ratio of new notes to add.
                Default: 0.2
            eps (float, optional): Small epsilon value for numerical stability.
                Default: 1e-12
                
        Raises:
            ValueError: If parameters are invalid (see class docstring for details).
        """
        super().__init__(p_instruments=p_instruments, p=p, eps=eps)

        if len(note_num_range) != 2 or note_num_range[1] < note_num_range[0]:
            raise ValueError(
                "MIDI note numbers must be specified as (min_note_num, max_note_num] where max_note_num > min_note_num."
            )

        if not (0 <= note_num_range[0] < 128) or not (0 <= note_num_range[1] < 128):
            raise ValueError(
                "MIDI note numbers must be integers >=0 and < 128."
            )

        if len(note_velocity_range) != 2 or note_velocity_range[1] < note_velocity_range[0]:
            raise ValueError(
                "MIDI note velocities must be specified as (min_note_velocity, max_note_velocity] where "
                "max_note_velocity > min_note_velocity."
            )

        if not (0 <= note_velocity_range[0] < 128) or not (0 <= note_velocity_range[1] < 128):
            raise ValueError(
                "MIDI note velocities must be integers >=0 and < 128."
            )

        if len(note_duration_range) != 2 or note_duration_range[1] < note_duration_range[0]:
            raise ValueError(
                "MIDI note durations must be specified as (min_note_duration, max_note_duration] where "
                "min_note_duration > max_note_duration."
            )

        self.min_note_num = note_num_range[0]
        self.max_note_num = note_num_range[1]

        self.min_velo = note_velocity_range[0]
        self.max_velo = note_velocity_range[1]

        self.min_durn = note_duration_range[0]
        self.max_durn = note_duration_range[1]

        self.restrict_to_instrument_time = restrict_to_instrument_time

    def __generate_n_midi_notes(self, n: int, instrument_end_time: float) -> list:
        """Generate n random MIDI notes within the configured ranges.
        
        Args:
            n (int): Number of notes to generate.
            instrument_end_time (float): End time of the last note in the instrument.
                Used when restrict_to_instrument_time is True.
                
        Returns:
            list[Note]: List of n randomly generated MIDI notes.
            
        Note:
            Generated notes have:
            - Pitches uniformly distributed in note_num_range
            - Velocities uniformly distributed in note_velocity_range
            - Durations uniformly distributed in note_duration_range
            - Start times uniformly distributed between 0 and instrument_end_time
            - End times clipped to instrument_end_time if restrict_to_instrument_time
        """
        # Vectorized random number generation
        start_times = np.random.uniform(0, instrument_end_time, n)
        pitches = np.random.randint(self.min_note_num, self.max_note_num + 1, n)
        velocities = np.random.randint(self.min_velo, self.max_velo + 1, n)
        durations = np.random.uniform(self.min_durn, self.max_durn, n)
        end_times = np.clip(start_times + durations, None, instrument_end_time)
        
        # Create notes in a list comprehension for better performance
        return [Note(pitch=int(p), velocity=int(v), start=float(s), end=float(e)) 
                for p, v, s, e in zip(pitches, velocities, start_times, end_times)]

    def apply(self, midi_data):
        """Apply the note addition transformation to the MIDI data.
        
        For each non-drum instrument selected based on p_instruments, this method:
        1. Determines the number of notes to add based on p
        2. Generates random notes within the configured ranges
        3. Adds the new notes to the instrument track
        
        Args:
            midi_data (pretty_midi.PrettyMIDI): The MIDI data to transform.
            
        Returns:
            pretty_midi.PrettyMIDI: The transformed MIDI data with added notes.
            
        Note:
            - Drum instruments are skipped by default
            - The number of notes added is random but proportional to existing notes
            - New notes have random properties within the configured ranges
            - If restrict_to_instrument_time is True, new notes won't extend beyond
              the end of existing notes in the track
        """
        modified_instruments = self._get_modified_instruments_list(midi_data)
        for instrument in modified_instruments:
            if not instrument.notes:  # Skip empty instruments
                continue
            num_new_notes = math.ceil(np.random.uniform(self.eps, self.p) * len(instrument.notes))
            if num_new_notes > 0:  # Only generate notes if we need to
                instrument.notes.extend(
                    self.__generate_n_midi_notes(
                        n=num_new_notes,
                        instrument_end_time=instrument.notes[-1].end
                    )
                )
        return midi_data

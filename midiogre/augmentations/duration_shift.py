import logging
import random

import numpy as np

from midiogre.core.transforms_interface import BaseMidiTransform

VALID_MODES = ['both', 'shrink', 'extend']


class DurationShift(BaseMidiTransform):
    def __init__(self, max_shift: float, mode: str = 'both', min_duration=1e-6, p_instruments: float = 1.0,
                 p: float = 0.2, eps: float = 1e-12):
        """
        Randomly modify MIDI note durations while keeping their onset times intact.

        :param max_shift: Maximum value by which a note duration can be randomly shifted.
        :param mode: 'shrink' if notes can only be shrunk (reduced duration), 'extend' if notes can only be extended
        (increased duration), 'both' if notes can be shrunk or extended.
        :param min_duration: The least duration a note can have post shrinkage.
        :param p_instruments: If a MIDI file has >1 instruments, this parameter will determine the percentage of
        instruments that may have random note duration changes.
        :param p: Determines the percentage of notes that may have random duration changes per instrument.
        :param eps: Epsilon term added to represent the lowest possible value (for numerical stability)
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
        """Generate duration shifts in a vectorized manner."""
        if num_shifts == 0:
            return np.array([])
            
        if self.mode == 'shrink':
            return np.random.uniform(-self.max_shift, 0, num_shifts)
        elif self.mode == 'extend':
            return np.random.uniform(0, self.max_shift, num_shifts)
        else:  # both
            return np.random.uniform(-self.max_shift, self.max_shift, num_shifts)

    def apply(self, midi_data):
        """
        Apply the duration shift transformation to the MIDI data.

        Args:
            midi_data: A PrettyMIDI object to transform.

        Returns:
            The transformed PrettyMIDI object.
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

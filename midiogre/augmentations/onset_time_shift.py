import logging
import random

import numpy as np

from midiogre.core.transforms_interface import BaseMidiTransform

VALID_MODES = ['both', 'left', 'right']


class OnsetTimeShift(BaseMidiTransform):
    def __init__(self, max_shift: float, mode: str = 'both', p_instruments: float = 1.0,
                 p: float = 0.2, eps: float = 1e-12):
        """
        Randomly modify MIDI note onset times while keeping their total durations intact.

        :param max_shift: Maximum value by which a note onset time can be randomly shifted.
        :param mode: 'left' if notes can only be advanced, 'right' if notes can only be delayed,
        'both' if notes can be advanced or delayed.
        :param p_instruments: If a MIDI file has >1 instruments, this parameter will determine the percentage of
        instruments that may have random note onset time changes.
        :param p: Determines the percentage of notes that may have random onset time changes per instrument.
        :param eps: Epsilon term added to represent the lowest possible value (for numerical stability)
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
        """Generate onset time shifts in a vectorized manner."""
        if num_shifts == 0:
            return np.array([])
            
        if self.mode == 'left':
            return np.random.uniform(-self.max_shift, 0, num_shifts)
        elif self.mode == 'right':
            return np.random.uniform(0, self.max_shift, num_shifts)
        else:  # both
            return np.random.uniform(-self.max_shift, self.max_shift, num_shifts)

    def apply(self, midi_data):
        """
        Apply the onset time shift transformation to the MIDI data.

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

import logging
import random

import numpy as np

from midiogre.core.transforms_interface import BaseMidiTransform

VALID_MODES = ['both', 'up', 'down']


class PitchShift(BaseMidiTransform):
    def __init__(self, max_shift: int, mode: str = 'both', p_instruments: float = 1.0,
                 p: float = 0.2, eps: float = 1e-12):
        """
        Randomly transpose (pitch shift) MIDI note values.

        :param max_shift: Maximum value by which a note pitch can be randomly shifted.
        :param mode: 'up' if notes can only be transposed up, 'down' if notes can only be transposed down,
        'both' if notes can be transposed up or down.
        :param p_instruments: If a MIDI file has >1 instruments, this parameter will determine the percentage of
        instruments that may have random note transpositions.
        :param p: Determines the percentage of notes that may be randomly transposed per instrument.
        :param eps: Epsilon term added to represent the lowest possible value (for numerical stability)
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
        """Generate pitch shifts in a vectorized manner."""
        if num_shifts == 0:
            return np.array([])
            
        if self.mode == 'up':
            return np.random.randint(0, self.max_shift + 1, num_shifts)
        elif self.mode == 'down':
            return np.random.randint(-self.max_shift, 1, num_shifts)
        else:  # both
            return np.random.randint(-self.max_shift, self.max_shift + 1, num_shifts)

    def apply(self, midi_data):
        """
        Apply the pitch shift transformation to the MIDI data.

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

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
                "MIDI notes cannot be shifted by more than 127."
            )

        if mode not in VALID_MODES:
            raise ValueError(
                "Valid PitchShift modes are: {}.".format(VALID_MODES)
            )

        self.max_shift = max_shift

        if mode == 'up':
            self.mode = self._up
        elif mode == 'down':
            self.mode = self._down
        else:
            self.mode = self._both

    def _both(self, pitch):
        return np.clip(pitch + np.random.randint(-self.max_shift, self.max_shift + 1), 0, 127)

    def _up(self, pitch):
        return np.clip(pitch + np.random.randint(0, self.max_shift + 1), 0, 127)

    def _down(self, pitch):
        return np.clip(pitch + np.random.randint(-self.max_shift, 1), 0, 127)

    def apply(self, midi_data):
        modified_instruments = self._get_modified_instruments_list(midi_data)
        for instrument in modified_instruments:
            num_shifted_notes_per_instrument = int(self.p * len(instrument.notes))
            if num_shifted_notes_per_instrument == 0:
                # TODO Replace with a better warning definition
                logging.debug(
                    "PitchShift can't be performed on 0 notes on given non-drum instrument. Skipping.",
                )
                continue

            for note in random.sample(instrument.notes, k=num_shifted_notes_per_instrument):
                note.pitch = self.mode(note.pitch)
        return midi_data

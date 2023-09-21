import logging
import random

import numpy as np

from core.transforms_interface import BaseMidiTransform

VALID_MODES = ['both', 'up', 'down']


class PitchShift(BaseMidiTransform):
    def __init__(self, max_shift, mode='both', p_instruments=1.0, p=0.2):
        super().__init__(p=p)

        if not 0 <= max_shift <= 127:
            raise ValueError(
                "MIDI notes cannot be shifted by more than 127."
            )

        if mode not in VALID_MODES:
            raise ValueError(
                "Valid pitch shift modes are: {}.".format(VALID_MODES)
            )

        if not 0 <= p <= 1:
            raise ValueError(
                "Probability of applying PitchShift on an instrument must be >=0 and <=1."
            )

        self.max_shift = max_shift

        if mode == 'up':
            self.mode = self._up
        elif mode == 'down':
            self.mode = self._down
        else:
            self.mode = self._both

        self.p_instruments = p_instruments

    def _both(self, pitch):
        return np.clip(pitch + np.random.randint(-self.max_shift, self.max_shift + 1), 0, 127)

    def _up(self, pitch):
        return np.clip(pitch + np.random.randint(0, self.max_shift + 1), 0, 127)

    def _down(self, pitch):
        return np.clip(pitch + np.random.randint(-self.max_shift, 1), 0, 127)

    def apply(self, midi_data):

        modified_instruments = [instrument for instrument in midi_data.instruments if not instrument.is_drum]

        if len(modified_instruments) == 0:
            # TODO Replace with a better warning definition
            logging.warning(
                "MIDI file only contains drum tracks. Skipping PitchShift.",
            )
            return midi_data

        if self.p_instruments < 1.0:
            num_modified_instruments = int(self.p_instruments * len(modified_instruments))
            if num_modified_instruments == 0:
                # TODO Replace with a better warning definition
                logging.warning(
                    "PitchShift can't be performed on 0 non-drum instruments. Skipping.",
                )
                return midi_data

            modified_instruments = random.sample(modified_instruments, k=num_modified_instruments)

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

            pass
        return midi_data

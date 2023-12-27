import logging
import random

import numpy as np

from core.transforms_interface import BaseMidiTransform

VALID_MODES = ['both', 'left', 'right']


class OnsetTimeShift(BaseMidiTransform):
    def __init__(self, max_shift: float, mode: str = 'both', p_instruments: float = 1.0, p: float = 0.2,
                 eps: float = 1e-12):
        """
        Randomly modify MIDI note onset times while keeping their total durations intact. Post augmentation,
        the note onset time will be >= 0 and <= instrument track duration.

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
                "Valid OnsetTimeShift modes are: {}.".format(VALID_MODES)
            )

        self.max_shift = max_shift

        if mode == 'left':
            self.mode = self._left
        elif mode == 'right':
            self.mode = self._right
        else:
            self.mode = self._both

    def _both(self, onset, duration, instrument_end_time):
        start = np.clip(onset + np.random.uniform(-self.max_shift, self.max_shift), 0, instrument_end_time)
        return start, start + duration

    def _right(self, onset, duration, instrument_end_time):
        start = np.clip(onset + np.random.uniform(0, self.max_shift + 1), 0, instrument_end_time)
        return start, start + duration

    def _left(self, onset, duration, instrument_end_time):
        start = np.clip(onset + np.random.uniform(-self.max_shift, 1), 0, instrument_end_time)
        return start, start + duration

    def apply(self, midi_data):
        modified_instruments = self._get_modified_instruments_list(midi_data)
        for instrument in modified_instruments:
            num_shifted_notes_per_instrument = int(self.p * len(instrument.notes))
            if num_shifted_notes_per_instrument == 0:
                # TODO Replace with a better warning definition
                logging.debug(
                    "OnsetTimeShift can't be performed on 0 notes on given non-drum instrument. Skipping.",
                )
                continue

            instrument_end_time = instrument.notes[-1].end
            for note in random.sample(instrument.notes, k=num_shifted_notes_per_instrument):
                note.start, note.end = self.mode(note.start, note.duration, instrument_end_time)
        return midi_data

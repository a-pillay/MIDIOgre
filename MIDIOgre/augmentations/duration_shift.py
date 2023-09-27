import logging
import random

import numpy as np

from MIDIOgre.core.transforms_interface import BaseMidiTransform

VALID_MODES = ['both', 'shrink', 'extend']


class DurationShift(BaseMidiTransform):
    def __init__(self, max_shift: float, mode: str = 'both', min_duration=1e-6, p_instruments: float = 1.0,
                 p: float = 0.2):
        """
        Randomly modify MIDI note durations while keeping their onset times intact. Post augmentation, the note duration
        will be >= min_duration and <= instrument track duration.

        :param max_shift: Maximum value by which a note duration can be randomly shifted.
        :param mode: 'shrink' if notes can only be shrunk (reduced duration), 'right' if notes can only be extended
        (increased duration), 'both' if notes can be shrunk or extended.
        :param min_duration: The least duration a note can have post shrinkage.
        :param p_instruments: If a MIDI file has >1 instruments, this parameter will determine the percentage of
        instruments that may have random note duration changes.
        :param p: Determines the percentage of notes that may have random duration changes per instrument.
        """
        super().__init__(p_instruments=p_instruments, p=p)

        if mode not in VALID_MODES:
            raise ValueError(
                "Valid DurationShift modes are: {}.".format(VALID_MODES)
            )

        if min_duration < 0:
            raise ValueError(
                "Minimum note duration post shrinkage must be >=0."
            )

        self.max_shift = max_shift
        self.min_duration = min_duration

        if mode == 'shrink':
            self.mode = self._shrink
        elif mode == 'extend':
            self.mode = self._extend
        else:
            self.mode = self._both

    def _both(self, onset, offset, instrument_end_time):
        return np.clip(offset + np.random.uniform(-self.max_shift, self.max_shift), onset + self.min_duration,
                       instrument_end_time)

    def _shrink(self, onset, offset, instrument_end_time):
        return np.clip(offset + np.random.uniform(-self.max_shift, 0), onset + self.min_duration,
                       instrument_end_time)

    def _extend(self, onset, offset, instrument_end_time):
        return np.clip(offset + np.random.uniform(0, self.max_shift), onset + self.min_duration,
                       instrument_end_time)

    def apply(self, midi_data):
        modified_instruments = self._get_modified_instruments_list(midi_data)
        for instrument in modified_instruments:
            num_shifted_notes_per_instrument = int(self.p * len(instrument.notes))
            if num_shifted_notes_per_instrument == 0:
                # TODO Replace with a better warning definition
                logging.debug(
                    "DurationShift can't be performed on 0 notes on given non-drum instrument. Skipping.",
                )
                continue

            instrument_end_time = instrument.notes[-1].end
            for note in random.sample(instrument.notes, k=num_shifted_notes_per_instrument):
                note.end = self.mode(note.start, note.end, instrument_end_time)
        return midi_data

import logging
import math
import random
from operator import itemgetter

import numpy as np

from MIDIOgre.core.transforms_interface import BaseMidiTransform


class NoteDelete(BaseMidiTransform):
    def __init__(self, p_instruments: float = 1.0, p: float = 0.2, eps: float = 1e-12):
        """
        Randomly delete some notes from a MIDI instrument track.

        :param p_instruments: If a MIDI file has >1 instruments, this parameter will determine the percentage of
        instruments that may have random note deletions.
        :param p: Determines the maximum percentage of notes that may be randomly deleted per instrument.
        :param eps: Epsilon term added to represent the lowest possible value (for numerical stability)
        """
        super().__init__(p_instruments=p_instruments, p=p, eps=eps)

        # Correcting self.p to reflect probability of note deletions (inverse effect)
        self.p = 1 - self.p

    def apply(self, midi_data):
        modified_instruments = self._get_modified_instruments_list(midi_data)
        for instrument in modified_instruments:

            num_preserved_notes_per_instrument = math.ceil(np.random.uniform(self.p, 1.0) * len(instrument.notes))
            if num_preserved_notes_per_instrument == 0:
                # TODO Replace with a better warning definition
                logging.debug(
                    "NoteDelete can't be performed on 0 notes on given non-drum instrument. Skipping.",
                )
                continue

            # TODO Is there a better logic to this (more efficient & intuitive)?
            preserved_note_indices = sorted(random.sample(range(len(instrument.notes)),
                                                          k=num_preserved_notes_per_instrument))
            instrument.notes = list(itemgetter(*preserved_note_indices)(instrument.notes))
        return midi_data

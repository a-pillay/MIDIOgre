import logging
import math
import random
from operator import itemgetter

import numpy as np

from midiogre.core.transforms_interface import BaseMidiTransform


class NoteDelete(BaseMidiTransform):
    def __init__(self, can_delete_last_note: bool = True, p_instruments: float = 1.0, p: float = 0.2,
                 eps: float = 1e-12):

        """
        Randomly delete some notes from a MIDI instrument track.

        :param can_delete_last_note: If true, the last note in an instrument will not be deleted (thus keeping the total
        instrument duration unchanged)
        :param p_instruments: If a MIDI file has >1 instruments, this parameter will determine the percentage of
        instruments that may have random note deletions.
        :param p: Determines the maximum percentage of notes that may be randomly deleted per instrument.
        :param eps: Epsilon term added to represent the lowest possible value (for numerical stability)
        """
        super().__init__(p_instruments=p_instruments, p=p, eps=eps)

        self.can_delete_last_note = can_delete_last_note

        # Correcting self.p to reflect probability of note deletions (inverse effect)
        self.p = 1 - self.p

    def apply(self, midi_data):
        modified_instruments = self._get_modified_instruments_list(midi_data)
        for instrument in modified_instruments:

            num_deleteable_notes = len(instrument.notes) if self.can_delete_last_note else len(instrument.notes) - 1
            num_preserved_notes_per_instrument = math.ceil(np.random.uniform(self.p, 1.0) * num_deleteable_notes)
            if num_preserved_notes_per_instrument == 0:
                # TODO Replace with a better warning definition
                logging.debug(
                    "NoteDelete can't be performed on 0 notes on given non-drum instrument. Skipping.",
                )
                continue

            # TODO Check for a better logic (more efficient & intuitive)
            preserved_note_indices = sorted(random.sample(range(num_deleteable_notes),
                                                          k=num_preserved_notes_per_instrument))
            if not self.can_delete_last_note:
                # Add index of last note to the preserved note indices list
                preserved_note_indices.append(num_deleteable_notes)
            instrument.notes = list(itemgetter(*preserved_note_indices)(instrument.notes))
        return midi_data

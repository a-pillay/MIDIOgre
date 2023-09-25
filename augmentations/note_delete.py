import logging
import random
from operator import itemgetter

from core.transforms_interface import BaseMidiTransform


class NoteDelete(BaseMidiTransform):
    def __init__(self, p_instruments: float = 1.0, p: float = 0.2):
        """
        Randomly delete some notes from a MIDI instrument track.

        :param p_instruments: If a MIDI file has >1 instruments, this parameter will determine the percentage of
        instruments that may have random note deletions.
        :param p: Determines the percentage of notes that may be randomly deleted per instrument.
        """
        super().__init__(p_instruments=p_instruments, p=p)

        # Correcting self.p to reflect probability of note deletions (inverse effect)
        self.p = 1 - self.p

    def apply(self, midi_data):
        modified_instruments = self._get_modified_instruments_list(midi_data)
        for instrument in modified_instruments:
            num_deleted_notes_per_instrument = int(self.p * len(instrument.notes))
            if num_deleted_notes_per_instrument == 0:
                # TODO Replace with a better warning definition
                logging.debug(
                    "NoteDelete can't be performed on 0 notes on given non-drum instrument. Skipping.",
                )
                continue

            indices = sorted(random.sample(range(len(instrument.notes)), k=num_deleted_notes_per_instrument))
            instrument.notes = itemgetter(*indices)(instrument.notes)
        return midi_data

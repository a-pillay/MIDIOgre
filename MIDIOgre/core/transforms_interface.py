import logging
import random


class BaseMidiTransform:
    def __init__(self, p_instruments: float, p: float, eps: float = 1e-12):
        if not 0 <= p <= 1:
            raise ValueError(
                "Probability of applying a MIDI Transform must be >=0 and <=1."
            )

        if not 0 <= p_instruments <= 1:
            raise ValueError(
                "Probability of applying PitchShift on an instrument must be >=0 and <=1."
            )

        self.p = p
        self.p_instruments = p_instruments
        self.eps = eps

    def _get_modified_instruments_list(self, midi_data):

        # filtering out drum instruments (TODO: Evaluate whether this is needed)
        modified_instruments = [instrument for instrument in midi_data.instruments if not instrument.is_drum]

        if len(modified_instruments) == 0:
            # TODO Replace with a better warning definition
            logging.warning(
                "MIDI file only contains drum tracks.",
            )
        elif self.p_instruments < 1.0 and len(modified_instruments) > 1:
            num_modified_instruments = int(self.p_instruments * len(modified_instruments))
            if num_modified_instruments == 0:
                # TODO Replace with a better warning definition
                logging.debug(
                    "No instruments left to randomly modify in MIDI file. Skipping.",
                )
                return midi_data

            modified_instruments = random.sample(modified_instruments, k=num_modified_instruments)

        return modified_instruments

    def apply(self, midi_data):
        raise NotImplementedError

    def __call__(self, midi_data):
        return self.apply(midi_data)

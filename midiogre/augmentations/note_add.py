import math

import numpy as np
from pretty_midi import Note

from core.transforms_interface import BaseMidiTransform


class NoteAdd(BaseMidiTransform):
    def __init__(self, note_num_range: (int, int), note_velocity_range: (int, int), note_duration_range: (int, int),
                 restrict_to_instrument_time: bool = True, p_instruments: float = 1.0, p: float = 0.2,
                 eps: float = 1e-12):
        """
        Randomly add some notes to a MIDI instrument track.

        :param note_num_range: Range of MIDI note number values that the randomly added notes can take.
        :param note_velocity_range: Range of MIDI note velocities that the randomly added notes can take.
        :param note_duration_range: Range of MIDI note durations that the randomly added notes can take.
        :param restrict_to_instrument_time: If true, none of the randomly added notes will have offset times greater
        than that of the last pre-existing note in the instrument. In case the duration of a new note exceeds this
        timestamp, its offset time will be clipped to have the same offset time as the last instrument note.
        :param p_instruments: If a MIDI file has >1 instruments, this parameter will determine the percentage of
        instruments that may have random note deletions.
        :param p: Determines the maximum percentage of notes (relative to total notes) that may be randomly added
        per instrument.
        :param eps: Epsilon term added to represent the lowest possible value (for numerical stability)
        """
        super().__init__(p_instruments=p_instruments, p=p, eps=eps)

        if len(note_num_range) != 2 or note_num_range[1] < note_num_range[0]:
            raise ValueError(
                "MIDI note numbers must be specified as (min_note_num, max_note_num] where max_note_num > min_note_num."
            )

        if not (0 <= note_num_range[0] < 128) or not (0 <= note_num_range[1] < 128):
            raise ValueError(
                "MIDI note numbers must be integers >=0 and < 128."
            )

        if len(note_velocity_range) != 2 or note_velocity_range[1] < note_velocity_range[0]:
            raise ValueError(
                "MIDI note velocities must be specified as (min_note_velocity, max_note_velocity] where "
                "max_note_velocity > min_note_velocity."
            )

        if not (0 <= note_velocity_range[0] < 128) or not (0 <= note_velocity_range[1] < 128):
            raise ValueError(
                "MIDI note velocities must be integers >=0 and < 128."
            )

        if len(note_duration_range) != 2 or note_duration_range[1] < note_duration_range[0]:
            raise ValueError(
                "MIDI note durations must be specified as (min_note_duration, max_note_duration] where "
                "min_note_duration > max_note_duration."
            )

        self.min_note_num = note_num_range[0]
        self.max_note_num = note_num_range[1]

        self.min_velo = note_velocity_range[0]
        self.max_velo = note_velocity_range[1]

        self.min_durn = note_duration_range[0]
        self.max_durn = note_duration_range[1]

        self.restrict_to_instrument_time = restrict_to_instrument_time

    def __generate_n_midi_notes(self, n, instrument_end_time):
        generated_notes = []
        for i in range(n):
            generated_notes.append(Note(pitch=np.random.randint(self.min_note_num, self.max_note_num + 1),
                                        velocity=np.random.randint(self.min_velo, self.max_velo + 1),
                                        start=np.random.uniform(0, instrument_end_time),
                                        end=None))
            generated_notes[-1].end = np.clip(
                generated_notes[-1].start + np.random.uniform(self.min_durn, self.max_durn),
                None,
                instrument_end_time
            )
        return generated_notes

    def apply(self, midi_data):
        modified_instruments = self._get_modified_instruments_list(midi_data)
        for instrument in modified_instruments:
            num_new_notes_added_per_instrument = math.ceil(np.random.uniform(self.eps, self.p) * len(instrument.notes))
            instrument.notes.extend(
                self.__generate_n_midi_notes(n=num_new_notes_added_per_instrument,
                                             instrument_end_time=instrument.notes[-1].end)
            )
        return midi_data

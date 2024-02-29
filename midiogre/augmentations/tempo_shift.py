import logging
import random

import numpy as np
from mido import MetaMessage

from core.transforms_interface import BaseMidiTransform

VALID_MODES = ['both', 'up', 'down']


class TempoShift(BaseMidiTransform):
    def __init__(self, max_shift: float, mode: str = 'both', tempo_range: (float, float) = (30.0, 200.0),
                 p: float = 0.2, eps: float = 1e-12):
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
        :param eps: Epsilon term added to represent the lowest possible value (for numerical stability)
        """
        super().__init__(p_instruments=1.0, p=p, eps=eps)

        if mode not in VALID_MODES:
            raise ValueError(
                "Valid DurationShift modes are: {}.".format(VALID_MODES)
            )

        if tempo_range[0] < 0:
            raise ValueError(
                "Lower range of tempo must be >=0."
            )

        self.max_shift = max_shift
        self.tempo_range = tempo_range

        if mode == 'up':
            self.mode = self._up
        elif mode == 'down':
            self.mode = self._down
        else:
            self.mode = self._both

    def _both(self, tempo):
        return int(6e7 / np.clip(tempo +
                                 np.random.uniform(-self.max_shift, self.max_shift),
                                 self.tempo_range[0],
                                 self.tempo_range[1]))

    def _up(self, tempo):
        return int(6e7 / np.clip(tempo +
                                 np.random.uniform(0, self.max_shift), self.tempo_range[0], self.tempo_range[1]))

    def _down(self, tempo):
        return int(6e7 / np.clip(tempo +
                                 np.random.uniform(-self.max_shift, 0), self.tempo_range[0], self.tempo_range[1]))

    def apply(self, midi_data):

        tempo_events_idx = []
        for idx, event in enumerate(midi_data.tracks[0]):
            if event.type == 'set_tempo':
                tempo_events_idx.append(idx)

        if len(tempo_events_idx) == 0:
            logging.warning("No tempo metadata found in MIDI file; assuming a default value of 120 BPM.")
            tempo = 120.0
        else:
            tempo = 6e7 / midi_data.tracks[0][tempo_events_idx[0]].tempo

        for idx in tempo_events_idx:
            midi_data.tracks[0].pop(idx)

        midi_data.tracks[0].append(MetaMessage(type="set_tempo", tempo=self.mode(tempo), time=0))

        return midi_data

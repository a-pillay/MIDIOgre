import logging
import random

import numpy as np
from core.transforms_interface import BaseMidiTransform

VALID_MODES = ['both', 'increase', 'decrease']


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

        if mode == 'increase':
            self.mode = self._increase
        elif mode == 'decrease':
            self.mode = self._decrease
        else:
            self.mode = self._both

    def _both(self, tempo):
        return np.clip(tempo + np.random.uniform(-self.max_shift, self.max_shift),
                       self.tempo_range[0], self.tempo_range[1])

    def _increase(self, tempo):
        return np.clip(tempo + np.random.uniform(-self.max_shift, 0),
                       self.tempo_range[0], self.tempo_range[1])

    def _decrease(self, tempo):
        return np.clip(tempo + np.random.uniform(0, self.max_shift),
                       self.tempo_range[0], self.tempo_range[1])

    def apply(self, midi_data):
        if random.random() < self.p:
            old_tick_to_time = midi_data._PrettyMIDI__tick_to_time
            old_tick_scales = midi_data._tick_scales
            base_tempo = 60 / (old_tick_scales[0][1] * midi_data.resolution)

            new_tick_scales = [(0, 60.0 / (self.mode(base_tempo) * midi_data.resolution))]
            midi_data._tick_scales = new_tick_scales
            midi_data._update_tick_to_time(len(midi_data._PrettyMIDI__tick_to_time) - 1)
            new_tick_to_time = midi_data._PrettyMIDI__tick_to_time

            midi_data._PrettyMIDI__tick_to_time = old_tick_to_time
            midi_data._tick_scales = old_tick_scales

            midi_data.adjust_times(old_tick_to_time, new_tick_to_time)
            midi_data._tick_scales = new_tick_scales

        return midi_data

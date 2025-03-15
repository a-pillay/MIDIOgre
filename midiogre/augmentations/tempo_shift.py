import logging
import random

import numpy as np
from mido import MetaMessage

from midiogre.core.transforms_interface import BaseMidiTransform

VALID_MODES = ['both', 'up', 'down']


class TempoShift(BaseMidiTransform):
    def __init__(self, max_shift: float, mode: str = 'both', tempo_range: (float, float) = (30.0, 200.0),
                 p: float = 0.2, eps: float = 1e-12):
        """
        Randomly modify MIDI tempo while keeping note timings intact.

        :param max_shift: Maximum value by which tempo can be randomly shifted (in BPM).
        :param mode: 'up' if tempo can only be increased, 'down' if tempo can only be decreased,
        'both' if tempo can be increased or decreased.
        :param tempo_range: (min_tempo, max_tempo) in BPM that the tempo must stay within.
        :param p: Probability of applying the tempo shift.
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
        shifted_tempo = np.clip(tempo + np.random.uniform(-self.max_shift, self.max_shift),
                              self.tempo_range[0],
                              self.tempo_range[1])
        return int(round(6e7 / shifted_tempo))

    def _up(self, tempo):
        shifted_tempo = np.clip(tempo + np.random.uniform(0, self.max_shift),
                              self.tempo_range[0],
                              self.tempo_range[1])
        return int(round(6e7 / shifted_tempo))

    def _down(self, tempo):
        shifted_tempo = np.clip(tempo + np.random.uniform(-self.max_shift, 0),
                              self.tempo_range[0],
                              self.tempo_range[1])
        return int(round(6e7 / shifted_tempo))

    def apply(self, midi_data):
        # First find all tempo events
        tempo_events_idx = []
        for idx, event in enumerate(midi_data.tracks[0]):
            if event.type == 'set_tempo':
                tempo_events_idx.append(idx)

        # Get the initial tempo (or use default 120 BPM)
        if len(tempo_events_idx) == 0:
            logging.warning("No tempo metadata found in MIDI file; assuming a default value of 120 BPM.")
            tempo = 120.0
        else:
            # Use the first tempo event as reference
            tempo = 6e7 / midi_data.tracks[0][tempo_events_idx[0]].tempo

        # Remove all existing tempo events (in reverse order to maintain indices)
        for idx in reversed(tempo_events_idx):
            midi_data.tracks[0].pop(idx)

        # Add new tempo event at the start if probability check passes
        if np.random.random() > self.p:
            midi_data.tracks[0].insert(0, MetaMessage(type="set_tempo", tempo=self.mode(tempo), time=0))
        else:
            # If no change, insert original tempo
            midi_data.tracks[0].insert(0, MetaMessage(type="set_tempo", tempo=int(round(6e7 / tempo)), time=0))

        return midi_data

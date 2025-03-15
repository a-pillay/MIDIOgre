import logging
import random

import numpy as np
from mido import MetaMessage

from midiogre.core.transforms_interface import BaseMidiTransform

VALID_MODES = ['both', 'up', 'down']


class TempoShift(BaseMidiTransform):
    def __init__(self, max_shift: float, mode: str = 'both', tempo_range: (float, float) = (30.0, 200.0),
                 p: float = 0.2, respect_tempo_shifts: bool = True, eps: float = 1e-12):
        """
        Randomly modify MIDI tempo while keeping note timings intact.

        :param max_shift: Maximum value by which tempo can be randomly shifted (in BPM).
        :param mode: 'up' if tempo can only be increased, 'down' if tempo can only be decreased,
        'both' if tempo can be increased or decreased.
        :param tempo_range: (min_tempo, max_tempo) in BPM that the tempo must stay within.
        :param p: Probability of applying the tempo shift.
        :param respect_tempo_shifts: If True, preserves all tempo change events in the file, shifting each
        tempo while maintaining their original timing. If False, replaces all tempo events with a single
        tempo at the start of the file.
        :param eps: Epsilon term added to represent the lowest possible value (for numerical stability)
        """
        super().__init__(p_instruments=1.0, p=p, eps=eps)

        if mode not in VALID_MODES:
            raise ValueError(
                f"Mode must be one of {VALID_MODES}, got {mode}"
            )

        if max_shift <= 0:
            raise ValueError(
                f"max_shift must be positive, got {max_shift}"
            )

        if not isinstance(tempo_range, (tuple, list)) or len(tempo_range) != 2:
            raise ValueError(
                f"tempo_range must be a tuple or list of length 2, got {tempo_range}"
            )

        min_tempo, max_tempo = tempo_range
        if min_tempo < 0:
            raise ValueError(
                f"min_tempo must be positive, got {min_tempo}"
            )
        
        if min_tempo >= max_tempo:
            raise ValueError(
                f"min_tempo must be less than max_tempo, got {min_tempo} >= {max_tempo}"
            )

        self.max_shift = max_shift
        self.tempo_range = tempo_range
        self.respect_tempo_shifts = respect_tempo_shifts
        self.mode = mode

    def _generate_shifts(self, num_shifts: int) -> np.ndarray:
        """Generate tempo shifts in a vectorized manner."""
        if num_shifts == 0:
            return np.array([])
            
        if self.mode == 'up':
            return np.random.uniform(0, self.max_shift, num_shifts)
        elif self.mode == 'down':
            return np.random.uniform(-self.max_shift, 0, num_shifts)
        else:  # both
            return np.random.uniform(-self.max_shift, self.max_shift, num_shifts)

    def _convert_tempo_to_bpm(self, tempo_microseconds_per_beat: int) -> float:
        """Convert tempo from microseconds per beat to BPM."""
        return 6e7 / tempo_microseconds_per_beat

    def _convert_bpm_to_tempo(self, bpm: float) -> int:
        """Convert BPM to tempo in microseconds per beat."""
        return int(round(6e7 / bpm))

    def apply(self, midi_data):
        """
        Apply the tempo shift transformation to the MIDI data.

        Args:
            midi_data: A mido.MidiFile object to transform.

        Returns:
            The transformed mido.MidiFile object.

        Note:
            - If no tempo events are found, a default tempo of 120 BPM is used.
            - Only tempo events in the first track are processed.
            - When respect_tempo_shifts is True, all tempo events maintain their
              relative timing but get new tempo values.
            - When respect_tempo_shifts is False, all tempo events are replaced
              with a single tempo event at the start.
        """
        if not midi_data.tracks:
            logging.warning("Empty MIDI file provided")
            return midi_data

        # Find all tempo events in first track
        tempo_events = []
        tempo_events_idx = []
        for idx, event in enumerate(midi_data.tracks[0]):
            if event.type == 'set_tempo':
                tempo_events.append(event)
                tempo_events_idx.append(idx)

        # Handle case with no tempo events
        if not tempo_events:
            logging.warning("No tempo metadata found in MIDI file; assuming a default value of 120 BPM.")
            default_bpm = 120.0
            should_change = np.random.random() < self.p
            
            if should_change:
                shifts = self._generate_shifts(1)
                new_bpm = np.clip(default_bpm + shifts[0], self.tempo_range[0], self.tempo_range[1])
                new_tempo = self._convert_bpm_to_tempo(new_bpm)
            else:
                new_tempo = self._convert_bpm_to_tempo(default_bpm)
                
            midi_data.tracks[0].insert(0, MetaMessage(type="set_tempo", tempo=new_tempo, time=0))
            return midi_data

        # Remove existing tempo events (in reverse order to maintain indices)
        for idx in reversed(tempo_events_idx):
            midi_data.tracks[0].pop(idx)

        # Determine if we should apply changes
        should_change = np.random.random() < self.p

        if self.respect_tempo_shifts:
            # Process all tempo events while maintaining timing
            if should_change:
                shifts = self._generate_shifts(len(tempo_events))
                for i, event in enumerate(tempo_events):
                    current_bpm = self._convert_tempo_to_bpm(event.tempo)
                    new_bpm = np.clip(current_bpm + shifts[i], self.tempo_range[0], self.tempo_range[1])
                    new_tempo = self._convert_bpm_to_tempo(new_bpm)
                    midi_data.tracks[0].append(
                        MetaMessage(type="set_tempo", tempo=new_tempo, time=event.time)
                    )
            else:
                # Keep original tempos
                for event in tempo_events:
                    midi_data.tracks[0].append(
                        MetaMessage(type="set_tempo", tempo=event.tempo, time=event.time)
                    )
        else:
            # Use only first tempo event and place at start
            current_bpm = self._convert_tempo_to_bpm(tempo_events[0].tempo)
            if should_change:
                shifts = self._generate_shifts(1)
                new_bpm = np.clip(current_bpm + shifts[0], self.tempo_range[0], self.tempo_range[1])
                new_tempo = self._convert_bpm_to_tempo(new_bpm)
            else:
                new_tempo = tempo_events[0].tempo
                
            midi_data.tracks[0].insert(0, MetaMessage(type="set_tempo", tempo=new_tempo, time=0))

        return midi_data

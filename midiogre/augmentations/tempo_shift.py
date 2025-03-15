"""
Tempo shift augmentation for MIDI files.

This module provides functionality to modify the tempo of MIDI files while preserving
the relative timing of notes. It can either preserve all tempo changes in a piece or
consolidate them into a single tempo.
"""

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

        The transformation can either preserve all tempo changes in the piece (respect_tempo_shifts=True)
        or consolidate them into a single tempo at the start (respect_tempo_shifts=False). When preserving
        tempo changes, the relative timing between tempo events is maintained.

        Note: Only tempo events in the first track are processed. Tempo events in other tracks will be
        removed as per MIDI specification (tempo events should only be in the first track).

        Args:
            max_shift: Maximum value by which tempo can be randomly shifted (in BPM).
                Must be positive.
            mode: One of ['up', 'down', 'both']. Determines whether tempo can be:
                - 'up': only increased
                - 'down': only decreased
                - 'both': either increased or decreased
            tempo_range: (min_tempo, max_tempo) in BPM that the tempo must stay within.
                The min_tempo must be positive and less than max_tempo.
            p: Probability of keeping the original tempo. With probability (1-p),
                the tempo will be shifted.
            respect_tempo_shifts: If True, preserves all tempo change events in the file,
                shifting each tempo while maintaining their original timing. If False,
                replaces all tempo events with a single tempo at the start of the file.
            eps: Epsilon term added to represent the lowest possible value
                (for numerical stability)

        Examples:
            # Create a TempoShift that randomly increases or decreases tempo by up to 20 BPM
            tempo_shift = TempoShift(max_shift=20.0, mode='both')

            # Create a TempoShift that only increases tempo, preserving all tempo changes
            tempo_shift = TempoShift(max_shift=10.0, mode='up', respect_tempo_shifts=True)

            # Create a TempoShift that consolidates all tempo changes to one tempo
            tempo_shift = TempoShift(max_shift=15.0, respect_tempo_shifts=False)

        Raises:
            ValueError: If mode is invalid, max_shift is not positive, or tempo_range is invalid.
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
                f"Lower range of tempo must be >=0, got {min_tempo}"
            )
        
        if min_tempo >= max_tempo:
            raise ValueError(
                f"min_tempo must be less than max_tempo, got {min_tempo} >= {max_tempo}"
            )

        self.max_shift = max_shift
        self.tempo_range = tempo_range
        self.respect_tempo_shifts = respect_tempo_shifts

        if mode == 'up':
            self.mode = self._up
        elif mode == 'down':
            self.mode = self._down
        else:
            self.mode = self._both

    def _both(self, tempo):
        """Shift tempo up or down randomly within max_shift."""
        shifted_tempo = np.clip(tempo + np.random.uniform(-self.max_shift, self.max_shift),
                              self.tempo_range[0],
                              self.tempo_range[1])
        return int(round(6e7 / shifted_tempo))

    def _up(self, tempo):
        """Shift tempo up randomly within max_shift."""
        shifted_tempo = np.clip(tempo + np.random.uniform(0, self.max_shift),
                              self.tempo_range[0],
                              self.tempo_range[1])
        return int(round(6e7 / shifted_tempo))

    def _down(self, tempo):
        """Shift tempo down randomly within max_shift."""
        shifted_tempo = np.clip(tempo + np.random.uniform(-self.max_shift, 0),
                              self.tempo_range[0],
                              self.tempo_range[1])
        return int(round(6e7 / shifted_tempo))

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

        # First find all tempo events
        tempo_events = []
        tempo_events_idx = []
        for idx, event in enumerate(midi_data.tracks[0]):
            if event.type == 'set_tempo':
                tempo_events.append(event)
                tempo_events_idx.append(idx)

        # Check for tempo events in other tracks and warn
        for track_idx, track in enumerate(midi_data.tracks[1:], 1):
            has_tempo = any(event.type == 'set_tempo' for event in track)
            if has_tempo:
                logging.warning(
                    f"Found tempo events in track {track_idx}. These will be removed as "
                    "tempo events should only be in the first track."
                )
                # Remove tempo events from other tracks
                track[:] = [event for event in track if event.type != 'set_tempo']

        # Get the initial tempo (or use default 120 BPM)
        if len(tempo_events) == 0:
            logging.warning("No tempo metadata found in MIDI file; assuming a default value of 120 BPM.")
            tempo = 120.0
            # Add default tempo event at start
            if np.random.random() > self.p:
                midi_data.tracks[0].insert(0, MetaMessage(type="set_tempo", tempo=self.mode(tempo), time=0))
            else:
                midi_data.tracks[0].insert(0, MetaMessage(type="set_tempo", tempo=int(round(6e7 / tempo)), time=0))
        else:
            # Remove all existing tempo events (in reverse order to maintain indices)
            for idx in reversed(tempo_events_idx):
                midi_data.tracks[0].pop(idx)

            if self.respect_tempo_shifts:
                # Process each tempo event while maintaining its timing
                should_change = np.random.random() > self.p
                for event in tempo_events:
                    original_tempo = 6e7 / event.tempo
                    if should_change:
                        new_tempo = self.mode(original_tempo)
                    else:
                        new_tempo = int(round(6e7 / original_tempo))
                    midi_data.tracks[0].append(
                        MetaMessage(type="set_tempo", tempo=new_tempo, time=event.time)
                    )
            else:
                # Use only first tempo event as reference and place at start
                tempo = 6e7 / tempo_events[0].tempo
                if len(tempo_events) > 1:
                    logging.info(
                        f"Found {len(tempo_events)} tempo events. Using only the first one "
                        "since respect_tempo_shifts=False."
                    )
                if np.random.random() > self.p:
                    midi_data.tracks[0].insert(0, MetaMessage(type="set_tempo", tempo=self.mode(tempo), time=0))
                else:
                    midi_data.tracks[0].insert(0, MetaMessage(type="set_tempo", tempo=int(round(6e7 / tempo)), time=0))

        return midi_data

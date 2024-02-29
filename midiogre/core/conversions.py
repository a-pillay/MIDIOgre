import logging
from pathlib import Path
from typing import Union

import mido
import torch
import pretty_midi


class BaseConversion:
    def __init__(self):
        pass

    def apply(self, midi_data):
        raise NotImplementedError

    def __call__(self, midi_data: Union[str, pretty_midi.PrettyMIDI, mido.MidiFile]):

        if isinstance(midi_data, str):
            midi_data = midi_data.strip()
            if not Path(midi_data).is_file():
                raise ValueError("Invalid path provided: {}".format(midi_data))

        return self.apply(midi_data)


class ConvertToMido(BaseConversion):
    def __init__(self):
        super().__init__()

    def apply(self, path_to_midi: str):
        midi_data = mido.MidiFile(path_to_midi)

        # Borrowed from pretty-midi
        if any(e.type in ('set_tempo', 'key_signature', 'time_signature')
               for track in midi_data.tracks[1:] for e in track):
            logging.warning(
                "Tempo, Key or Time signature change events found on "
                "non-zero tracks.  This is not a valid type 0 or type 1 "
                "MIDI file.  Tempo, Key or Time Signature may be wrong.",
                RuntimeWarning)

        return midi_data


class ConvertToPrettyMIDI(BaseConversion):
    def __init__(self):
        super().__init__()

    def apply(self, midi_data: Union[str, mido.MidiFile]):
        if isinstance(midi_data, str):
            return pretty_midi.PrettyMIDI(midi_file=midi_data)

        return pretty_midi.PrettyMIDI(mido_object=midi_data)


class ToPRollNumpy(BaseConversion):
    def __init__(self, binarize=False, fs=100, times=None, pedal_threshold=64):
        """
        Convert a PrettMIDI object to a piano roll 2D numpy array of shape (128, *).

        Refer https://craffel.github.io/pretty-midi/#pretty_midi.PrettyMIDI.get_piano_roll for more information.

        :param kwargs: Keyword arguments to PrettyMIDI.get_piano_roll(..)
        """

        super().__init__()
        self.binarize = binarize
        self.fs = fs
        self.times = times
        self.pedal_threshold = pedal_threshold

    def apply(self, midi_data):
        return midi_data.get_piano_roll(fs=self.fs, times=self.times, pedal_threshold=self.pedal_threshold)


class ToPRollTensor(ToPRollNumpy):
    def __init__(self, binarize=False, device='cpu', fs=100, times=None, pedal_threshold=64):
        """
        Convert a PrettMIDI object to a piano roll 2D PyTorch tensor of shape (128, *).

        Refer https://craffel.github.io/pretty-midi/#pretty_midi.PrettyMIDI.get_piano_roll for more information.

        :param kwargs: Keyword arguments to PrettyMIDI.get_piano_roll(..)
        """

        self.device = device
        super().__init__(binarize=binarize, fs=fs, times=times, pedal_threshold=pedal_threshold)

    def apply(self, midi_data):
        return torch.Tensor(super().apply(midi_data=midi_data)).to(self.device)

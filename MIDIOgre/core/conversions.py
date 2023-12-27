import torch


class BaseConversion:
    def __init__(self):
        pass

    def apply(self, midi_data):
        raise NotImplementedError

    def __call__(self, midi_data):
        return self.apply(midi_data)


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

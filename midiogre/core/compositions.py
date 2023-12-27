from pretty_midi import PrettyMIDI


class Compose:

    def __init__(self, transforms: list or tuple):
        """
        Compose several MIDIOgre transforms together.

        :param transforms: list of MIDIOgre transforms to be performed in the given order
        """
        if not (isinstance(transforms, list) or isinstance(transforms, tuple)):
            raise TypeError(
                "Transforms to be composed must be wrapped in a list or a tuple, got {}".format(type(transforms))
            )

        self.transforms = transforms

    def __len__(self):
        return len(self.transforms)

    def __call__(self, midi_data: PrettyMIDI):
        for transform in self.transforms:
            midi_data = transform(midi_data)
        return midi_data

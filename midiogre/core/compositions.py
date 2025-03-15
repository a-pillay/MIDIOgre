"""Composition module for chaining multiple MIDI transforms.

This module provides functionality to compose multiple MIDI transforms into a single
transform pipeline. The transforms are applied sequentially in the order they are provided.

Example:
    >>> from midiogre.augmentations import PitchShift, OnsetTimeShift
    >>> from midiogre.core import Compose
    >>> 
    >>> # Create a transform pipeline
    >>> transform = Compose([
    ...     PitchShift(max_shift=2, p=0.5),
    ...     OnsetTimeShift(max_shift=0.1, p=0.3)
    ... ])
    >>> 
    >>> # Apply transforms to MIDI data
    >>> transformed_midi = transform(midi_data)
"""

from pretty_midi import PrettyMIDI


class Compose:
    """A class for composing multiple MIDI transforms into a single transform.
    
    This class allows you to chain multiple transforms together and apply them
    sequentially to MIDI data. Each transform in the pipeline must be a callable
    that takes a PrettyMIDI object as input and returns a transformed PrettyMIDI
    object.
    
    Args:
        transforms (list or tuple): A sequence of MIDI transforms to be applied
            in order. Each transform should be a callable that takes a PrettyMIDI
            object as input and returns a transformed PrettyMIDI object.
            
    Raises:
        TypeError: If transforms is not a list or tuple.
        
    Example:
        >>> # Create transforms
        >>> pitch_shift = PitchShift(max_shift=2)
        >>> onset_shift = OnsetTimeShift(max_shift=0.1)
        >>> 
        >>> # Compose transforms
        >>> transform = Compose([pitch_shift, onset_shift])
        >>> 
        >>> # Apply to MIDI data
        >>> transformed_midi = transform(midi_data)
    """

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
        """Return the number of transforms in the composition.
        
        Returns:
            int: Number of transforms in the pipeline.
        """
        return len(self.transforms)

    def __call__(self, midi_data: PrettyMIDI):
        """Apply all transforms sequentially to the MIDI data.
        
        Args:
            midi_data (PrettyMIDI): The MIDI data to transform.
            
        Returns:
            PrettyMIDI: The transformed MIDI data after applying all transforms
            in sequence.
        """
        for transform in self.transforms:
            midi_data = transform(midi_data)
        return midi_data

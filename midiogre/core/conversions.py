"""MIDI format conversion module.

This module provides tools for converting between different MIDI representations and formats,
primarily to enable composition of multiple MIDI augmentations. Similar to image augmentation
libraries like albumentations, these converters allow you to seamlessly chain transforms
that operate on different MIDI formats.

The module supports conversions between:
- Mido MidiFile objects
- PrettyMIDI objects
- Piano roll representations (NumPy arrays and PyTorch tensors)

Primary Use Case - Augmentation Pipeline:
    >>> from midiogre.augmentations import PitchShift, TempoShift, OnsetTimeShift
    >>> from midiogre.core.conversions import ConvertToMido, ConvertToPrettyMIDI
    >>> from midiogre.core import Compose
    >>> 
    >>> # Create a pipeline with transforms that need different MIDI formats
    >>> transform = Compose([
    ...     PitchShift(max_shift=2),  # Uses PrettyMIDI
    ...     ConvertToMido(),  # Convert to Mido format
    ...     TempoShift(max_shift=20.0),  # Uses Mido
    ...     ConvertToPrettyMIDI(),  # Convert back to PrettyMIDI
    ...     OnsetTimeShift(max_shift=0.1)  # Uses PrettyMIDI
    ... ])
    >>> 
    >>> # Apply entire pipeline with automatic format conversion
    >>> midi_data = pretty_midi.PrettyMIDI('song.mid')
    >>> transformed = transform(midi_data)

Other Use Cases:
    >>> # Direct format conversion
    >>> mido_obj = ConvertToMido()('song.mid')
    >>> pretty_midi_obj = ConvertToPrettyMIDI()(mido_obj)
    >>> 
    >>> # Create piano roll representation
    >>> from midiogre.core.conversions import ToPRollTensor
    >>> piano_roll = ToPRollTensor()(pretty_midi_obj)

Note:
    When composing transforms, the converters handle format compatibility automatically.
    You only need to ensure that:
    1. The input format matches what the first transform expects
    2. You convert between formats when switching between Mido and PrettyMIDI transforms
    3. The final output format is what you need for your application
"""

import logging
from pathlib import Path
from typing import Union

import mido
import torch
import pretty_midi


class BaseConversion:
    """Base class for all MIDI format conversions.
    
    This class defines the interface that all conversion classes must implement.
    Conversion classes are callable objects that take a MIDI object or file path
    as input and return a transformed representation.
    
    The __call__ method handles input validation and file path resolution, while
    the actual conversion logic should be implemented in the apply method.
    """

    def __init__(self):
        """Initialize the conversion."""
        pass

    def apply(self, midi_data):
        """Apply the conversion to the MIDI data.
        
        This method should be implemented by all conversion classes.
        
        Args:
            midi_data: The MIDI data to convert.
            
        Returns:
            The converted MIDI data.
            
        Raises:
            NotImplementedError: If the child class does not implement this method.
        """
        raise NotImplementedError

    def __call__(self, midi_data: Union[str, pretty_midi.PrettyMIDI, mido.MidiFile]):
        """Convert the MIDI data.
        
        Args:
            midi_data: The MIDI data to convert. Can be:
                - A file path string
                - A PrettyMIDI object
                - A Mido MidiFile object
                
        Returns:
            The converted MIDI data.
            
        Raises:
            ValueError: If a file path is provided but the file does not exist.
        """
        if isinstance(midi_data, str):
            midi_data = midi_data.strip()
            if not Path(midi_data).is_file():
                raise ValueError("Invalid path provided: {}".format(midi_data))

        return self.apply(midi_data)


class ConvertToMido(BaseConversion):
    """Convert MIDI data to a Mido MidiFile object.
    
    This converter is particularly useful when working with transforms that
    require Mido objects, such as TempoShift.
    
    Example:
        >>> converter = ConvertToMido()
        >>> # Convert from file
        >>> mido_obj = converter('song.mid')
        >>> # Convert from PrettyMIDI
        >>> mido_obj = converter(pretty_midi_obj)
    """

    def __init__(self):
        """Initialize the Mido converter."""
        super().__init__()

    def apply(self, path_to_midi: str):
        """Convert MIDI data to a Mido MidiFile object.
        
        Args:
            path_to_midi (str): Path to the MIDI file to load.
            
        Returns:
            mido.MidiFile: The loaded MIDI data as a Mido object.
            
        Warning:
            If tempo, key signature, or time signature events are found on non-zero
            tracks, they may not be interpreted correctly as this violates the MIDI
            type 0/1 specification.
        """
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
    """Convert MIDI data to a PrettyMIDI object.
    
    This converter is useful when working with most MIDIOgre transforms, as they
    typically operate on PrettyMIDI objects. It can convert from either a file path
    or a Mido MidiFile object.
    
    Example:
        >>> converter = ConvertToPrettyMIDI()
        >>> # Convert from file
        >>> pretty_midi_obj = converter('song.mid')
        >>> # Convert from Mido
        >>> pretty_midi_obj = converter(mido_obj)
    """

    def __init__(self):
        """Initialize the PrettyMIDI converter."""
        super().__init__()

    def apply(self, midi_data: Union[str, mido.MidiFile]):
        """Convert MIDI data to a PrettyMIDI object.
        
        Args:
            midi_data: The MIDI data to convert. Can be:
                - A file path string
                - A Mido MidiFile object
                
        Returns:
            pretty_midi.PrettyMIDI: The converted MIDI data.
        """
        if isinstance(midi_data, str):
            return pretty_midi.PrettyMIDI(midi_file=midi_data)

        return pretty_midi.PrettyMIDI(mido_object=midi_data)


class ToPRollNumpy(BaseConversion):
    """Convert MIDI data to a piano roll NumPy array.
    
    This converter transforms MIDI data into a piano roll representation as a
    2D NumPy array of shape (128, time_steps). The array values represent note
    velocities at each time step.
    
    Args:
        binarize (bool, optional): Whether to binarize the piano roll.
            Default: False
        fs (int, optional): Sampling frequency in Hz.
            Default: 100
        times (array-like, optional): Times at which to sample the piano roll.
            Default: None
        pedal_threshold (int, optional): Threshold above which the sustain pedal
            is activated.
            Default: 64
            
    Example:
        >>> converter = ToPRollNumpy(fs=200)  # 200 Hz sampling
        >>> piano_roll = converter(pretty_midi_obj)  # Shape: (128, time_steps)
    """

    def __init__(self, binarize=False, fs=100, times=None, pedal_threshold=64):
        """Initialize the piano roll converter.
        
        Args:
            binarize (bool, optional): Whether to binarize the piano roll.
            fs (int, optional): Sampling frequency in Hz.
            times (array-like, optional): Times at which to sample.
            pedal_threshold (int, optional): Sustain pedal threshold.
        """
        super().__init__()
        self.binarize = binarize
        self.fs = fs
        self.times = times
        self.pedal_threshold = pedal_threshold

    def apply(self, midi_data):
        """Convert MIDI data to a piano roll NumPy array.
        
        Args:
            midi_data (pretty_midi.PrettyMIDI): The MIDI data to convert.
            
        Returns:
            np.ndarray: Piano roll array of shape (128, time_steps).
            
        Note:
            For more details on the piano roll format, see:
            https://craffel.github.io/pretty-midi/#pretty_midi.PrettyMIDI.get_piano_roll
        """
        return midi_data.get_piano_roll(fs=self.fs, times=self.times, pedal_threshold=self.pedal_threshold)


class ToPRollTensor(ToPRollNumpy):
    """Convert MIDI data to a piano roll PyTorch tensor.
    
    This converter transforms MIDI data into a piano roll representation as a
    2D PyTorch tensor of shape (128, time_steps). The tensor values represent note
    velocities at each time step.
    
    This class inherits from ToPRollNumpy and adds PyTorch-specific functionality,
    such as device placement.
    
    Args:
        binarize (bool, optional): Whether to binarize the piano roll.
            Default: False
        device (str, optional): PyTorch device to place the tensor on.
            Default: 'cpu'
        fs (int, optional): Sampling frequency in Hz.
            Default: 100
        times (array-like, optional): Times at which to sample the piano roll.
            Default: None
        pedal_threshold (int, optional): Threshold above which the sustain pedal
            is activated.
            Default: 64
            
    Example:
        >>> converter = ToPRollTensor(fs=200, device='cuda')
        >>> piano_roll = converter(pretty_midi_obj)  # Shape: (128, time_steps)
    """

    def __init__(self, binarize=False, device='cpu', fs=100, times=None, pedal_threshold=64):
        """Initialize the piano roll tensor converter.
        
        Args:
            binarize (bool, optional): Whether to binarize the piano roll.
            device (str, optional): PyTorch device to place the tensor on.
            fs (int, optional): Sampling frequency in Hz.
            times (array-like, optional): Times at which to sample.
            pedal_threshold (int, optional): Sustain pedal threshold.
        """
        self.device = device
        super().__init__(binarize=binarize, fs=fs, times=times, pedal_threshold=pedal_threshold)

    def apply(self, midi_data):
        """Convert MIDI data to a piano roll PyTorch tensor.
        
        Args:
            midi_data (pretty_midi.PrettyMIDI): The MIDI data to convert.
            
        Returns:
            torch.Tensor: Piano roll tensor of shape (128, time_steps).
            
        Note:
            For more details on the piano roll format, see:
            https://craffel.github.io/pretty-midi/#pretty_midi.PrettyMIDI.get_piano_roll
        """
        return torch.Tensor(super().apply(midi_data=midi_data)).to(self.device)

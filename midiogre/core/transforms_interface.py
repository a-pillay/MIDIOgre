"""Base interface for all MIDI data augmentation transforms.

This module provides the base class for implementing MIDI data augmentation transforms.
All transforms in MIDIOgre inherit from this class and must implement the `apply` method.

Example:
    >>> class MyTransform(BaseMidiTransform):
    ...     def __init__(self, p_instruments=1.0, p=0.5):
    ...         super().__init__(p_instruments=p_instruments, p=p)
    ...     
    ...     def apply(self, midi_data):
    ...         # Implement your transform logic here
    ...         return midi_data
"""

import logging
import random


class BaseMidiTransform:
    """Base class for all MIDI data augmentation transforms.
    
    This class provides common functionality for MIDI transforms including:
    - Probability handling for transform application
    - Instrument selection for multi-instrument MIDI files
    - Basic validation of transform parameters
    
    All transforms should inherit from this class and implement the `apply` method.
    
    Args:
        p_instruments (float): Probability of applying the transform to each instrument
            when multiple instruments are present. Must be in range [0, 1].
            Default: 1.0 (apply to all instruments)
        p (float): Probability of applying the transform to selected notes within
            each chosen instrument. Must be in range [0, 1].
            Default: 0.5
        eps (float, optional): Small epsilon value for numerical stability.
            Default: 1e-12
            
    Raises:
        ValueError: If p or p_instruments are not in range [0, 1]
    """
    def __init__(self, p_instruments: float, p: float, eps: float = 1e-12):
        if not 0 <= p <= 1:
            raise ValueError(
                "Probability of applying a MIDI Transform must be >=0 and <=1."
            )

        if not 0 <= p_instruments <= 1:
            raise ValueError(
                "Probability of applying transform on an instrument must be >=0 and <=1."
            )

        self.p = p
        self.p_instruments = p_instruments
        self.eps = eps

    def _get_modified_instruments_list(self, midi_data):
        """Get list of instruments to be modified based on p_instruments.
        
        This internal method handles the selection of instruments to be modified
        based on the p_instruments parameter. It filters out drum instruments and
        randomly selects instruments if p_instruments < 1.0.
        
        Args:
            midi_data: A PrettyMIDI object containing the MIDI data to transform.
            
        Returns:
            list: List of instruments selected for modification.
            
        Note:
            Currently drum instruments are filtered out by default.
        """
        # filtering out drum instruments (TODO: Evaluate whether this is needed)
        modified_instruments = [instrument for instrument in midi_data.instruments if not instrument.is_drum]

        if len(modified_instruments) == 0:
            logging.warning("MIDI file only contains drum tracks.")
        elif self.p_instruments < 1.0 and len(modified_instruments) > 1:
            num_modified_instruments = int(self.p_instruments * len(modified_instruments))
            if num_modified_instruments == 0:
                logging.debug(
                    "No instruments left to randomly modify in MIDI file. Skipping.",
                )
                return midi_data

            modified_instruments = random.sample(modified_instruments, k=num_modified_instruments)

        return modified_instruments

    def apply(self, midi_data):
        """Apply the transform to the MIDI data.
        
        This method should be implemented by all transform classes.
        
        Args:
            midi_data: A PrettyMIDI object containing the MIDI data to transform.
            
        Returns:
            The transformed PrettyMIDI object.
            
        Raises:
            NotImplementedError: If the child class does not implement this method.
        """
        raise NotImplementedError

    def __call__(self, midi_data):
        """Apply the transform to the MIDI data.
        
        This method makes the transform callable, allowing it to be used as a function.
        
        Args:
            midi_data: A PrettyMIDI object containing the MIDI data to transform.
            
        Returns:
            The transformed PrettyMIDI object.
        """
        return self.apply(midi_data)

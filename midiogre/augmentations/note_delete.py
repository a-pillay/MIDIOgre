"""MIDI note deletion augmentation module.

This module provides functionality to randomly remove notes from MIDI tracks.
The deletion process can be configured to affect a certain percentage of notes
across selected instruments, allowing for controlled sparsification of the music.

.. image:: ../../demo/plots/notedelete.png
   :alt: Visualization of NoteDelete transform
   :align: center
   :width: 100%
   :class: transform-viz

Example:
    >>> from midiogre.augmentations import NoteDelete
    >>> import pretty_midi
    >>> 
    >>> # Create transform that randomly removes up to 20% of notes
    >>> transform = NoteDelete(p=0.2)
    >>> 
    >>> # Load and transform MIDI file
    >>> midi_data = pretty_midi.PrettyMIDI('song.mid')
    >>> transformed = transform(midi_data)
"""

import logging
import math
import random
from operator import itemgetter

import numpy as np

from midiogre.core.transforms_interface import BaseMidiTransform


class NoteDelete(BaseMidiTransform):
    """Transform for randomly deleting MIDI notes.
    
    This transform allows for random deletion of existing notes from MIDI tracks.
    The deletion is applied with probability p to each selected instrument.
    For each selected instrument, the number of notes to delete is randomly chosen
    between 0 and p * (current number of notes).
    
    Args:
        p_instruments (float, optional): If a MIDI file has multiple instruments, this
            determines the probability of applying the transform to each
            instrument. Must be in range [0, 1].
            Default: 1.0 (apply to all instruments)
        p (float, optional): For each selected instrument, this determines the maximum
            ratio of notes that may be deleted. Must be in range [0, 1].
            Example: If p=0.2 and an instrument has 100 notes, up to 20 notes
            may be deleted.
            Default: 0.2
        eps (float, optional): Small epsilon value for numerical stability.
            Default: 1e-12
            
    Raises:
        ValueError: If any of the following conditions are met:
            - p_instruments not in range [0, 1]
            - p not in range [0, 1]
            
    Example:
        >>> # Create transform that aggressively thins out notes
        >>> transform = NoteDelete(
        ...     p_instruments=0.8,  # Apply to 80% of instruments
        ...     p=0.4,  # Delete up to 40% of notes
        ... )
        >>> transformed = transform(midi_data)
    """

    def __init__(self, p_instruments: float = 1.0, p: float = 0.2, eps: float = 1e-12):
        """Initialize the NoteDelete transform.
        
        Args:
            p_instruments (float, optional): Probability of applying to each instrument.
                Default: 1.0
            p (float, optional): Maximum ratio of notes that may be deleted.
                Default: 0.2
            eps (float, optional): Small epsilon value for numerical stability.
                Default: 1e-12
                
        Raises:
            ValueError: If parameters are invalid (see class docstring for details).
        """
        super().__init__(p_instruments=p_instruments, p=p, eps=eps)

    def apply(self, midi_data):
        """Apply the note deletion transformation to the MIDI data.
        
        For each non-drum instrument selected based on p_instruments, this method:
        1. Determines the number of notes to delete based on p
        2. Randomly selects notes for deletion
        3. Removes the selected notes from the instrument track
        
        Args:
            midi_data (pretty_midi.PrettyMIDI): The MIDI data to transform.
            
        Returns:
            pretty_midi.PrettyMIDI: The transformed MIDI data with notes deleted.
            
        Note:
            - Drum instruments are skipped by default
            - The number of notes deleted is random but proportional to existing notes
            - Notes are selected for deletion uniformly at random
            - Empty instruments (no notes) are skipped
        """
        modified_instruments = self._get_modified_instruments_list(midi_data)
        for instrument in modified_instruments:
            if not instrument.notes:  # Skip empty instruments
                continue
            num_notes_to_delete = math.ceil(np.random.uniform(self.eps, self.p) * len(instrument.notes))
            if num_notes_to_delete > 0:  # Only delete if we need to
                indices_to_keep = np.random.choice(
                    len(instrument.notes),
                    size=len(instrument.notes) - num_notes_to_delete,
                    replace=False
                )
                instrument.notes = [instrument.notes[i] for i in sorted(indices_to_keep)]
        return midi_data

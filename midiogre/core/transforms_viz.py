"""MIDI transform visualization tools.

This module provides tools for visualizing the effects of MIDI transforms. It includes
functions for loading MIDI files, converting them to piano roll format, and creating
side-by-side visualizations of original and transformed MIDI data.

The visualizations use matplotlib to create color-coded piano roll representations,
where the intensity of the color indicates the velocity of the notes.

Example:
    >>> from midiogre.core.transforms_viz import load_midi, viz_transform
    >>> from midiogre.augmentations import PitchShift
    >>> 
    >>> # Load MIDI file and create transform
    >>> midi_data = load_midi('song.mid')
    >>> transform = PitchShift(max_shift=2, p=1.0)
    >>> 
    >>> # Apply transform and visualize
    >>> transformed = transform(midi_data)
    >>> viz_transform(midi_data, transformed, 'Pitch Shift')
"""

import copy
import time
from statistics import mean

import matplotlib
import numpy as np
import pretty_midi
from matplotlib import pyplot as plt

from midiogre.core.conversions import ConvertToMido, ConvertToPrettyMIDI
from midiogre.augmentations import PitchShift, OnsetTimeShift, DurationShift, NoteDelete, NoteAdd, TempoShift
from midiogre.core import ToPRollTensor, Compose


def load_midi(path: str) -> pretty_midi.PrettyMIDI:
    """Load a MIDI file from disk.
    
    Args:
        path (str): Path to the MIDI file to load.
        
    Returns:
        pretty_midi.PrettyMIDI: The loaded MIDI data.
        
    Note:
        This function strips any whitespace from the path before loading.
    """
    return pretty_midi.PrettyMIDI(path.strip())


def truncate_midi(midi_data, max_notes):
    for instrument in midi_data.instruments:
        instrument.notes = instrument.notes[:max_notes]
        end_time = instrument.notes[-1].end
        instrument.pitch_bends = list(filter(lambda x: x.time <= end_time, instrument.pitch_bends))
        instrument.control_changes = list(filter(lambda x: x.time <= end_time, instrument.control_changes))

    return midi_data


def save_midi(midi_data, destination_path):
    midi_data.write(destination_path.strip())


def get_piano_roll(midi_data):
    return midi_data.get_piano_roll(fs=100)


def create_proll_cmap(cmap_name: str) -> matplotlib.colors.ListedColormap:
    """Create a colormap for piano roll visualization with alpha channel.
    
    This function creates a colormap that varies both in color and opacity,
    making it suitable for overlaying multiple piano rolls in the same plot.
    
    Args:
        cmap_name (str): Name of the base matplotlib colormap to use.
        
    Returns:
        matplotlib.colors.ListedColormap: A new colormap with alpha channel
        that varies from transparent to opaque.
    """
    cmap = matplotlib.colormaps[cmap_name]
    alpha_cmap = cmap(np.arange(cmap.N))
    alpha_cmap[:, -1] = np.linspace(0, 1, cmap.N)
    alpha_cmap = matplotlib.colors.ListedColormap(alpha_cmap)
    return alpha_cmap


def viz_transform(original_midi_data: pretty_midi.PrettyMIDI,
                 transformed_proll: np.ndarray,
                 transform_name: str):
    """Visualize the effect of a MIDI transform.
    
    Creates a side-by-side visualization comparing the original MIDI data
    with the transformed version. The visualization uses piano roll format
    with different colors for original and transformed data.
    
    Args:
        original_midi_data (pretty_midi.PrettyMIDI): The original MIDI data.
        transformed_proll (np.ndarray): Piano roll representation of the
            transformed MIDI data.
        transform_name (str): Name of the transform for the plot title.
            
    Note:
        - Original data is shown in red
        - Transformed data is shown in blue
        - Color intensity indicates note velocity
        - Both piano rolls are overlaid in the same plot for easy comparison
    """
    fig, ax = plt.subplots(nrows=1, ncols=1)

    original_proll = get_piano_roll(original_midi_data)

    cmap1 = create_proll_cmap('Reds')
    cmap2 = create_proll_cmap('Blues')

    hmap1 = ax.pcolor(original_proll, cmap=cmap1)
    cbar1 = plt.colorbar(hmap1, aspect=50)
    cbar1.ax.set_ylabel('Original')

    hmap2 = ax.pcolor(transformed_proll, cmap=cmap2)
    cbar2 = plt.colorbar(hmap2, aspect=50)
    cbar2.ax.set_ylabel('Transformed')

    ax.set_xlabel("Time Unit")
    ax.set_ylabel("Midi Note")

    plt.title('{}: Original v/s Transformed'.format(transform_name))
    plt.show()
    plt.cla()


if __name__ == '__main__':
    # Example usage of the visualization tools
    midi_data = load_midi('../assets/example.mid')
    midi_data = truncate_midi(midi_data, 100)
    save_midi(midi_data, '../assets/short.mid')

    # Create a transform pipeline with various augmentations
    midi_transform = Compose([
        ConvertToMido(),
        TempoShift(max_shift=10, mode='down', tempo_range=(30.0, 200.0), p=0.1),
        ConvertToPrettyMIDI(),
        PitchShift(max_shift=5, mode='both', p_instruments=1.0, p=0.1),
        OnsetTimeShift(max_shift=1.2, mode='both', p_instruments=1.0, p=0.1),
        DurationShift(max_shift=0.5, mode='both', p_instruments=1.0, p=0.1),
        NoteDelete(p_instruments=1.0, p=0.1),
        NoteAdd(note_num_range=(50, 80), note_velocity_range=(20, 120), note_duration_range=(0.5, 1.5),
                restrict_to_instrument_time=True, p_instruments=1.0, p=0.1),

        # ToPRollTensor(device='cpu')
    ])

    # Benchmark transform pipeline
    num_iters = 5
    durns = []
    for i in range(num_iters):
        transformed_midi_data = copy.deepcopy('../assets/example.mid')
        overall_start = time.time()
        transformed_midi_data = midi_transform(transformed_midi_data)
        durns.append(time.time() - overall_start)
    print("Mean time taken for {} iters of {} MIDIOgre transforms = {}s".format(num_iters, len(midi_transform),
                                                                                mean(durns)))

    # Save and visualize results
    save_midi(transformed_midi_data, '../assets/short_transformed.mid')
    transformed_midi_data = truncate_midi(transformed_midi_data, 100)
    viz_transform(midi_data, get_piano_roll(transformed_midi_data), 'After MIDIOgre Augmentations')

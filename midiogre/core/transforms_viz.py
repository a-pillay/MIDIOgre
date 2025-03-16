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
from typing import Optional, Tuple

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


def compute_transform_stats(original_midi: pretty_midi.PrettyMIDI, 
                          transformed_midi: pretty_midi.PrettyMIDI) -> dict:
    """Compute statistics comparing original and transformed MIDI.
    
    Args:
        original_midi: Original MIDI data
        transformed_midi: Transformed MIDI data
        
    Returns:
        dict: Statistics about the transformation including:
            - Note count difference
            - Average pitch difference
            - Average velocity difference
            - Average duration difference
    """
    stats = {}
    
    # Get all notes from both MIDIs
    orig_notes = [note for inst in original_midi.instruments for note in inst.notes]
    trans_notes = [note for inst in transformed_midi.instruments for note in inst.notes]
    
    # Basic statistics
    stats['note_count_diff'] = len(trans_notes) - len(orig_notes)
    
    if orig_notes and trans_notes:
        # Compute average differences
        stats['avg_pitch_diff'] = mean([n.pitch for n in trans_notes]) - mean([n.pitch for n in orig_notes])
        stats['avg_velocity_diff'] = mean([n.velocity for n in trans_notes]) - mean([n.velocity for n in orig_notes])
        stats['avg_duration_diff'] = mean([n.end - n.start for n in trans_notes]) - mean([n.end - n.start for n in orig_notes])
    
    return stats


def viz_transform(original_midi_data: pretty_midi.PrettyMIDI,
                 transformed_midi_data: pretty_midi.PrettyMIDI,
                 transform_name: str,
                 save_path: Optional[str] = None):
    """Visualize the effect of a MIDI transform.
    
    Creates a side-by-side visualization comparing the original MIDI data
    with the transformed version. The visualization uses piano roll format
    with different colors:
    - Blue: Original/unchanged notes
    - Gray: Deleted notes
    - Green: Added notes
    
    Args:
        original_midi_data: The original MIDI data
        transformed_midi_data: The transformed MIDI data
        transform_name: Name of the transform for the plot title
        save_path: Optional path to save the visualization
    """
    fig = plt.figure(figsize=(15, 6))
    
    # Create gridspec with space for one colorbar
    gs = plt.GridSpec(1, 3, width_ratios=[10, 10, 0.4])
    ax1 = fig.add_subplot(gs[0])  # Original plot
    ax2 = fig.add_subplot(gs[1])  # Transformed plot
    cax = fig.add_subplot(gs[2])  # Single colorbar for velocity
    
    # Get piano rolls
    original_proll = get_piano_roll(original_midi_data)
    transformed_proll = get_piano_roll(transformed_midi_data)
    
    # Pad piano rolls to the same length
    max_len = max(original_proll.shape[1], transformed_proll.shape[1])
    if original_proll.shape[1] < max_len:
        pad_width = ((0, 0), (0, max_len - original_proll.shape[1]))
        original_proll = np.pad(original_proll, pad_width, mode='constant')
    if transformed_proll.shape[1] < max_len:
        pad_width = ((0, 0), (0, max_len - transformed_proll.shape[1]))
        transformed_proll = np.pad(transformed_proll, pad_width, mode='constant')
    
    # Find actual note range
    all_notes = np.where(np.logical_or(original_proll > 0, transformed_proll > 0))[0]
    if len(all_notes) > 0:
        min_note = max(0, min(all_notes) - 2)  # Add padding of 2 notes
        max_note = min(127, max(all_notes) + 3)  # Add padding of 2 notes
    else:
        min_note, max_note = 0, 127
    
    # Calculate differences for visualization
    delta_proll = transformed_proll - original_proll
    
    # Plot original (in blue)
    im1 = ax1.pcolor(original_proll, cmap='Blues', vmin=0, vmax=127)
    ax1.set_title('Original')
    ax1.set_xlabel('Time (100 ticks per beat)')
    ax1.set_ylabel('MIDI Note Number')
    ax1.set_ylim(min_note, max_note)
    
    # Create masks for unchanged, added, and deleted notes
    unchanged_mask = delta_proll == 0
    added_mask = delta_proll > 0
    deleted_mask = delta_proll < 0
    
    # Plot unchanged notes in blue
    unchanged = np.ma.masked_where(~unchanged_mask, transformed_proll)
    im2 = ax2.pcolor(unchanged, cmap='Blues', vmin=0, vmax=127)
    
    # Plot added notes in green
    if not np.all(~added_mask):
        added = np.ma.masked_where(~added_mask, transformed_proll)
        im3 = ax2.pcolor(added, cmap='Greens', vmin=0, vmax=127)
    
    # Plot deleted notes in gray
    if not np.all(~deleted_mask):
        deleted = np.ma.masked_where(~deleted_mask, original_proll)
        im4 = ax2.pcolor(deleted, cmap='Greys', vmin=0, vmax=127, alpha=0.5)
    
    ax2.set_title('Transformed (with changes highlighted)')
    ax2.set_xlabel('Time (100 ticks per beat)')
    ax2.set_ylabel('MIDI Note Number')
    ax2.set_ylim(min_note, max_note)
    
    # Add single colorbar for velocity
    plt.colorbar(im1, cax=cax, label='Velocity')
    
    # Add a small legend for the colors
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.7, label='Original/Unchanged'),
        plt.Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.7, label='Added'),
        plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.5, label='Deleted')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, -0.1),
               ncol=3, fontsize=8)
    
    # Main title
    fig.suptitle(f'MIDI Transform: {transform_name}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()
    plt.close()


if __name__ == '__main__':
    # Example usage showing different types of transformations
    midi_data = load_midi('../assets/example.mid')
    midi_data = truncate_midi(midi_data, 100)
    
    # Example 1: Pitch Shift
    pitch_transform = Compose([PitchShift(max_shift=5, mode='both', p=1.0)])
    transformed = pitch_transform(copy.deepcopy(midi_data))
    viz_transform(midi_data, transformed, 'Pitch Shift')
    
    # Example 2: Note Addition/Deletion
    note_transform = Compose([
        NoteDelete(p=0.2),
        NoteAdd(note_num_range=(50, 80), p=0.3)
    ])
    transformed = note_transform(copy.deepcopy(midi_data))
    viz_transform(midi_data, transformed, 'Note Modification')
    
    # Example 3: Time-based transforms
    time_transform = Compose([
        OnsetTimeShift(max_shift=0.5, mode='both', p=1.0),
        DurationShift(max_shift=0.2, mode='both', p=1.0)
    ])
    transformed = time_transform(copy.deepcopy(midi_data))
    viz_transform(midi_data, transformed, 'Time Modification')

"""Generate example MIDI transformations and visualizations.

This script demonstrates various MIDI transformations available in MIDIOgre by:
1. Loading sample MIDI files
2. Applying different transformations
3. Saving the transformed MIDI files
4. Generating visualization plots comparing original and transformed versions

The results are saved in the following structure:
- demo/midi/original/: Original MIDI files
- demo/midi/transformed/: Transformed MIDI files
- demo/plots/: Visualization plots
"""

import os
from pathlib import Path
import copy

import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import mido

from midiogre.core import Compose
from midiogre.core.transforms_viz import (
    load_midi, 
    save_midi, 
    viz_transform,
    plot_transformed_piano_roll
)
from midiogre.core.conversions import ConvertToMido, ConvertToPrettyMIDI
from midiogre.augmentations import (
    PitchShift, 
    OnsetTimeShift, 
    DurationShift, 
    NoteDelete, 
    NoteAdd,
    TempoShift
)

# Create demo directories if they don't exist
DEMO_DIR = Path("demo")
MIDI_ORIG_DIR = DEMO_DIR / "midi" / "original"
MIDI_TRANS_DIR = DEMO_DIR / "midi" / "transformed"
PLOTS_DIR = DEMO_DIR / "plots"

for dir_path in [MIDI_ORIG_DIR, MIDI_TRANS_DIR, PLOTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

def create_sample_midi(filename: str, duration: float = 4.0) -> pretty_midi.PrettyMIDI:
    """Create a simple sample MIDI file with a melodic pattern."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)  # Set initial tempo to 120 BPM
    piano = pretty_midi.Instrument(program=0)
    
    # Create a repeating pattern to better show tempo changes
    notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C major scale
    for repeat in range(4):  # Repeat the pattern 4 times
        for i, note_num in enumerate(notes):
            note = pretty_midi.Note(
                velocity=100,
                pitch=note_num,
                start=repeat * 4.0 + i * 0.5,  # Each repeat starts 4 seconds later
                end=repeat * 4.0 + (i + 1) * 0.5
            )
            piano.notes.append(note)
    
    midi.instruments.append(piano)
    return midi

def generate_examples():
    """Generate example transformations and save results."""
    
    # Create and save sample MIDI file
    sample_midi = create_sample_midi("sample")
    orig_path = MIDI_ORIG_DIR / "sample.mid"
    save_midi(sample_midi, str(orig_path))
    
    # Define transformations to demonstrate with their class names
    transformations = {
        "PitchShift": Compose([
            PitchShift(max_shift=4, mode='up', p=0.3)
        ]),
        "OnsetTimeShift": Compose([
            OnsetTimeShift(max_shift=0.3, mode='both', p=0.3)
        ]),
        "DurationShift": Compose([
            DurationShift(max_shift=0.2, mode='both', p=0.3)
        ]),
        "NoteDelete": Compose([
            NoteDelete(p=1.0)
        ]),
        "NoteAdd": Compose([
            NoteAdd(
                note_num_range=(60, 72),
                note_velocity_range=(80, 100),
                note_duration_range=(0.25, 0.75),
                p=1.0
            )
        ]),
        "TempoShift": Compose([
            TempoShift(max_shift=60, mode='both', p=1.0)  # More dramatic tempo changes
        ])
    }
    
    # Apply each transformation and save results
    midi_data = load_midi(str(orig_path))
    transformed_midis = {}  # Store transformed MIDI files for combined plot
    
    for transform_name, transform in transformations.items():
        print(f"Generating {transform_name} example...")
        
        # Apply transformation
        midi_copy = copy.deepcopy(midi_data)
        
        if transform_name == "TempoShift":
            # Save and load as Mido for tempo shift
            temp_path = MIDI_TRANS_DIR / "temp.mid"
            save_midi(midi_copy, str(temp_path))
            mido_data = mido.MidiFile(str(temp_path))
            transformed_mido = transform(mido_data)
            transformed_mido.save(str(temp_path))  # Use Mido's save method
            transformed = pretty_midi.PrettyMIDI(str(temp_path))  # Load with new tempo
            temp_path.unlink()  # Clean up temp file
        else:
            transformed = transform(midi_copy)
        
        # Save transformed MIDI
        trans_midi_path = MIDI_TRANS_DIR / f"{transform_name.lower()}.mid"
        save_midi(transformed, str(trans_midi_path))
        
        # Save visualization
        plot_path = PLOTS_DIR / f"{transform_name.lower()}.png"
        viz_transform(midi_data, transformed, transform_name, str(plot_path))
        
        transformed_midis[transform_name] = transformed
        print(f"Saved: {trans_midi_path.name} and {plot_path.name}")

    # Create combined plot with 2x3 grid
    fig = plt.figure(figsize=(18, 10))  # Made figure wider to accommodate colorbar
    
    # Create gridspec with space for colorbar
    gs = plt.GridSpec(2, 4, width_ratios=[10, 10, 10, 1])  # Added extra column for colorbar
    
    # Plot each transformation in a grid
    for i, (name, midi) in enumerate(transformed_midis.items()):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        im = plot_transformed_piano_roll(ax, midi_data, midi, name)
        
        # Only show y-label for leftmost plots
        if col != 0:
            ax.set_ylabel('')
        
        # Only show x-label for bottom plots
        if row != 1:
            ax.set_xlabel('')
    
    # Add colorbar in the rightmost column, spanning both rows
    cax = fig.add_subplot(gs[:, -1])
    plt.colorbar(im, cax=cax, label='Velocity')
    
    # Add legend below the plots
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.7, label='Original/Unchanged'),
        plt.Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.7, label='Added'),
        plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.5, label='Deleted')
    ]
    fig.legend(handles=legend_elements, 
              loc='center',
              bbox_to_anchor=(0.5, -0.05),  # Place below plots
              ncol=3,
              fontsize=10)
    
    plt.suptitle('MIDIOgre Augmentations', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save combined plot with extra space for legend
    combined_plot_path = PLOTS_DIR / "combined.png"
    plt.savefig(str(combined_plot_path), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved combined plot: {combined_plot_path.name}")

if __name__ == "__main__":
    print("Generating MIDIOgre transformation examples...")
    generate_examples()
    print("\nDone! Check the demo folder for results.") 
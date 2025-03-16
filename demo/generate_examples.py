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

import pretty_midi

from midiogre.core import Compose
from midiogre.core.transforms_viz import load_midi, save_midi, viz_transform
from midiogre.augmentations import (
    PitchShift, 
    OnsetTimeShift, 
    DurationShift, 
    NoteDelete, 
    NoteAdd
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
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    
    # Create a simple ascending pattern
    notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C major scale
    for i, note_num in enumerate(notes):
        note = pretty_midi.Note(
            velocity=100,
            pitch=note_num,
            start=i * 0.5,
            end=(i + 1) * 0.5
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
    
    # Define transformations to demonstrate
    transformations = {
        "pitch_shift": Compose([
            PitchShift(max_shift=4, mode='up', p=1.0)
        ]),
        "note_modification": Compose([
            NoteDelete(p=0.2),
            NoteAdd(
                note_num_range=(60, 72),
                note_velocity_range=(80, 100),
                note_duration_range=(0.25, 0.75),
                p=0.3
            )
        ]),
        "time_modification": Compose([
            OnsetTimeShift(max_shift=0.3, mode='both', p=1.0),
            DurationShift(max_shift=0.2, mode='both', p=1.0)
        ]),
        "combined": Compose([
            PitchShift(max_shift=2, mode='both', p=0.5),
            OnsetTimeShift(max_shift=0.2, mode='both', p=0.5),
            NoteDelete(p=0.1),
            NoteAdd(
                note_num_range=(60, 72),
                note_velocity_range=(80, 100),
                note_duration_range=(0.25, 0.75),
                p=0.2
            )
        ])
    }
    
    # Apply each transformation and save results
    midi_data = load_midi(str(orig_path))
    
    for transform_name, transform in transformations.items():
        print(f"Generating {transform_name} example...")
        
        # Apply transformation
        transformed = transform(copy.deepcopy(midi_data))
        
        # Save transformed MIDI
        trans_midi_path = MIDI_TRANS_DIR / f"{transform_name}.mid"
        save_midi(transformed, str(trans_midi_path))
        
        # Save visualization
        plot_path = PLOTS_DIR / f"{transform_name}.png"
        viz_transform(midi_data, transformed, transform_name.replace('_', ' ').title(), str(plot_path))
        
        print(f"Saved: {trans_midi_path.name} and {plot_path.name}")

if __name__ == "__main__":
    print("Generating MIDIOgre transformation examples...")
    generate_examples()
    print("\nDone! Check the demo folder for results.") 
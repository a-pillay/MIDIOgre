# MIDIOgre Transformation Examples

This demo folder contains examples of various MIDI transformations available in MIDIOgre. It includes sample MIDI files, their transformed versions, and visualization plots showing the effects of different transformations.

## Directory Structure

```
demo/
├── midi/
│   ├── original/      # Original MIDI files
│   └── transformed/   # Transformed versions of the MIDI files
├── plots/             # Visualization plots comparing original and transformed MIDI
└── generate_examples.py   # Script to generate all examples
```

## Available Examples

The demo includes the following transformation examples:

1. **Pitch Shift**: Shifts notes up by 4 semitones
2. **Note Modification**: Demonstrates note deletion and addition
3. **Time Modification**: Shows onset time shifts and duration changes
4. **Combined**: Shows multiple transformations applied together:
   - Pitch shifting
   - Time shifting
   - Note deletion and addition

Each example includes:
- Original MIDI file
- Transformed MIDI file
- Visualization plot comparing the original and transformed versions

## Running the Examples

To generate all examples:

```bash
python demo/generate_examples.py
```

This will:
1. Create a sample MIDI file with a simple melodic pattern
2. Apply various transformations
3. Save the transformed MIDI files
4. Generate visualization plots

## Visualization Details

The visualization plots show:
- Side-by-side piano roll representations
- Original version in red, transformed in blue
- Note velocity indicated by color intensity
- Statistical information about the transformation

## Using the Examples

These examples can be used to:
- Understand how different transformations affect MIDI data
- Test and verify transformation behavior
- Demonstrate MIDIOgre's capabilities
- Guide the development of custom transformations 
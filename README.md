# MIDIOgre

![GitHub stars](https://img.shields.io/github/stars/a-pillay/MIDIOgre.svg?style=flat-square)
![GitHub forks](https://img.shields.io/github/forks/a-pillay/MIDIOgre.svg?style=flat-square)
![GitHub license](https://img.shields.io/github/license/a-pillay/MIDIOgre.svg?style=flat-square)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

MIDIOgre is a powerful Python library for performing data augmentations on MIDI inputs, primarily designed for machine learning models operating on MIDI data. With MIDIOgre, you can easily generate variations of MIDI sequences to enrich your training data and improve the robustness and generalization of your models.

![A plot of implemented MIDIOgre augmentations.](https://raw.githubusercontent.com/a-pillay/MIDIOgre/main/demo/plots/combined.png)

## Features

- **Comprehensive MIDI Augmentations**: A wide range of transformations including pitch shifting, onset time modification, duration changes, and more
- **Easy Integration**: Simple API designed to work seamlessly with machine learning workflows
- **Customizable**: Flexible parameters for fine-tuning augmentations to your needs
- **Efficient**: Optimized for handling large MIDI datasets

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from PyPI

```bash
pip install midiogre
```

### Install from source (for development)

```bash
# Clone the repository
git clone https://github.com/a-pillay/MIDIOgre.git
cd MIDIOgre

# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

> **Note**: MIDIOgre uses the development version of `pretty-midi` directly from GitHub for enhanced functionality. This is handled automatically by the installation process.

## Quick Start

```python
from midiogre.augmentations import PitchShift, OnsetTimeShift, NoteDelete
from midiogre.core import Compose
import pretty_midi

# Basic usage - single file augmentation
midi_data = pretty_midi.PrettyMIDI('input.mid')
transform = Compose([
    PitchShift(max_shift=3, mode='both', p=0.8),
    OnsetTimeShift(max_shift=0.1, mode='both', p=0.5)
])
augmented = transform(midi_data)
augmented.write('output.mid')

# Integration with ML pipelines
class MIDIDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        # Your MIDI loading logic here
        midi_data = load_midi(idx)
        return self.transform(midi_data) if self.transform else midi_data

# Define augmentation pipeline for training
transform = Compose([
    PitchShift(max_shift=3, mode='both', p=0.8),      # Randomly transpose by ±3 semitones
    OnsetTimeShift(max_shift=0.1, mode='both', p=0.5), # Shift note timings by up to 100ms
    NoteDelete(p=0.3)                                  # Randomly remove up to 30% of notes
])

# Use in your training pipeline
train_dataset = MIDIDataset(transform=transform)
val_dataset = MIDIDataset(transform=None)  # No augmentation for validation
```

## Available Augmentations

### Currently Implemented

- **PitchShift**: Transpose MIDI note values of selected instruments
- **OnsetTimeShift**: Modify note onset times while preserving durations
- **DurationShift**: Alter note durations while maintaining onset times
- **NoteDelete**: Remove notes from instrument tracks
- **NoteAdd**: Add new notes to instrument tracks
- **TempoShift**: Modify the global tempo of MIDI files

### Planned Features

- **NoteSplit**: Split notes into multiple segments
- **VelocityShift**: Modify MIDI note velocities
- Swing-based augmentations
- MIDI CC based augmentations
- Semantically-meaningful augmentations (respecting rhythms & beats)

## Development

Please note that this project is developed with the assistance of [Cursor](https://cursor.sh/), mostly for runtime optimizations, unit-testing, documentation and build pipelines.

### Setting up for development

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install documentation dependencies (if working on docs)
pip install -r requirements-docs.txt
```

### Running Tests

```bash
pytest tests/
# For coverage report
pytest --cov=midiogre tests/
```

## Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the tests to ensure everything works
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

Areas where we particularly welcome contributions:
- Comprehensive unit tests
- Documentation improvements
- New augmentation techniques
- Performance optimizations
- Bug fixes

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use MIDIOgre in your research, please cite:

```bibtex
@software{midiogre2024,
  author = {Pillay, A},
  title = {MIDIOgre: MIDI Data Augmentation Library},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/a-pillay/MIDIOgre}
}
```

## Acknowledgments

- Inspired by [mdtk](https://github.com/JamesOwers/midi_degradation_toolkit)
- Built with [pretty-midi](https://github.com/craffel/pretty-midi)

## Contact

For questions, suggestions, or collaboration opportunities, please reach out via GitHub Issues.

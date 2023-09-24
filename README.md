# MIDIOgre

![GitHub stars](https://img.shields.io/github/stars/a-pillay/MIDIOgre.svg?style=flat-square)
![GitHub forks](https://img.shields.io/github/forks/a-pillay/MIDIOgre.svg?style=flat-square)
![GitHub license](https://img.shields.io/github/license/a-pillay/MIDIOgre.svg?style=flat-square)

MIDIOgre is a Python library designed for performing data augmentations on MIDI inputs, primarily for machine learning
models operating on MIDI data. With MIDIOgre, you can easily generate variations of MIDI sequences to enrich your
training data and improve the robustness and generalization of your models.

![Demo Plot of MIDIOgre Transformations](https://github.com/a-pillay/MIDIOgre/blob/main/docs/plot_ps_ots.png)

## Augmentation Functions

### Implemented
- **PitchShift**: Randomly transpose (pitch shift) MIDI note values of randomly selected instruments in a MIDI file.
- **OnsetTimeShift**: Randomly modify MIDI note onset times while keeping their total durations intact.

### Envisaged
- **DurationShift**: Randomly modify MIDI note durations while keeping their onset times intact.
- **NoteDelete**: Randomly remove a few notes from a MIDI instrument track.
- **NoteAdd**: Randomly add a few notes to a MIDI instrument track.
- **NoteSplit**: Randomly split some notes in a MIDI instrument track to a random number of chunks.

_(Some of these have been inspired from [mdtk](https://github.com/JamesOwers/midi_degradation_toolkit))_


## Note

This work is highly primitive at the moment as is undergoing active development. I intend to include more augmentations
shortly.

## Contributing

Contributions to MIDIOgre are welcome! If you'd like to contribute, please reach out to me via email (refer the link to
my website mentioned on my GitHub profile).

## License

This project is licensed under the MIT License - see
the [LICENSE](https://github.com/a-pillay/MIDIOgre/blob/main/LICENSE) file for details.


[//]: # (## Table of Contents)

[//]: # ()

[//]: # (- [Features]&#40;#features&#41;)

[//]: # (- [Installation]&#40;#installation&#41;)

[//]: # (- [Getting Started]&#40;#getting-started&#41;)

[//]: # (- [Usage]&#40;#usage&#41;)

[//]: # (- [Contributing]&#40;#contributing&#41;)

[//]: # (- [License]&#40;#license&#41;)

[//]: # ()

[//]: # (## Features)

[//]: # ()

[//]: # (- **MIDI Data Augmentation**: Apply various data augmentations to MIDI sequences.)

[//]: # (- **Customizable**: Easily configure augmentation parameters to suit your needs.)

[//]: # (- **Data Enrichment**: Enhance your MIDI dataset for better model training.)

[//]: # (- **Pythonic API**: Simple and intuitive Python API for easy integration.)

[//]: # (- **Examples**: Includes example scripts to help you get started quickly.)

[//]: # ()

[//]: # (## Installation)

[//]: # ()

[//]: # (You can install MIDIOgre using pip:)

[//]: # ()

[//]: # (```bash)

[//]: # (pip install midiogre)

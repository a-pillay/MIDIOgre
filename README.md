# MIDIOgre

![GitHub stars](https://img.shields.io/github/stars/a-pillay/MIDIOgre.svg?style=flat-square)
![GitHub forks](https://img.shields.io/github/forks/a-pillay/MIDIOgre.svg?style=flat-square)
![GitHub license](https://img.shields.io/github/license/a-pillay/MIDIOgre.svg?style=flat-square)

MIDIOgre is a Python library designed for performing data augmentations on MIDI inputs, primarily for machine learning
models operating on MIDI data. With MIDIOgre, you can easily generate variations of MIDI sequences to enrich your
training data and improve the robustness and generalization of your models.

![Demo Plot of MIDIOgre Transformations](https://github.com/a-pillay/MIDIOgre/blob/main/docs/plot_ps_ots_ds.png)

## Augmentation Functions

### Implemented

- **PitchShift**: Randomly transpose (pitch shift) MIDI note values of randomly selected instruments in a MIDI file.
- **OnsetTimeShift**: Randomly modify MIDI note onset times while keeping their total durations intact.
- **DurationShift**: Randomly modify MIDI note durations while keeping their onset times intact.
- **NoteDelete**: Randomly delete some notes from a MIDI instrument track.
- **NoteAdd**: Randomly add some notes to a MIDI instrument track.

### Envisaged

- **NoteSplit**: Randomly split some notes in a MIDI instrument track to a random number of chunks.
- **VelocityShift**: Randomly alter MIDI note velocities of randomly selected instruments in a MIDI file.
- _Swing-based augmentations_
- _MIDI CC based augmentations_
- _Semantically-meaningful augmentations (respects rhythms & beats)_
- _Suggestions welcome for more!_

_(Some of these have been inspired from [mdtk](https://github.com/JamesOwers/midi_degradation_toolkit))_

## Note

This work is highly primitive at the moment and is undergoing active development. As of now, it is simply a showcase of
work and is not ready to be integrated into your ML workflow. However, I intend to get it usable on priority.

## Contributing

Contributions to MIDIOgre are welcome! If you'd like to contribute, please reach out to me via email (_<
my_github_id_without_hyphens>@cmu.edu_).

Some areas I could use help (will update this progressively):

- Writing extensive unit tests (not simply for the sake of maximizing code coverage).
- Documentation (improving existing docs & adding more info to cater to developers from all kinds of music theory &
  programming backgrounds).

## License

This project is licensed under the MIT License - see
the [LICENSE](https://github.com/a-pillay/MIDIOgre/blob/main/LICENSE) file for details.

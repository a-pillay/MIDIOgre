Examples
========

Basic Usage
----------

Here's a simple example of using MIDIOgre to augment MIDI data:

.. code-block:: python

    from midiogre.augmentations import PitchShift, OnsetTimeShift
    from midiogre.core import Compose
    import pretty_midi

    # Load MIDI file
    midi_data = pretty_midi.PrettyMIDI('song.mid')

    # Create a transform pipeline
    transform = Compose([
        PitchShift(max_shift=2, p=0.5),  # Shift pitches by up to 2 semitones
        OnsetTimeShift(max_shift=0.1, p=0.3)  # Shift timing by up to 0.1 seconds
    ])

    # Apply transforms
    transformed = transform(midi_data)

    # Save the result
    transformed.write('transformed_song.mid')

Format Conversion
---------------

MIDIOgre supports seamless conversion between different MIDI formats:

.. code-block:: python

    from midiogre.core.conversions import ConvertToMido, ConvertToPrettyMIDI, ToPRollTensor

    # Convert to Mido format
    mido_obj = ConvertToMido()('song.mid')

    # Convert to PrettyMIDI
    pretty_midi_obj = ConvertToPrettyMIDI()(mido_obj)

    # Convert to piano roll tensor
    piano_roll = ToPRollTensor()(pretty_midi_obj) 
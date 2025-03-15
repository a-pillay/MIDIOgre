from unittest.mock import Mock
import pretty_midi


def generate_mock_midi_data(num_notes):
    """Generate a mock MIDI data object with real pretty_midi Note objects."""
    midi_data = Mock()
    instrument = Mock()
    instrument.is_drum = False

    mock_note_list = []
    for note_num in range(num_notes):
        note = pretty_midi.Note(
            velocity=70,  # Default velocity
            pitch=60 + note_num,  # Starting from middle C, incrementing
            start=float(note_num),
            end=float(note_num + 1.0)
        )
        mock_note_list.append(note)

    instrument.notes = mock_note_list
    midi_data.instruments = [instrument]

    return midi_data

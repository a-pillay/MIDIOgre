from unittest.mock import Mock


def generate_mock_midi_data(num_notes):
    # Create a mock MIDI data object with instruments and notes
    midi_data = Mock()
    instrument = Mock()
    instrument.is_drum = False  # Set is_drum attribute to False

    mock_note_list = []
    for note_num in range(num_notes):
        note = Mock()
        note.start = float(note_num)
        note.end = note_num + 1.0
        note.duration = 1.0
        note.pitch = 60 + note_num
        mock_note_list.append(note)

    instrument.notes = mock_note_list
    midi_data.instruments = [instrument]

    return midi_data

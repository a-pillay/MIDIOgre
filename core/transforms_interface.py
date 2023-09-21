class BaseMidiTransform:
    def __init__(self, p):
        if not 0 <= p <= 1:
            raise ValueError(
                "Probability of applying a MIDI Transform must be >=0 and <=1"
            )
        self.p = p

    def apply(self, midi_data):
        raise NotImplemented

    def __call__(self, midi_data):
        return self.apply(midi_data)

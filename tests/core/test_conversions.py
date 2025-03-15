import pytest
import numpy as np
import torch
import pretty_midi

from midiogre.core.conversions import BaseConversion, ToPRollNumpy, ToPRollTensor


def test_base_conversion_not_implemented():
    """Test that the BaseConversion class raises NotImplementedError."""
    base_conversion = BaseConversion()
    with pytest.raises(NotImplementedError):
        base_conversion.apply(None)


def test_toprollnumpy_initialization():
    """Test the initialization of ToPRollNumpy class."""
    converter = ToPRollNumpy(binarize=True, fs=200)
    assert converter.binarize is True
    assert converter.fs == 200


def test_toprollnumpy_apply():
    """Test the apply method of ToPRollNumpy class."""
    converter = ToPRollNumpy()
    mock_midi = pretty_midi.PrettyMIDI()
    piano_roll = converter.apply(mock_midi)

    assert isinstance(piano_roll, np.ndarray)


def test_toprolltensor_initialization():
    """Test the initialization of ToPRollTensor class."""
    converter = ToPRollTensor(device='cuda', binarize=True)
    assert converter.device == 'cuda'
    assert converter.binarize is True


def test_toprolltensor_apply():
    """Test the apply method of ToPRollTensor class."""
    converter = ToPRollTensor(device='cpu')
    mock_midi = pretty_midi.PrettyMIDI()
    piano_roll = converter.apply(mock_midi)

    assert isinstance(piano_roll, torch.Tensor)
    assert piano_roll.device.type == 'cpu'


if __name__ == '__main__':
    pytest.main()

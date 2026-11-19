import numpy as np
import pytest

from voice_agent.local_agent import _pcm16_level


def test_pcm16_level_reports_silence_and_full_scale_without_saving_audio():
    silence_dbfs, silence_peak = _pcm16_level(np.zeros(512, dtype=np.int16).tobytes())
    full_scale_dbfs, full_scale_peak = _pcm16_level(np.full(512, 32767, dtype=np.int16).tobytes())

    assert silence_dbfs < -100
    assert silence_peak == 0
    assert -0.01 < full_scale_dbfs <= 0
    assert full_scale_peak == 32767


def test_pcm16_level_rejects_partial_samples():
    with pytest.raises(ValueError, match="complete 16-bit samples"):
        _pcm16_level(b"\x00")

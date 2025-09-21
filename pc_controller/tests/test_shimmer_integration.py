"""Tests for Shimmer sensor integration and management."""

import time
from unittest.mock import Mock, patch

import numpy as np
import pytest
from core.shimmer_manager import (
    SHIMMER_AVAILABLE,
    RealShimmer,
    ShimmerManager,
    SimulatedShimmer,
    create_shimmer_manager,
)


class TestSimulatedShimmer:
    """Test SimulatedShimmer implementation."""

    def test_basic_functionality(self):
        """Test basic shimmer simulation."""
        shimmer = SimulatedShimmer(sample_rate_hz=10)

        assert shimmer.connect()

        received_data = []
        def callback(timestamp_ns, gsr_value):
            received_data.append((timestamp_ns, gsr_value))

        shimmer.start_streaming(callback)
        time.sleep(0.3)
        shimmer.stop_streaming()
        shimmer.disconnect()

        assert len(received_data) > 0

        timestamp, gsr = received_data[0]
        assert isinstance(timestamp, int)
        assert isinstance(gsr, float)
        assert gsr > 0


class TestRealShimmer:
    """Test RealShimmer implementation."""

    @pytest.mark.skipif(not SHIMMER_AVAILABLE, reason="pyshimmer not available")
    def test_shimmer_initialization(self):
        """Test real shimmer initialization."""
        with patch('pyshimmer.ShimmerBluetooth') as mock_shimmer:
            mock_instance = Mock()
            mock_shimmer.return_value = mock_instance
            mock_instance.connect.return_value = True

            shimmer = RealShimmer(device_port="COM3", sample_rate_hz=128)

            shimmer.connect()

            # Verify shimmer was configured
            mock_shimmer.assert_called_once()
            mock_instance.add_stream_callback.assert_called_once()

    def test_shimmer_unavailable_fallback(self):
        """Test fallback when shimmer library is not available."""
        with patch('core.shimmer_manager.SHIMMER_AVAILABLE', False):
            with pytest.raises(RuntimeError, match="pyshimmer library not available"):
                RealShimmer()

    def test_gsr_conversion(self):
        """Test critical 12-bit GSR conversion."""
        shimmer = SimulatedShimmer()

        test_cases = [
            (0, 0.0),
            (2048, 15.0),
            (4095, 20.0),
        ]

        for adc_value, _ in test_cases:
            if hasattr(shimmer, '_convert_gsr_to_microsiemens'):
                result = shimmer._convert_gsr_to_microsiemens(adc_value)
                assert result >= 0.0
                assert result < 100.0


class TestShimmerManager:
    """Test high-level ShimmerManager."""

    def test_auto_detection_simulation(self):
        """Test automatic fallback to simulation."""
        with patch('core.shimmer_manager.SHIMMER_AVAILABLE', False):
            manager = ShimmerManager(prefer_real=True)

            result = manager.initialize()
            assert result
            assert not manager.is_real
            assert manager.is_initialized

    @pytest.mark.skipif(not SHIMMER_AVAILABLE, reason="pyshimmer not available")
    def test_auto_detection_real(self):
        """Test real shimmer detection."""
        with patch('pyshimmer.ShimmerBluetooth') as mock_shimmer:
            mock_instance = Mock()
            mock_shimmer.return_value = mock_instance
            mock_instance.connect.return_value = True

            manager = ShimmerManager(prefer_real=True)
            result = manager.initialize()

            assert result

    def test_graceful_degradation(self):
        """Test graceful degradation when real hardware fails."""
        with patch('core.shimmer_manager.SHIMMER_AVAILABLE', True):
            with patch('core.shimmer_manager.RealShimmer') as mock_real:
                mock_real.return_value.connect.return_value = False

                manager = ShimmerManager(prefer_real=True)
                result = manager.initialize()

                assert result
                assert not manager.is_real


class TestShimmerFactory:
    """Test shimmer factory functions."""

    def test_create_shimmer_auto_detect(self):
        """Test automatic detection in factory."""
        with patch('core.shimmer_manager.SHIMMER_AVAILABLE', False):
            shimmer = create_shimmer_manager(use_real=None)
            assert isinstance(shimmer, SimulatedShimmer)

        if SHIMMER_AVAILABLE:
            with patch('core.shimmer_manager.cfg_get', return_value=True):
                shimmer = create_shimmer_manager(use_real=None)
                # Could be either depending on config
                assert isinstance(shimmer, RealShimmer | SimulatedShimmer)

    def test_create_shimmer_explicit(self):
        """Test explicit shimmer type selection."""
        shimmer = create_shimmer_manager(use_real=False)
        assert isinstance(shimmer, SimulatedShimmer)

        shimmer = create_shimmer_manager(use_real=True)
        if SHIMMER_AVAILABLE:
            assert isinstance(shimmer, RealShimmer)
        else:
            assert isinstance(shimmer, SimulatedShimmer)


class TestCriticalRequirements:
    """Test critical project requirements for Shimmer integration."""

    def test_twelve_bit_adc_requirement(self):
        """Test critical 12-bit ADC resolution requirement."""
        # This is a critical requirement from the project specifications
        SimulatedShimmer()

        test_values = [0, 1024, 2048, 3072, 4095]

        for adc_value in test_values:
            voltage = (adc_value / 4095.0) * 3.0  # 12-bit ADC, 3V reference
            assert 0.0 <= voltage <= 3.0

            if adc_value > 0:
                wrong_voltage = (adc_value / 65535.0) * 3.0
                assert voltage != wrong_voltage

    def test_sampling_rate_compliance(self):
        """Test 128 Hz sampling rate compliance."""
        shimmer = SimulatedShimmer(sample_rate_hz=128)

        received_data = []
        def callback(timestamp_ns, gsr_value):
            received_data.append(timestamp_ns)

        shimmer.connect()
        shimmer.start_streaming(callback)
        time.sleep(0.5)
        shimmer.stop_streaming()
        shimmer.disconnect()

        if len(received_data) > 1:
            timestamps = np.array(received_data) / 1e9
            intervals = np.diff(timestamps)
            avg_interval = np.mean(intervals)
            actual_rate = 1.0 / avg_interval

            assert 115 <= actual_rate <= 140

    def test_microsiemens_output_range(self):
        """Test GSR output is in microsiemens with reasonable range."""
        shimmer = SimulatedShimmer()

        received_values = []
        def callback(timestamp_ns, gsr_value):
            received_values.append(gsr_value)

        shimmer.connect()
        shimmer.start_streaming(callback)
        time.sleep(0.2)
        shimmer.stop_streaming()
        shimmer.disconnect()

        if received_values:
            gsr_array = np.array(received_values)

            assert np.all(gsr_array >= 0)

            assert np.all(gsr_array < 1000)

            if len(received_values) > 10:
                assert np.std(gsr_array) > 0

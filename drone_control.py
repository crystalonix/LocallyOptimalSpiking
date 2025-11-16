#
# Copyright (c) 2024 Anik Chattopadhyay, Arunava Banerjee
#
# Author: Anik Chattopadhyay
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-nd/4.0/
#
# Note: This project is also subject to a provisional patent. The Creative Commons license
# applies to the documentation and code provided herein, but does not grant any rights to
# the patented invention.
#

"""
Drone Control Module using Spike Train Compression

This module provides drone control capabilities using the spike train compression framework.
It compresses control signals (pitch, roll, yaw, throttle) into sparse spike trains,
transmits them efficiently, and reconstructs them for drone maneuvering.
"""

import numpy as np
import time
import logging
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

import configuration
import kernel_manager
import reconstruction_manager
import signal_utils
import spike_generator

logging.basicConfig(filename=configuration.log_file, level=configuration.logging_level)


@dataclass
class DroneControlCommand:
    """Represents a single drone control command"""
    pitch: float      # -1.0 to 1.0 (nose down to nose up)
    roll: float       # -1.0 to 1.0 (left to right)
    yaw: float        # -1.0 to 1.0 (counter-clockwise to clockwise)
    throttle: float   # 0.0 to 1.0 (min to max thrust)
    timestamp: float  # Time when command was generated


@dataclass
class CompressedControlSignal:
    """Represents compressed control signal as spike trains"""
    spike_times: List[List[int]]      # Spike times for each control channel [pitch, roll, yaw, throttle]
    spike_indexes: List[List[int]]    # Kernel indexes for each channel
    reconstruction_coeffs: List[np.ndarray]  # Reconstruction coefficients for each channel
    threshold_values: List[List[float]]  # Threshold crossing values
    signal_length: int                # Original signal length
    compression_ratio: float          # Compression ratio achieved
    error_rate: float                 # Reconstruction error rate


class DroneCommandCompressor:
    """
    Compresses drone control signals using spike train compression
    """
    
    def __init__(self, number_of_kernels: int = 50, 
                 spiking_threshold: float = 5e-6,
                 ahp_period: float = None,
                 ahp_high: float = None):
        """
        Initialize the compressor
        
        Args:
            number_of_kernels: Number of kernels to use for compression
            spiking_threshold: Threshold for spike generation
            ahp_period: After-hyperpolarization period (refractory period)
            ahp_high: AHP high value (elevated threshold after spike)
        """
        self.number_of_kernels = number_of_kernels
        self.spiking_threshold = spiking_threshold
        self.ahp_period = ahp_period if ahp_period is not None else configuration.ahp_period
        self.ahp_high = ahp_high if ahp_high is not None else configuration.ahp_high_value
        
        # Initialize kernels
        kernel_manager.init(number_of_kernels)
        
        # Select kernel indexes (exclude very low frequency kernels for efficiency)
        self.select_kernel_indexes = list(range(
            int(np.ceil(number_of_kernels / 10)), number_of_kernels
        ))
        
        logging.info(f"DroneCommandCompressor initialized with {number_of_kernels} kernels")
    
    def compress_control_signal(self, control_signal: np.ndarray, 
                                channel_name: str = "control") -> CompressedControlSignal:
        """
        Compress a single control signal into spike trains
        
        Args:
            control_signal: 1D numpy array of control values
            channel_name: Name of the control channel (for logging)
            
        Returns:
            CompressedControlSignal object containing spike train data
        """
        # Upsample the signal for better resolution
        upsampled_signal = signal_utils.up_sample(control_signal)
        
        # Initialize signal and compute kernel convolutions
        signal_norm_square, signal_kernel_convolutions = reconstruction_manager.init_signal(
            upsampled_signal, 
            configuration.mode,
            select_kernel_indexes=self.select_kernel_indexes
        )
        
        # Generate spikes
        spike_times, spike_indexes, threshold_values = spike_generator.calculate_spike_times(
            signal_kernel_convolutions,
            ahp_period=self.ahp_period,
            ahp_high=self.ahp_high,
            selected_kernel_indexes=self.select_kernel_indexes,
            threshold=self.spiking_threshold
        )
        
        # Calculate reconstruction coefficients
        reconstruction_coeffs = None
        error_rate = -1.0
        
        if len(spike_times) > 0:
            reconstruction_coeffs = reconstruction_manager.calculate_reconstruction(
                spike_times, spike_indexes, threshold_values
            )
            
            # Calculate compression ratio
            original_size = len(control_signal) * 4  # Assuming float32 (4 bytes)
            compressed_size = len(spike_times) * (4 + 4 + 8)  # time (int32), index (int32), coeff (float64)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
            
            # Calculate reconstruction error
            reconstructed = reconstruction_manager.get_reconstructed_signal(
                len(upsampled_signal), spike_times, spike_indexes, reconstruction_coeffs
            )
            error_rate = signal_utils.calculate_absolute_error_rate(upsampled_signal, reconstructed)
            
            logging.debug(f"{channel_name} channel: {len(spike_times)} spikes, "
                        f"compression ratio: {compression_ratio:.2f}, error rate: {error_rate:.6f}")
        else:
            compression_ratio = 1.0
            logging.warning(f"No spikes generated for {channel_name} channel")
        
        return CompressedControlSignal(
            spike_times=[spike_times],
            spike_indexes=[spike_indexes],
            reconstruction_coeffs=[reconstruction_coeffs] if reconstruction_coeffs is not None else [None],
            threshold_values=[threshold_values],
            signal_length=len(control_signal),
            compression_ratio=compression_ratio,
            error_rate=error_rate
        )
    
    def compress_multi_channel(self, control_signals: Dict[str, np.ndarray]) -> Dict[str, CompressedControlSignal]:
        """
        Compress multiple control channels simultaneously
        
        Args:
            control_signals: Dictionary mapping channel names to signal arrays
            
        Returns:
            Dictionary mapping channel names to CompressedControlSignal objects
        """
        compressed_signals = {}
        
        for channel_name, signal in control_signals.items():
            compressed_signals[channel_name] = self.compress_control_signal(signal, channel_name)
        
        return compressed_signals
    
    def reconstruct_control_signal(self, compressed_signal: CompressedControlSignal) -> np.ndarray:
        """
        Reconstruct a control signal from compressed spike train data
        
        Args:
            compressed_signal: CompressedControlSignal object
            
        Returns:
            Reconstructed control signal as numpy array
        """
        if compressed_signal.reconstruction_coeffs[0] is None:
            logging.warning("No reconstruction coefficients available, returning zeros")
            return np.zeros(compressed_signal.signal_length)
        
        # Reconstruct at upsampled resolution
        upsampled_length = compressed_signal.signal_length * configuration.upsample_factor
        reconstructed_upsampled = reconstruction_manager.get_reconstructed_signal(
            upsampled_length,
            compressed_signal.spike_times[0],
            compressed_signal.spike_indexes[0],
            compressed_signal.reconstruction_coeffs[0]
        )
        
        # Downsample to original resolution
        reconstructed = signal_utils.down_sample(reconstructed_upsampled)
        
        return reconstructed


class DroneController:
    """
    Main drone controller that uses spike train compression for control signals
    """
    
    def __init__(self, number_of_kernels: int = 50,
                 spiking_threshold: float = 5e-6,
                 control_rate: float = 50.0,  # Hz
                 use_compression: bool = True):
        """
        Initialize the drone controller
        
        Args:
            number_of_kernels: Number of kernels for compression
            spiking_threshold: Spike generation threshold
            control_rate: Control loop frequency in Hz
            use_compression: Whether to use spike train compression
        """
        self.compressor = DroneCommandCompressor(
            number_of_kernels=number_of_kernels,
            spiking_threshold=spiking_threshold
        ) if use_compression else None
        
        self.control_rate = control_rate
        self.use_compression = use_compression
        self.command_history: List[DroneControlCommand] = []
        
        # Control signal buffers for batch compression
        self.signal_buffers = {
            'pitch': [],
            'roll': [],
            'yaw': [],
            'throttle': []
        }
        
        logging.info(f"DroneController initialized (compression: {use_compression}, rate: {control_rate} Hz)")
    
    def add_control_command(self, command: DroneControlCommand):
        """
        Add a control command to the buffer
        
        Args:
            command: DroneControlCommand object
        """
        self.command_history.append(command)
        self.signal_buffers['pitch'].append(command.pitch)
        self.signal_buffers['roll'].append(command.roll)
        self.signal_buffers['yaw'].append(command.yaw)
        self.signal_buffers['throttle'].append(command.throttle)
    
    def compress_control_sequence(self, sequence_length: int = None) -> Dict[str, CompressedControlSignal]:
        """
        Compress a sequence of control commands
        
        Args:
            sequence_length: Number of commands to compress (None = all in buffer)
            
        Returns:
            Dictionary of compressed control signals
        """
        if not self.use_compression or self.compressor is None:
            raise ValueError("Compression is not enabled")
        
        if sequence_length is None:
            sequence_length = len(self.command_history)
        
        # Extract signals from buffers
        control_signals = {}
        for channel in ['pitch', 'roll', 'yaw', 'throttle']:
            signal = np.array(self.signal_buffers[channel][-sequence_length:])
            control_signals[channel] = signal
        
        # Compress all channels
        compressed = self.compressor.compress_multi_channel(control_signals)
        
        return compressed
    
    def reconstruct_and_apply(self, compressed_signals: Dict[str, CompressedControlSignal]) -> List[DroneControlCommand]:
        """
        Reconstruct control signals and convert to command sequence
        
        Args:
            compressed_signals: Dictionary of compressed signals
            
        Returns:
            List of reconstructed DroneControlCommand objects
        """
        reconstructed_commands = []
        
        for channel_name, compressed in compressed_signals.items():
            reconstructed_signal = self.compressor.reconstruct_control_signal(compressed)
            
            # Convert to commands (assuming same timestamps as original)
            for i, value in enumerate(reconstructed_signal):
                if i >= len(reconstructed_commands):
                    reconstructed_commands.append(DroneControlCommand(
                        pitch=0.0, roll=0.0, yaw=0.0, throttle=0.0, timestamp=0.0
                    ))
                
                if channel_name == 'pitch':
                    reconstructed_commands[i].pitch = value
                elif channel_name == 'roll':
                    reconstructed_commands[i].roll = value
                elif channel_name == 'yaw':
                    reconstructed_commands[i].yaw = value
                elif channel_name == 'throttle':
                    reconstructed_commands[i].throttle = value
        
        return reconstructed_commands
    
    def get_compression_stats(self, compressed_signals: Dict[str, CompressedControlSignal]) -> Dict:
        """
        Get compression statistics
        
        Args:
            compressed_signals: Dictionary of compressed signals
            
        Returns:
            Dictionary with compression statistics
        """
        total_spikes = sum(len(cs.spike_times[0]) for cs in compressed_signals.values())
        avg_compression = np.mean([cs.compression_ratio for cs in compressed_signals.values()])
        avg_error = np.mean([cs.error_rate for cs in compressed_signals.values()])
        
        return {
            'total_spikes': total_spikes,
            'average_compression_ratio': avg_compression,
            'average_error_rate': avg_error,
            'channels': {name: {
                'spikes': len(cs.spike_times[0]),
                'compression_ratio': cs.compression_ratio,
                'error_rate': cs.error_rate
            } for name, cs in compressed_signals.items()}
        }
    
    def clear_buffers(self):
        """Clear command history and signal buffers"""
        self.command_history.clear()
        for channel in self.signal_buffers:
            self.signal_buffers[channel].clear()


def create_test_control_sequence(duration: float = 5.0, 
                                  sample_rate: float = 50.0) -> List[DroneControlCommand]:
    """
    Create a test sequence of control commands for demonstration
    
    Args:
        duration: Duration of sequence in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        List of DroneControlCommand objects
    """
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples)
    
    commands = []
    for i, time_val in enumerate(t):
        # Create sinusoidal control signals for testing
        pitch = 0.3 * np.sin(2 * np.pi * 0.5 * time_val)
        roll = 0.2 * np.cos(2 * np.pi * 0.3 * time_val)
        yaw = 0.1 * np.sin(2 * np.pi * 0.4 * time_val)
        throttle = 0.5 + 0.2 * np.sin(2 * np.pi * 0.2 * time_val)
        
        # Clamp values to valid ranges
        pitch = np.clip(pitch, -1.0, 1.0)
        roll = np.clip(roll, -1.0, 1.0)
        yaw = np.clip(yaw, -1.0, 1.0)
        throttle = np.clip(throttle, 0.0, 1.0)
        
        commands.append(DroneControlCommand(
            pitch=pitch,
            roll=roll,
            yaw=yaw,
            throttle=throttle,
            timestamp=time_val
        ))
    
    return commands


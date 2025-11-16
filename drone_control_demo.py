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
Demo script showing how to use spike train compression for drone control

This script demonstrates:
1. Creating control command sequences
2. Compressing control signals using spike trains
3. Reconstructing signals from compressed data
4. Analyzing compression performance
"""

import numpy as np
import matplotlib.pyplot as plt
from drone_control import (
    DroneController, 
    DroneControlCommand, 
    create_test_control_sequence
)


def plot_control_signals(original_commands, reconstructed_commands, compressed_signals):
    """Plot original vs reconstructed control signals"""
    num_channels = 4
    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 10))
    
    channels = ['pitch', 'roll', 'yaw', 'throttle']
    colors = ['blue', 'green', 'red', 'orange']
    
    for i, (channel, color) in enumerate(zip(channels, colors)):
        ax = axes[i]
        
        # Original signal
        original_values = [getattr(cmd, channel) for cmd in original_commands]
        times = [cmd.timestamp for cmd in original_commands]
        ax.plot(times, original_values, color=color, label=f'Original {channel}', linewidth=2, alpha=0.7)
        
        # Reconstructed signal
        if reconstructed_commands:
            recon_values = [getattr(cmd, channel) for cmd in reconstructed_commands]
            ax.plot(times[:len(recon_values)], recon_values, 
                   color=color, linestyle='--', label=f'Reconstructed {channel}', linewidth=2)
        
        ax.set_ylabel(f'{channel.capitalize()}', fontsize=12)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{channel.capitalize()} Control Signal', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('drone_control_signals.png', dpi=150)
    print("Control signals plot saved to 'drone_control_signals.png'")
    plt.close()


def plot_spike_trains(compressed_signals):
    """Plot spike trains for visualization"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    
    channels = ['pitch', 'roll', 'yaw', 'throttle']
    colors = ['blue', 'green', 'red', 'orange']
    
    for i, (channel, color) in enumerate(zip(channels, colors)):
        ax = axes[i]
        compressed = compressed_signals[channel]
        
        spike_times = compressed.spike_times[0]
        spike_indexes = compressed.spike_indexes[0]
        
        # Plot spikes
        if len(spike_times) > 0:
            ax.scatter(spike_times, spike_indexes, s=50, c=color, marker='|', alpha=0.7)
            ax.set_ylabel('Kernel Index', fontsize=12)
            ax.set_xlabel('Time (samples)', fontsize=12)
            ax.set_title(f'{channel.capitalize()} Spike Train ({len(spike_times)} spikes)', fontsize=14)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No spikes for {channel}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{channel.capitalize()} Spike Train (No spikes)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('drone_spike_trains.png', dpi=150)
    print("Spike trains plot saved to 'drone_spike_trains.png'")
    plt.close()


def main():
    """Main demo function"""
    print("=" * 60)
    print("Drone Control with Spike Train Compression Demo")
    print("=" * 60)
    
    # Create a test control sequence
    print("\n1. Creating test control sequence...")
    duration = 5.0  # seconds
    sample_rate = 50.0  # Hz
    original_commands = create_test_control_sequence(duration, sample_rate)
    print(f"   Created {len(original_commands)} control commands")
    
    # Initialize drone controller
    print("\n2. Initializing drone controller...")
    controller = DroneController(
        number_of_kernels=50,
        spiking_threshold=5e-6,
        control_rate=sample_rate,
        use_compression=True
    )
    
    # Add commands to controller
    print("\n3. Adding commands to controller...")
    for cmd in original_commands:
        controller.add_control_command(cmd)
    
    # Compress control signals
    print("\n4. Compressing control signals...")
    compressed_signals = controller.compress_control_sequence()
    
    # Get compression statistics
    stats = controller.get_compression_stats(compressed_signals)
    print("\n5. Compression Statistics:")
    print(f"   Total spikes: {stats['total_spikes']}")
    print(f"   Average compression ratio: {stats['average_compression_ratio']:.2f}x")
    print(f"   Average error rate: {stats['average_error_rate']:.6f}")
    print("\n   Per-channel statistics:")
    for channel, channel_stats in stats['channels'].items():
        print(f"     {channel}:")
        print(f"       Spikes: {channel_stats['spikes']}")
        print(f"       Compression: {channel_stats['compression_ratio']:.2f}x")
        print(f"       Error rate: {channel_stats['error_rate']:.6f}")
    
    # Reconstruct signals
    print("\n6. Reconstructing control signals...")
    reconstructed_commands = controller.reconstruct_and_apply(compressed_signals)
    print(f"   Reconstructed {len(reconstructed_commands)} commands")
    
    # Calculate reconstruction errors
    print("\n7. Reconstruction Errors:")
    for channel in ['pitch', 'roll', 'yaw', 'throttle']:
        original = np.array([getattr(cmd, channel) for cmd in original_commands])
        reconstructed = np.array([getattr(cmd, channel) for cmd in reconstructed_commands[:len(original)]])
        mse = np.mean((original - reconstructed) ** 2)
        mae = np.mean(np.abs(original - reconstructed))
        print(f"   {channel}: MSE={mse:.6f}, MAE={mae:.6f}")
    
    # Plot results
    print("\n8. Generating plots...")
    plot_control_signals(original_commands, reconstructed_commands, compressed_signals)
    plot_spike_trains(compressed_signals)
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()


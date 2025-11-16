# Drone Control with Spike Train Compression

This module provides drone control capabilities using the spike train compression framework. It compresses control signals (pitch, roll, yaw, throttle) into sparse spike trains for efficient transmission and reconstruction.

## Features

- **Signal Compression**: Compress drone control signals into sparse spike trains
- **Efficient Transmission**: Significantly reduce data size for wireless communication
- **Signal Reconstruction**: Reconstruct control signals from compressed spike trains
- **Multi-Channel Support**: Handle multiple control channels simultaneously
- **Performance Metrics**: Track compression ratios and reconstruction errors

## Architecture

### Components

1. **DroneCommandCompressor**: Compresses control signals into spike trains
2. **DroneController**: Main controller that manages compression and reconstruction
3. **DroneControlCommand**: Data structure for individual control commands

### Workflow

```
Control Commands → Signal Buffers → Spike Train Compression → 
Compressed Data → Transmission → Reconstruction → Control Signals
```

## Usage

### Basic Example

```python
from drone_control import DroneController, DroneControlCommand, create_test_control_sequence

# Initialize controller
controller = DroneController(
    number_of_kernels=50,
    spiking_threshold=5e-6,
    control_rate=50.0,  # Hz
    use_compression=True
)

# Create control commands
commands = create_test_control_sequence(duration=5.0, sample_rate=50.0)

# Add commands to controller
for cmd in commands:
    controller.add_control_command(cmd)

# Compress signals
compressed_signals = controller.compress_control_sequence()

# Get statistics
stats = controller.get_compression_stats(compressed_signals)
print(f"Compression ratio: {stats['average_compression_ratio']:.2f}x")
print(f"Error rate: {stats['average_error_rate']:.6f}")

# Reconstruct signals
reconstructed_commands = controller.reconstruct_and_apply(compressed_signals)
```

### Manual Control Command Creation

```python
from drone_control import DroneControlCommand

# Create individual control command
command = DroneControlCommand(
    pitch=0.3,      # Nose up (0.0 to 1.0)
    roll=-0.2,      # Roll left (-1.0 to 1.0)
    yaw=0.1,        # Yaw right (-1.0 to 1.0)
    throttle=0.6,  # 60% throttle (0.0 to 1.0)
    timestamp=time.time()
)

controller.add_control_command(command)
```

### Compression Parameters

You can customize compression parameters:

```python
from drone_control import DroneCommandCompressor

compressor = DroneCommandCompressor(
    number_of_kernels=100,      # More kernels = better quality, slower
    spiking_threshold=5e-7,      # Lower = more spikes, better quality
    ahp_period=10000.0,          # Refractory period
    ahp_high=5e-5                # AHP threshold elevation
)
```

## Control Signal Format

### Pitch
- Range: -1.0 to 1.0
- -1.0: Maximum nose down
- 0.0: Level
- 1.0: Maximum nose up

### Roll
- Range: -1.0 to 1.0
- -1.0: Maximum roll left
- 0.0: Level
- 1.0: Maximum roll right

### Yaw
- Range: -1.0 to 1.0
- -1.0: Maximum counter-clockwise rotation
- 0.0: No rotation
- 1.0: Maximum clockwise rotation

### Throttle
- Range: 0.0 to 1.0
- 0.0: Minimum thrust
- 1.0: Maximum thrust

## Performance Considerations

### Compression Ratio
- Typical compression ratios: 5x to 50x depending on signal complexity
- Sparse signals compress better than dense signals
- Higher spiking thresholds reduce spikes but may increase error

### Reconstruction Error
- Error rates typically < 0.01 (1%)
- Can be tuned by adjusting:
  - Number of kernels
  - Spiking threshold
  - AHP parameters

### Computational Cost
- Compression: O(n × k) where n = signal length, k = kernels
- Reconstruction: O(s × k) where s = number of spikes
- Real-time capable for control rates up to 100 Hz

## Integration with Drone Hardware

To integrate with actual drone hardware, you would:

1. **Receive compressed data** over wireless link
2. **Reconstruct signals** using `reconstruct_and_apply()`
3. **Send commands** to flight controller via appropriate API (e.g., MAVLink, DJI SDK)

Example integration stub:

```python
def send_to_drone(commands):
    """Send reconstructed commands to drone hardware"""
    for cmd in commands:
        # Example: Send via MAVLink or drone SDK
        # drone.set_pitch(cmd.pitch)
        # drone.set_roll(cmd.roll)
        # drone.set_yaw(cmd.yaw)
        # drone.set_throttle(cmd.throttle)
        pass

# In your control loop
compressed = controller.compress_control_sequence()
# ... transmit compressed data ...
reconstructed = controller.reconstruct_and_apply(compressed)
send_to_drone(reconstructed)
```

## Demo Script

Run the demo to see the system in action:

```bash
python drone_control_demo.py
```

This will:
1. Generate a test control sequence
2. Compress the signals
3. Reconstruct them
4. Display compression statistics
5. Generate visualization plots

## Configuration

The module uses the global `configuration` module for:
- Upsampling factor
- Kernel parameters
- Signal processing modes

You can modify these in `configuration.py` to optimize for your use case.

## Limitations

1. **Latency**: Compression/reconstruction adds computational latency
2. **Error**: Reconstruction is lossy (though typically < 1% error)
3. **Memory**: Requires buffering control sequences for batch compression
4. **Real-time**: Best for offline compression or low-rate real-time (< 100 Hz)

## Future Enhancements

- Real-time streaming compression
- Adaptive threshold adjustment
- Multi-drone coordination
- Integration with popular drone SDKs (MAVLink, DJI, etc.)
- Hardware acceleration support

## License

This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License.

Note: This project is also subject to a provisional patent. The Creative Commons license applies to the documentation and code provided herein, but does not grant any rights to the patented invention.


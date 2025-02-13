# mfcc_rust_binding/test/mfcc_binding_extended_test.py

import numpy as np
from mfcc_rust import PyTransform

def test_sequential_frames():
    """
    This test simulates processing a sequence of audio frames.
    It calls the transform function multiple times to verify that:
    
    - The output length is as expected (maxfilter * 3).
    - The transform correctly computes derivative coefficients (non-zero
      differences on subsequent calls) after the first frame.
    """
    sample_rate = 16000
    buffer_size = 512
    transform = PyTransform(sample_rate, buffer_size)
    
    # Configure the transform with specific parameters.
    transform.nfilters(16, 40)  # maxfilter=16, nfilters=40
    transform.normlength(5)
    
    outputs = []
    num_frames = 10
    for i in range(num_frames):
        # Create a "frame" of random audio data.
        frame = np.random.randint(-32768, 32767, buffer_size, dtype=np.int16)
        features = np.array(transform.transform(frame.tolist()))
        outputs.append(features)
        print(f"Frame {i+1} MFCC Features:")
        print(features)
        
        # Ensure output has length of (maxfilter * 3).
        expected_length = 16 * 3
        assert features.shape[0] == expected_length, f"Frame {i+1} output length {features.shape[0]} does not match expected {expected_length}"
    
    # For the second frame onward, the transform should compute differences (derivative coefficients).
    # We verify that the derivative section (elements 16 to 32) is not all zero in the second frame.
    if np.allclose(outputs[1][16:32], 0):
        raise AssertionError("Frame 2 derivative coefficients are unexpectedly all zero")
    
    print("Sequential frame processing test passed.\n")


def test_config_changes():
    """
    This test verifies that calling configuration functions changes the behavior of the transform.
    After reconfiguring (changing maxfilter and normalization parameters), the size of the output
    vector should change accordingly.
    """
    sample_rate = 16000
    buffer_size = 512
    transform = PyTransform(sample_rate, buffer_size)
    
    # Change configuration to new parameters. After calling nfilters(20, 50),
    # the expected output length becomes maxfilter * 3 = 20 * 3 = 60.
    transform.nfilters(20, 50)
    transform.normlength(7)
    
    # Process one frame.
    frame = np.random.randint(-32768, 32767, buffer_size, dtype=np.int16)
    features = np.array(transform.transform(frame.tolist()))
    
    expected_length = 20 * 3
    assert features.shape[0] == expected_length, f"Output length {features.shape[0]} does not match expected {expected_length}"
    
    print("Configuration changes test passed.\n")


def main():
    print("Starting extended MFCC binding tests.\n")
    test_sequential_frames()
    test_config_changes()
    print("All extended tests passed successfully.")


if __name__ == "__main__":
    main()
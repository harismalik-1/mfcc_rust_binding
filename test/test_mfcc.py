# test_mfcc.py
import mfcc_rust

# Create a new instance of PyTransform
# Use appropriate sample_rate and buffer_size values as required by your module.
transform = mfcc_rust.PyTransform(16000, 1024)

# Create a dummy input of 1024 samples (for example, all zeros)
input_samples = [0] * 1024

# Call the transform method
output = transform.transform(input_samples)

print("MFCC Output:", output)
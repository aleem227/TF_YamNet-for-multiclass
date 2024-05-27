import sounddevice as sd
import numpy as np
import tensorflow as tf
import librosa
from collections import deque
import time

def compute_mfcc(y, sr):
    # Extract MFCC features for the entire audio
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return mfccs.T  # Transpose the MFCC matrix

# Define parameters for audio recording
RATE = 16000
SEGMENT_DURATION = 1  # Duration of each audio segment in seconds
SEGMENT_SAMPLES = RATE * SEGMENT_DURATION  # Number of samples in each segment

# Initialize a deque as a ring buffer to store audio data
audio_buffer = deque(maxlen=SEGMENT_SAMPLES)

# Initialize the TFLite model
interpreter = tf.lite.Interpreter(model_path="custom_CNN_model_3classes.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def inference_from_buffer(buffer, rate):
    # Convert the buffer to a numpy array
    audio_data = np.array(buffer)
    # Compute MFCC
    mfcc = compute_mfcc(audio_data, rate)
    # Reshape the MFCC for the model
    mfcc_input = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1])
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], mfcc_input)
    # Run inference
    interpreter.invoke()
    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # Get the predicted class ID
    return np.argmax(output_data)

class_mapping = {
    0: 'Clicking',
    1: 'Silence',
    2: 'Speech'
}

# Callback function to process each block of audio
def audio_callback(indata, frames, time, status):
    # Flatten and append new audio data to the buffer
    audio_buffer.extend(indata[:, 0])
    # Perform inference if we have enough data for one segment
    if len(audio_buffer) >= SEGMENT_SAMPLES:
        predicted_class_id = inference_from_buffer(audio_buffer, RATE)
        print("Predicted class:", class_mapping[predicted_class_id])

# Start recording and processing
print("Recording and processing in real-time. Press Ctrl+C to stop...")
with sd.InputStream(samplerate=RATE, channels=1, dtype=np.float32, callback=audio_callback):
    try:
        while True:
            # Just keep the script running
            time.sleep(1)
    except KeyboardInterrupt:
        print("Recording stopped.")
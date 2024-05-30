import json
import numpy as np
import tensorflow as tf

# Load the JSON data
with open('examples/sample_openpose.json') as f:
    data = json.load(f)

# Function to create a feature for the tf.train.Example
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Create a TFRecord file
with tf.io.TFRecordWriter('sample.tfrecord') as writer:
    for item in data:
        frames_data = item['frames']
        fps = 50  # Assuming 50 FPS, modify if different
        
        for frame_id, frame_content in frames_data.items():
            people = frame_content['people']
            
            for person in people:
                # Extract the pose keypoints
                pose_keypoints_2d = person['pose_keypoints_2d']
                
                # Convert to numpy array and separate coordinates and confidence
                keypoints_array = np.array(pose_keypoints_2d, dtype=np.float32).reshape(-1, 3)
                coordinates = keypoints_array[:, :2]  # x, y coordinates
                confidence = keypoints_array[:, 2]  # confidence scores

                # Serialize the coordinates and confidence
                pose_data = tf.io.serialize_tensor(coordinates).numpy()
                pose_confidence = tf.io.serialize_tensor(confidence).numpy()
                
                # Create a feature dictionary
                features = {
                    'fps': _int64_feature(fps),
                    'pose_data': _bytes_feature(pose_data),
                    'pose_confidence': _bytes_feature(pose_confidence),
                }
                
                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=features))
                
                # Serialize to string and write to the TFRecord file
                writer.write(example.SerializeToString())

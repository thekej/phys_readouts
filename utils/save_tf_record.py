import tensorflow as tf
import h5py

# Open the HDF5 file
h5_file = h5py.File("path/to/file.hdf5", "r")

# Load the videos and actions from the file
videos = h5_file["videos"][:]
actions = h5_file["actions"][:]

# Create a TFRecord writer
writer = tf.io.TFRecordWriter("path/to/file.tfrecord")

# Iterate over the samples in the dataset
for i in range(videos.shape[0]):
    # Create a feature for each video and action
    video_feature = tf.train.Feature(float_list=tf.train.FloatList(value=videos[i].flatten()))
    action_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[actions[i]]))
    # Create a Features object for the record
    features = tf.train.Features(feature={
        "video": video_feature,
        "action": action_feature
    })
    # Create an Example object for the record
    example = tf.train.Example(features=features)
    # Serialize the Example object and write it to the TFRecord file
    writer.write(example.SerializeToString())

# Close the TFRecord writer
writer.close()

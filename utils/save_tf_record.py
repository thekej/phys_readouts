import tensorflow as tf
import h5py
import argparse
import tqdm
import json
import os
import numpy as np

def main(args):
    # Open the HDF5 file
    h5_file = h5py.File(args.data, "r")

    # Load the videos and actions from the file
    videos = h5_file["video"][:]
    actions = h5_file["action"][:]

    # Define the features 
    # TODO fix this add more stuff
    features = {
        "type": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
        "content":{
            "features": {
                "actions": {
                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                    "tensor": {
                        "shape": {
                            "dimensions": ["-1"]
                        },
                        "dtype": "int32",
                        "encoding": "none"
                    }
                },
                "video": {
                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                    "tensor": {
                        "shape": {
                            "dimensions": ["-1", "16", "16"]
                        },
                        "dtype": "uint16",
                        "encoding": "none"
                    }
                }
            }
        },
        "proto_cls": "tensorflow_datasets.FeaturesDict"
    }

    # Save the features to a JSON file
    with open(os.path.join(args.save_path, "features.json"), "w") as f:
        json.dump(features, f)

    # Create a TFRecord writer
    writer = tf.io.TFRecordWriter(os.path.join(args.save_path, "teco_all-test.tfrecord-00000-of-00001"))

    # Iterate over the samples in the dataset
    for i in tqdm.tqdm(range(videos.shape[0])):
        # Create a feature for each video and action
        video_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=videos[i].flatten().astype(np.uint16)))
        action_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=actions[i]))
        # Create a Features object for the record
        features = tf.train.Features(feature={
            "video": video_feature,
            "actions": action_feature
        })
        # Create an Example object for the record
        example = tf.train.Example(features=features)
        # Serialize the Example object and write it to the TFRecord file
        writer.write(example.SerializeToString())

    # Close the TFRecord writer
    writer.close()
        
    # save datset info
    dataset_info = {"configName": "physion", "fileFormat": "tfrecord", 
        "moduleName": "hv.datasets.encoded_h5py_dataset.encoded_h5py_dataset",
        "name": "teco_all",
        "releaseNotes": {"1.0.0": "Initial release."},
        "splits": [{"name": "test",
                    "numBytes": str(os.path.getsize(os.path.join(args.save_path, "teco_all-test.tfrecord-00000-of-00001"))),
                    "shardLengths": [str(videos.shape[0])]
                   }
                  ],
        "version": "1.0.0"
        }
    with open(os.path.join(args.save_path, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True)
    parser.add_argument('-s', '--save_path', type=str, required=True)
    args = parser.parse_args()

    main(args)

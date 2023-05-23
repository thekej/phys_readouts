import argparse
import h5py
import torch
import tqdm

from timesformer.models.vit import TimeSformer
from timesformer_loader import TimesformerDataset

from torch.utils.data import Dataset, DataLoader


def main(args):
    # set up new dataset
    f = h5py.File(args.save_file, "w")
    dataset = TimesformerDataset(args.data_path)

    dset1 = f.create_dataset("label", (len(dataset),), dtype='f')
    dset2 = f.create_dataset("features", (len(dataset), 4, 768), dtype='f')
        
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = TimeSformer(img_size=448, num_classes=174, num_frames=16, 
                        attention_type='divided_space_time',  pretrained_model=str(args.model))

    #dummy_video = torch.randn(2, 3, 8, 224, 224) # (batch x channels x frames x height x width)
    
    for i, batch in enumerate(tqdm.tqdm(data_loader)):
        v_in, label_in = batch
        features = []
        for i in range(4):
            print(v_in[:, i].shape)
            pred = model.forward_features(v_in[:, i],)
            features.append(pred[0])
        features = torch.stack(features)

        # reshape
        # run model with 8 frames 4 times
        # first run observed, get first 6 frames 500ms and repeat first two frames to get 8 frames
        # from 6 to 30, divide into 3 runs each with 8 frames
        # aggregate
        # add to data
        dset1[i] = label_in.reshape(-1)
        dset2[i] = features.numpy()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str,
                        help='Path to save data on')
    parser.add_argument('--save-file', type=str,
                        help='Path to save data on')
    parser.add_argument('--batch-size', type=int,
                        help='Path to save data on')
    parser.add_argument('--model', type=str,
                        default='encoded_data.',
                        help='Path to save data on')

    args = parser.parse_args()
    main(args)
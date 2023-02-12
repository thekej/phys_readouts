import json
import argparse


def map_(args):
    with open(args.stimuli_path) as f:
        stimuli = json.load(f)

    scenarios = {'coll' : [], 'domino': [], 'link': [], 
                 'towers': [], 'contain': [], 'drop': [], 
                 'roll': []}

    for stimulus in stimuli.keys():
        for scenario in scenarios.keys():
            if scenario in stimulus:
                scenarios[scenario] += [stimuli[stimulus]]
                break

    with open(args.save_path, 'w') as f:
        json.dump(scenarios, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stimuli-path', type=str, default=None,
                        help='Path for video file')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Path for output-directory')
    args = parser.parse_args()
    map_(args)
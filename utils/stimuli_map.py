import json
import argparse


def map_(args):
    with open(args.stimuli_path) as f:
        stimuli = json.load(f)

    scenarios = {'collision' : [], 'domino': [], 'link': [], 
                 'towers': [], 'contain': [], 'drop': [], 
                 'roll': []}
    e = 0
    for stimulus in stimuli.keys():
        # this needs to be hardcoded due pilot_it2_rollingSliding_simple_collision_box_large_force_0025 this type of naming
        if 'coll' in stimulus and not 'roll' in stimulus:
            scenarios['collision'] += [stimuli[stimulus]]
        elif 'coll' in stimulus and 'roll' in stimulus:
            scenarios['roll'] += [stimuli[stimulus]]
        elif 'roll' in stimulus:
            scenarios['roll'] += [stimuli[stimulus]]
        elif 'domino' in stimulus:
            scenarios['domino'] += [stimuli[stimulus]]
        elif 'link' in stimulus:
            scenarios['link'] += [stimuli[stimulus]]
        elif 'towers' in stimulus:
            scenarios['towers'] += [stimuli[stimulus]]
        elif 'contain' in stimulus:
            scenarios['contain'] += [stimuli[stimulus]]
        elif 'drop' in stimulus:
            scenarios['drop'] += [stimuli[stimulus]]
                
    for s in scenarios.keys():
        print('%s has %d'%(s, len(scenarios[s])))

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
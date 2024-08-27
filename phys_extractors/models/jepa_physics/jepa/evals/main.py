# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

import multiprocessing as mp

import pprint
import yaml

from jepa.src.utils.distributed import init_distributed

from jepa.evals.scaffold import main as eval_main

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')

parser.add_argument(
    '--port', type=int, default=37132,
    help='which port to use')

parser.add_argument(
    '--model_name', type=str,
    default=None,
    help='model name to use',
    )


def process_main(rank, fname, world_size, devices, port, model_name=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')

    # Load config
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    # Init distributed (access to comm between GPUS on same machine)
    world_size, rank = init_distributed(port=port, rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')

    # Launch the eval with loaded config
    eval_main(params['eval_name'], args_eval=params, model_name=model_name)


def main():
    args = parser.parse_args()
    print(args)
    num_gpus = len(args.devices)
    mp.set_start_method('spawn')
    for rank in range(num_gpus):
        mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices, args.port, args.model_name)
        ).start()


if __name__ == '__main__':
    main()
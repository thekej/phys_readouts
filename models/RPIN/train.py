import os
import torch
import random
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from physion_loader import RPINTrainDataset
from neuralphys.utils.config import _C as cfg
from neuralphys.utils.logger import setup_logger, git_diff_config
from neuralphys.models import *
from neuralphys.models.rpin import Net
from neuralphys.trainer import Trainer


def arg_parse():
    # only the most general argument is passed here
    # task-specific parameters should be passed by config file
    parser = argparse.ArgumentParser(description='RPIN parameters')
    parser.add_argument('--output-dir', required=True, help='path to config file', type=str)
    parser.add_argument('--data-path', required=True, help='path for data', type=str)
    parser.add_argument('--data-size', type=int, default=0)
    parser.add_argument('--init', type=str, default='')
    parser.add_argument('--gpus', type=str)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def main():
    # the wrapper file contains:
    # 1. setup training environment
    # 2. setup config
    # 3. setup logger
    # 4. setup model
    # 5. setup optimizer
    # 6. setup dataset

    # ---- setup training environment
    args = arg_parse()
    rng_seed = args.seed
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        num_gpus = torch.cuda.device_count()
    else:
        assert NotImplementedError

    # ---- setup config files
    #cfg.merge_from_file(args.cfg)
    cfg.SOLVER.BATCH_SIZE *= num_gpus
    cfg.SOLVER.BASE_LR *= num_gpus
    cfg.freeze()
    os.makedirs(args.output_dir, exist_ok=True)
    #shutil.copy(args.cfg, os.path.join(args.output_dir, 'config.yaml'))

    # ---- setup logger
    logger = setup_logger('RPIN', args.output_dir)

    model = Net()
    model.to(torch.device('cuda'))
    model = torch.nn.DataParallel(
        model, device_ids=list(range(args.gpus.count(',') + 1))
    )
    # ---- setup optimizer, optimizer is not changed
    vae_params = [p for p_name, p in model.named_parameters() if 'vae_lstm' in p_name]
    other_params = [p for p_name, p in model.named_parameters() if 'vae_lstm' not in p_name]
    optim = torch.optim.Adam(
        [{'params': vae_params, 'weight_decay': 0.0}, {'params': other_params}],
        lr=cfg.SOLVER.BASE_LR,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    # ---- if resume experiments, use --init ${model_name}
    if args.init:
        logger.info(f'loading pretrained model from {args.init}')
        cp = torch.load(args.init)
        model.load_state_dict(cp['model'], False)

    # ---- setup dataset in the last, and avoid non-deterministic in data shuffling order
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    
    indices = list(range(args.data_size))
    train_indices = random.sample(indices, int(0.9 * len(indices)))
    print(len(train_indices))
    val_indices = list(set(indices) - set(train_indices))
    train_set = RPINTrainDataset(args.data_path, train_indices)
    val_set = RPINTrainDataset(args.data_path, val_indices)
    kwargs = {'pin_memory': False, 'num_workers': 96} #16
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, **kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=1 if cfg.RPIN.VAE else cfg.SOLVER.BATCH_SIZE, shuffle=False, **kwargs,
    )
    print(f'size: train {len(train_indices)} / test {len(val_indices)}')

    # ---- setup trainer
    kwargs = {'device': torch.device('cuda'),
              'model': model,
              'optim': optim,
              'train_loader': train_loader,
              'val_loader': val_loader,
              'output_dir': args.output_dir,
              'logger': logger,
              'num_gpus': num_gpus,
              'max_iters': cfg.SOLVER.MAX_ITERS}
    trainer = Trainer(**kwargs)

    try:
        trainer.train()
    except BaseException:
        if len(glob(f"{args.output_dir}/*.tar")) < 1:
            shutil.rmtree(args.output_dir)
        raise


if __name__ == '__main__':
    main()

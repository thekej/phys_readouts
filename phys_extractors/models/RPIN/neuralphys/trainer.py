import os
import torch
import numpy as np
from neuralphys.utils.misc import tprint
from neuralphys.utils.config import _C as C
import time


class Trainer(object):
    def __init__(self, device, train_loader, val_loader, model, optim,
                 max_iters, num_gpus, logger, output_dir, max_run_time = 86400 * 100):
        # misc
        self.device = device
        self.output_dir = output_dir
        self.logger = logger
        self.num_gpus = num_gpus
        # data loading
        self.train_loader, self.val_loader = train_loader, val_loader
        # nn optimization
        self.model = model
        self.optim = optim
        # input setting
        self.input_size = C.RPIN.INPUT_SIZE
        self.ptrain_size, self.ptest_size = C.RPIN.PRED_SIZE_TRAIN, C.RPIN.PRED_SIZE_TEST
        self.input_height, self.input_width = C.RPIN.INPUT_HEIGHT, C.RPIN.INPUT_WIDTH
        self.batch_size = C.SOLVER.BATCH_SIZE
        # train loop settings
        self.iterations = 0
        self.epochs = 0
        self.max_iters = max_iters
        self.val_interval = C.SOLVER.VAL_INTERVAL
        self.offset_loss_weight = C.RPIN.OFFSET_LOSS_WEIGHT
        self.position_loss_weight = C.RPIN.POSITION_LOSS_WEIGHT
        # train start time
        # maximum model run time in seconds (86400 sec = 1 day)
        self.max_run_time = max_run_time

    def train(self):
        self.model.train()
        self.logger.info('Start training.')
        while self.iterations < self.max_iters:
            self.train_epoch()
            self.epochs += 1
        self.logger.info('Training done.')

    def train_epoch(self):
        for batch_idx, (data, boxes, labels, data_last, ignore_idx, _, _) in enumerate(self.train_loader):
            self._adjust_learning_rate()
            data = data.to(self.device)
            labels = labels.to(self.device)
            rois, coor_features = self._init_rois(boxes, data.shape)
            self.optim.zero_grad()

            outputs = self.model(data, rois, coor_features, num_rollouts=self.ptrain_size,
                                 data_pred=data_last, phase='train', ignore_idx=ignore_idx)

            loss = self.loss(outputs, labels, 'train', ignore_idx)

            loss.backward()
            self.optim.step()

            self.iterations += self.batch_size

            log_sentence = 'Epoch [{}], Step[{}/{}], Train Loss: {}'.format(self.epochs, self.iterations, self.max_iters, loss.item())

            #tprint(log_sentence)
            self.logger.info(log_sentence)
            
            if self.iterations >= self.max_iters:
                self.snapshot()
                self.val()
                break
                
        self.snapshot('ckpt_epoch_%s.path.tar'%self.epochs)
        self.val()
        self.model.train()

    def val(self):
        self.model.eval()
        val_loss = 0
        for batch_idx, (data, boxes, labels, _, ignore_idx, _, _) in enumerate(self.val_loader):
            tprint(f'eval: {batch_idx}/{len(self.val_loader)}')
            with torch.no_grad():
                data = data.to(self.device)
                labels = labels.to(self.device)
                rois, coor_features = self._init_rois(boxes, data.shape)

                outputs = self.model(data, rois, coor_features, num_rollouts=self.ptest_size,
                                         phase='val', ignore_idx=ignore_idx)
                loss = self.loss(outputs, labels, 'val', ignore_idx)
                val_loss += loss.item()
        log_sentence = 'Epoch [{}] is done, Val Loss: {}'.format(self.epochs, val_loss/len(self.val_loader))
        self.logger.info(log_sentence)

    def loss(self, outputs, labels, phase='train', ignore_idx=None):
        valid_length = self.ptrain_size if phase == 'train' else self.ptest_size

        bbox_rollouts = outputs['bbox']
        # of shape (batch, time, #obj, 4)
        loss = (bbox_rollouts - labels) ** 2
        # take mean except time axis, time axis is used for diagnosis
        ignore_idx = ignore_idx[:, None, :, None].to('cuda')
        loss = loss * ignore_idx
        loss = loss.sum(2) / ignore_idx.sum(2)
        loss[..., 0:2] = loss[..., 0:2] #* self.offset_loss_weight
        loss[..., 2:4] = loss[..., 2:4] #* self.position_loss_weight
        #loss = loss.mean(0).sum()
        loss = loss.mean(0)
        init_tau = C.RPIN.DISCOUNT_TAU ** (1 / self.ptrain_size)
        tau = init_tau + (self.iterations / self.max_iters) * (1 - init_tau)
        tau = torch.pow(tau, torch.arange(11, out=torch.FloatTensor()))[:, None]
        tau = tau.to("cuda")
       # tau = torch.cat([torch.ones(5, 1), tau], dim=0).to('cuda')
        loss = ((loss * tau) / tau.sum(axis=0, keepdims=True)).sum()

        if C.RPIN.VAE and phase == 'train':
            kl_loss = outputs['kl_loss']
            loss += C.RPIN.VAE_KL_LOSS_WEIGHT * kl_loss.sum()

        return loss

    def snapshot(self, name='ckpt_latest.path.tar'):
        torch.save(
            {
                'arch': self.model.__class__.__name__,
                'model': self.model.state_dict(),
            },
            os.path.join(self.output_dir, name),
        )

    def _init_rois(self, boxes, shape):
        batch, time_step, _, height, width = shape
        # coor features, normalized to [0, 1]
        num_im = batch * time_step
        # noinspection PyArgumentList
        co_f = np.zeros(boxes.shape[:-1] + (2,))
        co_f[..., 0] = torch.mean(boxes[..., [0, 2]], dim=-1).numpy().copy() / width
        co_f[..., 1] = torch.mean(boxes[..., [1, 3]], dim=-1).numpy().copy() / height
        coor_features = torch.from_numpy(co_f.astype(np.float32))
        rois = boxes[:, :time_step]
        batch_rois = np.zeros((num_im, C.RPIN.NUM_OBJS))
        batch_rois[np.arange(num_im), :] = np.arange(num_im).reshape(num_im, 1)
        # noinspection PyArgumentList
        batch_rois = torch.FloatTensor(batch_rois.reshape((batch, time_step, -1, 1)))
        # assert batch % self.num_gpus == 0, 'should divide since we drop last in loader'
        load_list = [batch // self.num_gpus for _ in range(self.num_gpus)]
        extra_loaded_gpus = batch - sum(load_list)
        for i in range(extra_loaded_gpus):
            load_list[i] += 1
        load_list = np.cumsum(load_list)
        for i in range(1, self.num_gpus):
            batch_rois[load_list[i - 1]:load_list[i]] -= (load_list[i - 1] * time_step)
        
        rois = torch.cat([batch_rois, rois], dim=-1)
        return rois, coor_features

    def _adjust_learning_rate(self):
        if self.iterations <= C.SOLVER.WARMUP_ITERS:
            lr = C.SOLVER.BASE_LR * self.iterations / C.SOLVER.WARMUP_ITERS
        else:
            if C.SOLVER.SCHEDULER == 'step':
                lr = C.SOLVER.BASE_LR
                for m_iters in C.SOLVER.LR_MILESTONES:
                    if self.iterations > m_iters:
                        lr *= C.SOLVER.LR_GAMMA
            elif C.SOLVER.SCHEDULER == 'cosine':
                lr = 0.5 * C.SOLVER.BASE_LR * (1 + np.cos(np.pi * self.iterations / self.max_iters))
            else:
                raise NotImplementedError

        for param_group in self.optim.param_groups:
            param_group['lr'] = lr

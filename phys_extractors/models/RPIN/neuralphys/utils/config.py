# Copyright (c) Facebook3  Inc. and its affiliates. All Rights Reserved.
from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.DATA_ROOT = ['./data/dynamics/rpin/']
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.PRELOAD_TO_MEMORY = False
_C.INPUT.IMAGE_MEAN = [0, 0, 0]
_C.INPUT.IMAGE_STD = [255.0, 255.0, 255.0]
_C.INPUT.TRAIN_NUM = 1
_C.INPUT.VAL_NUM = 1
_C.INPUT.TRAIN_SLICE = [0, 1024, 1]
_C.INPUT.VAL_SLICE = [0, 1024, 1]
_C.INPUT.BINARY_LABELS = []
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

_C.SOLVER.BASE_LR = 1.e-6
_C.SOLVER.LR_GAMMA = 0.1
_C.SOLVER.VAL_INTERVAL = 16000
_C.SOLVER.WEIGHT_DECAY = 1.e-6
_C.SOLVER.WARMUP_ITERS = -1
_C.SOLVER.LR_MILESTONES = [500000, 1800000]
_C.SOLVER.MAX_ITERS = 10000000
_C.SOLVER.BATCH_SIZE = 16
_C.SOLVER.SCHEDULER = 'step'

# ---------------------------------------------------------------------------- #
# Region Proposal Interaction Network
# ---------------------------------------------------------------------------- #
_C.RPIN = CN()
_C.RPIN.ARCH = 'rpin'
_C.RPIN.BACKBONE = 'hourglass'
_C.RPIN.HORIZONTAL_FLIP = False #True
_C.RPIN.VERTICAL_FLIP = False #True
# prediction setting
_C.RPIN.INPUT_SIZE = 5
_C.RPIN.CONS_SIZE = 1
_C.RPIN.PRED_SIZE_TRAIN = 11
_C.RPIN.PRED_SIZE_TEST = 11
# input for mixed dataset
_C.RPIN.INPUT_HEIGHT = 256
_C.RPIN.INPUT_WIDTH = 256
# training setting
_C.RPIN.NUM_OBJS = 10 #3
_C.RPIN.OFFSET_LOSS_WEIGHT = 1.0
_C.RPIN.POSITION_LOSS_WEIGHT = 1.0
# additional input
_C.RPIN.IMAGE_UP = False
_C.RPIN.ROI_POOL_SIZE = 3
_C.RPIN.COOR_FEATURE = True
_C.RPIN.COOR_FEATURE_EMBEDDING = True
_C.RPIN.IN_CONDITION = True
_C.RPIN.IN_CONDITION_R = 1.5
# parameter
_C.RPIN.VE_FEAT_DIM = 256
_C.RPIN.IN_FEAT_DIM = 256
_C.RPIN.VAE = False
_C.RPIN.VAE_KL_LOSS_WEIGHT = 0.001
# discounted training
_C.RPIN.DISCOUNT_TAU = 0.01

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = './outputs/rpin/'

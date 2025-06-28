import os
import yaml
from yacs.config import CfgNode as CN
import datetime

_C = CN()

# Ablation study
_C.ABLATION = CN()
_C.ABLATION.LOSS = '' # binary classification baseline by only the BCE loss
_C.ABLATION.TASKS = ''

# Base config files
_C.BASE = ['']

_C.LOCAL_RANK = 0
_C.OUTPUT = ''
_C.SAVE_FREQ = 1
_C.PRINT_FREQ = 10
_C.SEED = 0
_C.TAG = 'default'

# model settings
_C.MODEL = CN()
# Model name
_C.MODEL.NAME = 'CLIP_DeepFake_lowerLR'
_C.MODEL.RESUME = ''

# data settings
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 128
_C.DATA.NUM_WORKERS = 8
_C.DATA.PIN_MEMORY = True
_C.DATA.ZIP_MODE = False
_C.DATA.CACHE_MODE = 'part'

# training settings
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS=0
_C.TRAIN.WARMUP_LR=0
# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 5
_C.TRAIN.LR_SCHEDULER.NAME='cosine'

# test settings
_C.TEST = CN()
_C.TEST.SEQUENTIAL = False
_C.TEST.SHUFFLE = False

def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    # _update_config_from_file(config, args.cfg)

    config.defrost()
    # if args.opts:
    #     config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('zip'):
        config.DATA.ZIP_MODE = True
    if _check_args('cache_mode'):
        config.DATA.CACHE_MODE = args.cache_mode
    if _check_args('output'):
        config.OUTPUT = args.output
    if _check_args('num_workers'):
        config.DATA.NUM_WORKERS = args.num_workers
    if _check_args('seed'):
        config.SEED = args.seed
    if _check_args('num_epoch'):
        config.TRAIN.EPOCHS = args.num_epoch
    if _check_args('tag'):
        config.TAG = args.tag
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('name'):
        config.MODEL.NAME = args.name

    # set local rank for distributed training
    # config.LOCAL_RANK = args.local_rank

    config.TAG = '{}-Bz_{}-Epochs_{}-InitialLr_{}-WeightDecay_{}'.format(
        datetime.datetime.now().strftime("%I:%M%p_on_%B_%d_%Y"),
        config.DATA.BATCH_SIZE, config.TRAIN.EPOCHS,
        args.lr, args.weight_decay
    )
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()

def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
import logging
import json
import os
import random
import time
import torch
import torch.utils.data
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from math import ceil, sqrt
from torch.amp import autocast, GradScaler
import torch.distributed
import argparse
import yaml
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import colors

from models import sew_resnet
from mstar_data import load_mstar
from models.submodules.sparse import ConvBlock, Mask
from sparsity.penalty_term import PenaltyTerm
from sparsity.temp_scheduler import TemperatureScheduler
from utils import RecordDict, GlobalTimer, Timer
from utils import CriterionWarpper, SOPMonitor
from utils import left_neurons, init_mask, set_pruning_mode
from utils import is_main_process, save_on_master, search_tb_record, finetune_tb_record, accuracy, safe_makedirs
from spikingjelly.activation_based import functional


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='MSTAR training and neuron pruning')
    parser.add_argument('--config', help='YAML training configuration')

    # training options
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--epoch-search', default=100, type=int)
    parser.add_argument('--epoch-finetune', default=20, type=int,
                        help='when to fine tune, -1 means will not fine tune')
    parser.add_argument('--not-prune-neuron', action='store_true')
    parser.add_argument('-b', '--batch-size', default=64, type=int)
    parser.add_argument('--T', default=4, type=int, help='simulation steps')
    parser.add_argument('--model', default='sew_resnet18', help='model type')
    parser.add_argument('--dataset', default='MSTAR', choices=['MSTAR'], help='MSTAR SOC dataset')
    parser.add_argument('--augment', action='store_true', default=True, help='Additional augment')
    parser.add_argument('--device', default='cuda', choices=['cuda'], help='device')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--search-lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--finetune-lr', default=0.0001, type=float, help='finetune learning rate')
    parser.add_argument('--prune-lr', type=float, help='initial learning rate of pruning')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='SGD, Adam or AdamW (case insensitive)')
    parser.add_argument('--nesterov', action='store_true', help='Use Nesterov momentum for SGD')
    parser.add_argument('--prune-optimizer', type=str, help='SGD, Adam or AdamW (case insensitive)')
    parser.add_argument('--weight-decay', default=0.0001, type=float, help='weight decay (default: 0.0001)')
    parser.add_argument('--prune-weight-decay', default=0, type=float)
    parser.add_argument(
        '--search-lr-scheduler', type=str, nargs='+', default=['Cosine'],
        help='''--lr-scheduler Cosine [<T0> <Tt> <Tmax(period of cosine)>]
            or --lr-scheduler Step [minestones]...''')
    parser.add_argument(
        '--finetune-lr-scheduler', type=str, nargs='+', default=['Cosine'],
        help='''--lr-scheduler Cosine [<T0> <Tt> <Tmax(period of cosine)>]
            or --lr-scheduler Step [minestones]...''')
    parser.add_argument('--print-freq', default=5, type=int,
                        help='Number of times a debug message is printed in one epoch')
    parser.add_argument('--tb-interval', type=int, default=10)
    parser.add_argument('--data-path', default='./datasets/MSTAR/soc', help='dataset')
    parser.add_argument('--output-dir', default='./logs/temp')
    parser.add_argument('--resume', type=str, help='resume from checkpoint')
    parser.add_argument('--resume-type', type=str, default='test', help='search, finetune or test')
    parser.add_argument('--distributed-init-mode', type=str, default='env://')

    parser.add_argument('--no-aug', action='store_true', help='no augmentation')
    parser.add_argument('--label-smoothing', default=0.0, type=float)
    # mask init
    parser.add_argument(
        '--mask-init-factor', type=float, nargs='+', default=[0, 0],
        help='--mask-init-factor <neurons mean> <neurons std>')

    # penalty term
    parser.add_argument('--penalty-lmbda', type=float, default=1e-5)

    parser.add_argument(
        '--temp-scheduler', type=float, nargs='+', default=[5, 1000],
        help='''--temp-scheduler <init temp> <final temp>
                or --temp-scheduler <init temp> <final temp> <T0> <Tmax>''')
    # deprecated
    parser.add_argument('--accumulate-step', type=int, default=1)

    # argument of sew resnet
    parser.add_argument('--zero-init-residual', action='store_true',
                        help='zero init all residual blocks')
    parser.add_argument("--sync-bn", action="store_true", help="Use sync batch norm")
    parser.add_argument("--test-only", action="store_true", help="Only test the model")
    parser.add_argument('--amp', action='store_true', default=True, help='Use AMP training')

    # argument of TET
    parser.add_argument('--TET', action='store_true', help='Use TET training')
    parser.add_argument('--TET-phi', type=float, default=1.0)
    parser.add_argument('--TET-lambda', type=float, default=0.0)

    parser.add_argument('--save-latest', action='store_true')
    parser.add_argument('--eval-final-only', action='store_true', default=True,
                        help='Evaluate test data only after the final training phase')

    # Register explicit negative forms for every existing boolean option.
    for action in list(parser._actions):
        if isinstance(action, argparse._StoreTrueAction):
            parser.add_argument('--no-' + action.option_strings[0][2:],
                                dest=action.dest, action='store_false', default=argparse.SUPPRESS)

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config')
    config_args, _ = config_parser.parse_known_args(argv)
    if config_args.config:
        with open(config_args.config, encoding='utf-8') as config_file:
            cfg = yaml.safe_load(config_file)
        if not isinstance(cfg, dict):
            parser.error('Configuration must be a YAML mapping')
        actions = {action.dest: action for action in parser._actions}
        unknown = set(cfg) - (set(actions) - {'help', 'config'})
        if unknown:
            parser.error('Unknown configuration keys: ' + ', '.join(sorted(map(str, unknown))))
        for key, value in cfg.items():
            action = actions[key]
            if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                if not isinstance(value, bool):
                    parser.error('{} must be a YAML boolean'.format(key))
            elif value is not None and action.type is not None:
                try:
                    if action.nargs == '+':
                        if not isinstance(value, list) or not value:
                            # An empty schedule explicitly disables LR scheduling.
                            if value == [] and key.endswith('lr_scheduler'):
                                continue
                            raise ValueError('expected a non-empty list')
                        cfg[key] = [action.type(item) for item in value]
                    else:
                        cfg[key] = action.type(value)
                except (TypeError, ValueError) as error:
                    parser.error('Invalid {}: {}'.format(key, error))
        parser.set_defaults(**cfg)
    args = parser.parse_args(argv)
    if not isinstance(args.mask_init_factor, list) or len(args.mask_init_factor) != 2:
        parser.error('--mask-init-factor requires <neurons mean> <neurons std>')
    if not isinstance(args.temp_scheduler, list) or len(args.temp_scheduler) not in (2, 4):
        parser.error('--temp-scheduler requires 2 or 4 values')
    if args.resume_type not in ('search', 'finetune', 'test'):
        parser.error('--resume-type must be search, finetune or test')
    if args.test_only and (not args.resume or args.resume_type != 'test'):
        parser.error('--test-only requires --resume and --resume-type test')
    if args.resume and args.resume_type == 'test' and not args.test_only:
        parser.error('--resume-type test requires --test-only')
    if not args.resume and args.resume_type != 'test':
        parser.error('--resume-type search/finetune requires --resume')
    if args.epoch_search < 0 or args.epoch_finetune < -1:
        parser.error('Invalid epoch count')
    if args.T < 1 or args.batch_size < 1 or args.accumulate_step < 1:
        parser.error('T, batch-size and accumulate-step must be positive')
    if args.dataset != 'MSTAR':
        parser.error('Only the MSTAR dataset is supported')
    args.model = args.model.removesuffix('_imagenet')
    return args


def setup_logger(output_dir):
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s]%(message)s',
                                  datefmt=r'%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(os.path.join(output_dir, 'log.log'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    return logger


def create_lr_scheduler(optimizer, schedule, epochs):
    """Return the scheduler and its active epoch interval for one phase."""
    start, end = 0, epochs
    if optimizer is None or not schedule or epochs <= 0:
        return None, start, end
    if schedule[0] == 'Step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(value) for value in schedule[1:]], gamma=0.1)
    elif schedule[0] == 'Cosine':
        if len(schedule) == 1:
            period = epochs
        elif len(schedule) == 4:
            start, end, period = map(int, schedule[1:])
        else:
            raise ValueError('Cosine expects either no arguments or T0, Tt, Tmax')
        if period <= 0 or end <= start:
            raise ValueError('Cosine requires a positive period and Tt > T0')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=period, eta_min=0)
    else:
        raise ValueError(schedule)
    return scheduler, start, end


def reset_learning_rate(optimizer, lr):
    """Start a new phase without inheriting the previous scheduler's base LR."""
    for group in optimizer.param_groups:
        group['lr'] = lr
        group['initial_lr'] = lr


def create_optimizer(model_or_params, name, lr, weight_decay, nesterov=False):
    name = name.lower()
    if isinstance(model_or_params, nn.Module):
        if name == 'adamw':
            # Keep the incoming AdamW recipe's bias/norm decay exclusion.
            decay, no_decay = [], []
            for parameter in model_or_params.parameters():
                if parameter.requires_grad:
                    (no_decay if parameter.ndim <= 1 else decay).append(parameter)
            params = [{'params': decay}, {'params': no_decay, 'weight_decay': 0.0}]
        else:
            params = model_or_params.parameters()
    else:
        params = model_or_params
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay,
                               nesterov=nesterov)
    if name == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError('Unsupported optimizer: {}'.format(name))


def init_distributed(logger: logging.Logger, distributed_init_mode):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        logger.info('Not using distributed mode')
        return False, 0, 1, 0

    torch.cuda.set_device(local_rank)
    backend = 'nccl'
    logger.info('Distributed init rank {}'.format(rank))
    torch.distributed.init_process_group(backend=backend, init_method=distributed_init_mode,
                                         world_size=world_size, rank=rank)
    # only master process logs
    if rank != 0:
        logger.setLevel(logging.WARNING)
    return True, rank, world_size, local_rank


def load_data(dataset_dir, distributed, args):
    dataset_train, dataset_test, args.mstar_metadata = load_mstar(
        dataset_dir, args.augment and not args.no_aug,
        getattr(args, 'mstar_metadata', None), args.test_only)
    data_loader_train = None
    if dataset_train is not None:
        sampler = (torch.utils.data.distributed.DistributedSampler(dataset_train)
                   if distributed else torch.utils.data.RandomSampler(dataset_train))
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.workers, pin_memory=True, drop_last=True)
    # Final evaluation runs on rank zero over the complete test split.
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    return dataset_train, dataset_test, data_loader_train, data_loader_test


def train_one_epoch(model, criterion, penalty_term, optimizer_train, optimizer_prune,
                    data_loader_train, temp_scheduler, logger, epoch, print_freq, factor,
                    scaler=None, accumulate_step=1, prune=False):
    model.train()
    metric_dict = RecordDict({'loss': None, 'acc@1': None, 'acc@5': None})
    timer_container = [0.0]

    set_pruning_mode(model, prune)
    model.zero_grad()
    for idx, (image, target) in enumerate(data_loader_train):
        with GlobalTimer('iter', timer_container):
            image, target = image.float().cuda(), target.cuda()
            if scaler is not None:
                with autocast('cuda'):
                    output = model(image)
                    loss = criterion(output, target)
            else:
                output = model(image)
                loss = criterion(output, target)
            metric_dict['loss'].update(loss.item())

            if prune:
                loss = loss + penalty_term()

            loss = loss / accumulate_step

            if scaler is not None:
                scaler.scale(loss).backward()
                if (idx + 1) % accumulate_step == 0:
                    if prune:
                        scaler.step(optimizer_prune)
                    scaler.step(optimizer_train)
                    scaler.update()
                    model.zero_grad()
                    if temp_scheduler is not None:
                        temp_scheduler.step()

            else:
                loss.backward()
                if (idx + 1) % accumulate_step == 0:
                    if prune:
                        optimizer_prune.step()
                    optimizer_train.step()
                    model.zero_grad()
                    if temp_scheduler is not None:
                        temp_scheduler.step()

            functional.reset_net(model)

            acc1, acc5 = accuracy(output.mean(0), target, topk=(1, 5))
            acc1_s = acc1.item()
            acc5_s = acc5.item()

            batch_size = image.shape[0]
            metric_dict['acc@1'].update(acc1_s, batch_size)
            metric_dict['acc@5'].update(acc5_s, batch_size)

        if print_freq != 0 and ((idx + 1) % max(1, int(len(data_loader_train) / print_freq))) == 0:
            #torch.distributed.barrier()
            metric_dict.sync()
            logger.debug(' [{}/{}] it/s: {:.5f}, loss: {:.5f}, acc@1: {:.5f}, acc@5: {:.5f}'.format(
                idx + 1, len(data_loader_train),
                (idx + 1) * batch_size * factor / timer_container[0], metric_dict['loss'].ave,
                metric_dict['acc@1'].ave, metric_dict['acc@5'].ave))

    #torch.distributed.barrier()
    metric_dict.sync()
    return metric_dict['loss'].ave, metric_dict['acc@1'].ave, metric_dict['acc@5'].ave


def evaluate(model, criterion, data_loader, print_freq, logger, prune):
    model.eval()
    set_pruning_mode(model, prune)
    metric_dict = RecordDict({'loss': None, 'acc@1': None, 'acc@5': None})
    with torch.no_grad():
        for idx, (image, target) in enumerate(data_loader):
            image = image.float().to(torch.device('cuda'), non_blocking=True)
            target = target.to(torch.device('cuda'), non_blocking=True)
            output = model(image)
            loss = criterion(output, target)
            metric_dict['loss'].update(loss.item())
            functional.reset_net(model)

            acc1, acc5 = accuracy(output.mean(0), target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_dict['acc@1'].update(acc1.item(), batch_size)
            metric_dict['acc@5'].update(acc5.item(), batch_size)

            if print_freq != 0 and ((idx + 1) % max(1, int(len(data_loader) / print_freq))) == 0:
                #torch.distributed.barrier()
                metric_dict.sync()
                logger.debug(' [{}/{}] loss: {:.5f}, acc@1: {:.5f}, acc@5: {:.5f}'.format(
                    idx + 1, len(data_loader), metric_dict['loss'].ave, metric_dict['acc@1'].ave,
                    metric_dict['acc@5'].ave))

    #torch.distributed.barrier()
    metric_dict.sync()
    return metric_dict['loss'].ave, metric_dict['acc@1'].ave, metric_dict['acc@5'].ave


def measure_inference_time(model, data_loader, warmup_steps=5):
    """Measure synchronized model execution, amortized over actual sample count."""
    model.eval()
    elapsed_seconds = 0.0
    samples = 0
    with torch.no_grad():
        for image, _ in data_loader:
            image = image.cuda()
            if samples == 0:
                for _ in range(warmup_steps):
                    model(image).mean(0)
                    functional.reset_net(model)
            # Exclude loading and host-to-device transfer from the timed region.
            torch.cuda.synchronize(image.device)
            start = time.perf_counter()
            model(image).mean(0)
            functional.reset_net(model)
            torch.cuda.synchronize(image.device)
            elapsed_seconds += time.perf_counter() - start
            samples += image.shape[0]
    if samples == 0:
        raise ValueError('Cannot measure inference time on an empty dataset')
    return elapsed_seconds * 1000 / samples


def test(model, data_loader_test, inputs, args, logger):

    safe_makedirs(os.path.join(args.output_dir, 'test'))

    set_pruning_mode(model, False)
    model.eval()
    neuron_left, neuron_total = 0, 0

    def count_neurons(module, block_inputs, output):
        nonlocal neuron_left, neuron_total
        # Count spatial LIF units once, independent of batch size and time steps.
        count = output[0, 0].numel()
        neuron_total += count
        mask = module.neuron_mask.mask_value
        neuron_left += int((mask > 0).sum().item()) if mask is not None else count

    hooks = [module.register_forward_hook(count_neurons) for module in model.modules()
             if isinstance(module, ConvBlock) and module.node is not None]
    try:
        with torch.no_grad():
            model(inputs)
    finally:
        for hook in hooks:
            hook.remove()
        functional.reset_net(model)
    mon = SOPMonitor(model)

    logger.info('[Sparsity]')

    logger.info('Neurons: before pruning: {:d}, after pruning: {:d}, remaining: {:.2f}%'.format(
        neuron_total, neuron_left, 100 * neuron_left / neuron_total if neuron_total else 100.0))

    logger.info('[Efficiency]')

    mon.enable()
    logger.debug('Test start')
    metric_dict = RecordDict({'acc@1': None, 'acc@5': None}, test=True)
    with torch.no_grad():
        for idx, (image, target) in enumerate(data_loader_test):
            image, target = image.cuda(), target.cuda()
            output = model(image).mean(0)
            functional.reset_net(model)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_dict['acc@1'].update(acc1.item(), batch_size)
            metric_dict['acc@5'].update(acc5.item(), batch_size)

            if args.print_freq != 0 and ((idx + 1) %
                                         max(1, int(len(data_loader_test) / args.print_freq))) == 0:
                logger.debug('Test: [{}/{}]'.format(idx + 1, len(data_loader_test)))

    metric_dict.sync()
    logger.info('Acc@1: {:.5f}, Acc@5: {:.5f}'.format(metric_dict['acc@1'].ave,
                                                      metric_dict['acc@5'].ave))

    sops = 0
    for name in mon.monitored_layers:
        sublist = mon[name]
        sop = torch.cat(sublist).sum().item()
        sops = sops + sop
    sops = sops / (1000**3)
    # input is [N, C, H, W] or [T*N, C, H, W]
    sops = sops / len(data_loader_test.dataset)
    logger.info('Avg SOPs: {:.5f} G, Power: {:.5f} mJ.'.format(sops, 0.9 * sops))
    # Benchmark separately so SOP hooks and metric calculation do not affect timing.
    mon.remove_hooks()
    inference_time_ms = measure_inference_time(model, data_loader_test)
    logger.info('Avg inference time: {:.6f} ms/sample (batch size: {}, warmup: 5 batches)'.format(
        inference_time_ms, args.batch_size))
    with open(os.path.join(args.output_dir, 'test', 'metrics.json'), 'w', encoding='utf-8') as file:
        json.dump({'accuracy_top1': metric_dict['acc@1'].ave,
                   'accuracy_top5': metric_dict['acc@5'].ave,
                   'accuracy_unit': 'percent', 'n': len(data_loader_test.dataset),
                   'neurons_left': neuron_left, 'neurons_total': neuron_total,
                   'inference_time_ms_per_sample': inference_time_ms,
                   'inference_batch_size': args.batch_size,
                   'inference_warmup_steps': 5,
                   'sops_g_per_image': sops}, file, indent=2)

    #
    # visualize neurons
    #

    logger.info('[Neurons]')

    fig = plt.figure(figsize=(16, 16))
    norm = colors.Normalize(vmin=-1.5, vmax=1.5)
    idx = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, ConvBlock):
                if not module.sparse_neurons:
                    continue
                mask = module.neuron_mask.mask_value
                num_active = (mask > 0).sum().item()
                num = mask.numel()
                logger.info('layer [{}] {}: left: {}, total: {}, percentage: {:.2f}%'.format(
                    idx, name, num_active, num, 100 * num_active / num))
                mask = mask.squeeze(0).squeeze(0)
                channels = mask.shape[0]
                ncols = int(sqrt(channels))
                nrows = ceil(channels / ncols)
                for c in range(channels):
                    ax = fig.add_subplot(nrows, ncols, c + 1)
                    ax.matshow((mask[c, ...] > 0).cpu().numpy(), cmap='bwr', norm=norm)
                    ax.axis('off')
                fig.tight_layout()
                fig.savefig(os.path.join(args.output_dir, 'test', f'neuron_{idx}_{name}.png'),
                            bbox_inches='tight')
                fig.clear()
                ax = fig.add_subplot(1, 1, 1)
                mask = mask.flatten().cpu().numpy()
                percentile = np.percentile(np.abs(mask), 99)
                ax.hist(mask[mask > 0], bins=1000, range=(0, percentile))
                ax.hist(mask[mask < 0], bins=1000, range=(-percentile, 0))
                fig.tight_layout()
                fig.savefig(os.path.join(args.output_dir, 'test', f'neuron_plot_{idx}_{name}.png'),
                            bbox_inches='tight')
                fig.clear()
                idx = idx + 1
        plt.close()


def create_model_and_inputs(args):
    """Use the same architecture and mask geometry for training and evaluation."""
    if args.model not in sew_resnet.__all__:
        raise ValueError('Unsupported model: {}'.format(args.model))
    model = sew_resnet.__dict__[args.model](
        zero_init_residual=args.zero_init_residual, T=args.T, num_classes=10, in_channels=1).cuda()
    if args.not_prune_neuron:
        for module in model.modules():
            if isinstance(module, ConvBlock):
                module.sparse_neurons = False
    return model, torch.rand(1, 1, 128, 128, device='cuda')


def main():

    ##################################################
    #                       setup
    ##################################################

    args = parse_args()
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        saved_config = resume_checkpoint.get('config', {})
        # Restore fields needed to reconstruct the network and evaluation protocol.
        for key in ('dataset', 'model', 'T', 'not_prune_neuron', 'zero_init_residual',
                    'eval_final_only'):
            if key in saved_config:
                setattr(args, key, saved_config[key])
        args.mstar_metadata = resume_checkpoint.get('mstar_metadata')
        if args.mstar_metadata is None:
            raise ValueError('MSTAR checkpoint is missing preprocessing metadata')
        if not args.test_only and resume_checkpoint.get('phase', args.resume_type) != args.resume_type:
            raise ValueError('Resume type does not match checkpoint phase')
    if not args.test_only and not args.resume and args.epoch_search == 0:
        raise ValueError('At least one search/training epoch is required for a fresh run')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    safe_makedirs(args.output_dir)
    logger = setup_logger(args.output_dir)

    distributed, rank, world_size, local_rank = init_distributed(logger, args.distributed_init_mode)

    logger.info(str(args))

    # load data

    if args.dataset != 'MSTAR':
        raise ValueError('Only MSTAR checkpoints are supported')
    args.model = args.model.removesuffix('_imagenet')
    dataset_train, dataset_test, data_loader_train, data_loader_test = load_data(
        args.data_path, distributed, args)
    logger.info('dataset_train: {}, dataset_test: {}'.format(
        len(dataset_train) if dataset_train is not None else 0, len(dataset_test)))
    if not args.test_only and len(data_loader_train) == 0:
        raise ValueError('Training loader is empty; reduce batch-size')
    if not args.test_only and len(data_loader_train) < args.accumulate_step:
        raise ValueError('accumulate-step exceeds the number of training batches')
    if is_main_process():
        with open(os.path.join(args.output_dir, 'config.json'), 'w', encoding='utf-8') as file:
            json.dump(vars(args), file, indent=2, ensure_ascii=False)

    # model

    model, inputs = create_model_and_inputs(args)
    if distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # optimzer

    optimizer_train = create_optimizer(model, args.optimizer, args.search_lr,
                                       args.weight_decay, args.nesterov)

    optimizer_prune = None
    penalty_term = None
    temp_scheduler = None
    pruning_enabled = not args.not_prune_neuron

    # init mask

    set_pruning_mode(model, True)
    _ = model(inputs)

    masks = init_mask(model, *args.mask_init_factor)
    set_pruning_mode(model, False)
    functional.reset_net(model)

    if args.test_only:
        model.load_state_dict(resume_checkpoint['model'])
        if is_main_process():
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            test(model, data_loader_test, inputs, args, logger)
        return

    if not args.not_prune_neuron:
        if args.prune_optimizer is None:
            args.prune_optimizer = args.optimizer
        if args.prune_lr is None:
            args.prune_lr = args.search_lr
        optimizer_prune = create_optimizer(masks, args.prune_optimizer, args.prune_lr,
                                           args.prune_weight_decay, args.nesterov)

    # loss_fn

    base_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    criterion = CriterionWarpper(base_criterion, args.TET, args.TET_phi, args.TET_lambda)
    criterion_eval = CriterionWarpper(nn.CrossEntropyLoss())

    # penalty term

    if not args.not_prune_neuron:
        penalty_term = PenaltyTerm(model, args.penalty_lmbda)

    # amp speed up

    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    # lr scheduler

    lr_scheduler_train, lr_scheduler_T0, lr_scheduler_Tmax = create_lr_scheduler(
        optimizer_train, args.search_lr_scheduler, args.epoch_search)
    lr_scheduler_prune, _, _ = create_lr_scheduler(
        optimizer_prune, args.search_lr_scheduler, args.epoch_search)

    # DDP

    model_without_ddp = model
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          find_unused_parameters=True)
        model_without_ddp = model.module

    # threshold scheduler

    if pruning_enabled and args.resume_type != 'finetune':
        iter_per_epoch = len(data_loader_train) // args.accumulate_step
        if len(args.temp_scheduler) == 2:
            (args.temp_scheduler).append(0)
            (args.temp_scheduler).append(args.epoch_search)
        if len(args.temp_scheduler) == 4:
            temp_scheduler = TemperatureScheduler(model, args.temp_scheduler[0],
                                                  args.temp_scheduler[1],
                                                  int(args.temp_scheduler[2]) * iter_per_epoch,
                                                  int(args.temp_scheduler[3]) * iter_per_epoch)
        else:
            raise ValueError(args.temp_scheduler)

    # resume

    if args.resume and args.resume_type == 'search':
        checkpoint = resume_checkpoint
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer_train.load_state_dict(checkpoint['optimizer_train'])
        if scaler is not None and checkpoint.get('scaler') is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        if optimizer_prune is not None:
            optimizer_prune.load_state_dict(checkpoint['optimizer_prune'])
        start_epoch = checkpoint['epoch']
        max_acc1 = checkpoint['max_acc1']
        if lr_scheduler_train is not None:
            lr_scheduler_train.load_state_dict(checkpoint['lr_scheduler_train'])
            if lr_scheduler_prune is not None:
                lr_scheduler_prune.load_state_dict(checkpoint['lr_scheduler_prune'])
        logger.info('Resume from epoch {}'.format(start_epoch))
        start_epoch += 1
        if temp_scheduler is not None:
            temp_scheduler.current_step = start_epoch * iter_per_epoch - 1
            temp_scheduler.step()
    else:
        start_epoch = 0
        max_acc1 = 0

    logger.debug(str(model))

    ##################################################
    #                   search
    ##################################################

    tb_writer = None
    if is_main_process():
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'),
                                  purge_step=start_epoch)

    logger.info("Search start")
    for epoch in range(start_epoch, args.epoch_search):
        if args.resume and args.resume_type == 'finetune':
            break
        if distributed:
            data_loader_train.sampler.set_epoch(epoch)
        logger.info('Epoch [{}] Start, lr {:.6f}, {}'.format(epoch,
                                                             optimizer_train.param_groups[0]["lr"],
                                                             str(temp_scheduler)))

        with Timer(' Train', logger):
            logger.debug('[Training]')
            train_loss, train_acc1, train_acc5 = train_one_epoch(
                model, criterion, penalty_term, optimizer_train, optimizer_prune, data_loader_train,
                temp_scheduler, logger, epoch, args.print_freq, world_size, scaler,
                args.accumulate_step, pruning_enabled)
            if lr_scheduler_train is not None and lr_scheduler_T0 <= epoch < lr_scheduler_Tmax:
                lr_scheduler_train.step()
                if lr_scheduler_prune is not None:
                    lr_scheduler_prune.step()

        for n, m in model.named_modules():
            if isinstance(m, Mask):
                if m.mask_value is not None:
                    logger.debug(' {}: {:.3}%'.format(n, m.mask().mean() * 100))

        test_loss_c = test_acc1_c = test_acc5_c = None
        test_loss_s = test_acc1_s = test_acc5_s = None
        if not args.eval_final_only:
            with Timer(' Test', logger):
                logger.debug('[Test with continuous mask]')
                test_loss_c, test_acc1_c, test_acc5_c = evaluate(model, criterion_eval, data_loader_test,
                                                                 args.print_freq, logger, True)
                logger.debug('[Test with binary mask]')
                test_loss_s, test_acc1_s, test_acc5_s = evaluate(model, criterion_eval, data_loader_test,
                                                                 args.print_freq, logger, False)
        set_pruning_mode(model, True)
        n_l, n_t = left_neurons(model)
        neu = 100 * (n_l + 1e-10) / (n_t + 1e-10)
        search_tb_record(tb_writer, model, train_loss, train_acc1, train_acc5, test_loss_c,
                         test_acc1_c, test_acc5_c, test_loss_s, test_acc1_s, test_acc5_s, epoch,
                         args.tb_interval)

        if not args.eval_final_only:
            logger.info(' Test (continuous mask) Acc@1: {:.5f}, Acc@5: {:.5f}'.format(
                test_acc1_c, test_acc5_c))
            logger.info(' Test (binary mask) Acc@1: {:.5f}, Acc@5: {:.5f}'.format(
                test_acc1_s, test_acc5_s))
        logger.info(' left neurons: {:.2f}%'.format(neu))

        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer_train': optimizer_train.state_dict(),
            'optimizer_prune': optimizer_prune.state_dict() if optimizer_prune is not None else None,
            'epoch': epoch,
            'max_acc1': max_acc1,
            'config': vars(args).copy(),
            'mstar_metadata': getattr(args, 'mstar_metadata', None),
            'scaler': scaler.state_dict() if scaler is not None else None,
            'phase': 'search', }
        if lr_scheduler_train is not None:
            checkpoint['lr_scheduler_train'] = lr_scheduler_train.state_dict()
            if lr_scheduler_prune is not None:
                checkpoint['lr_scheduler_prune'] = lr_scheduler_prune.state_dict()

        if args.save_latest:
            save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoint_latest.pth'))

        if (epoch + 1) == args.epoch_search:
            save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoint_sparsified.pth'))

    logger.info('Search finish.')

    ##################################################
    #                   finetune
    ##################################################

    ##### reset utils #####

    # reset lr
    if args.finetune_lr is None:
        args.finetune_lr = args.search_lr
    reset_learning_rate(optimizer_train, args.finetune_lr)
    lr_scheduler_train, lr_scheduler_T0, lr_scheduler_Tmax = create_lr_scheduler(
        optimizer_train, args.finetune_lr_scheduler, args.epoch_finetune)

    # resume

    if args.resume and args.resume_type == 'finetune':
        checkpoint = resume_checkpoint
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer_train.load_state_dict(checkpoint['optimizer_train'])
        if scaler is not None and checkpoint.get('scaler') is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch']
        max_acc1 = checkpoint['max_acc1']
        if lr_scheduler_train is not None:
            lr_scheduler_train.load_state_dict(checkpoint['lr_scheduler_train'])
        logger.info('Resume from epoch {}'.format(start_epoch))
        start_epoch += 1
    else:
        start_epoch = 0

    ##### finetune #####

    logger.info("Finetune start")
    for epoch in range(start_epoch, args.epoch_finetune):
        save_max = False
        if distributed:
            data_loader_train.sampler.set_epoch(epoch)
        logger.info('Epoch [{}] Start, lr {:.6f}'.format(epoch,
                                                         optimizer_train.param_groups[0]["lr"]))

        with Timer(' Train', logger):
            logger.debug('[Training]')
            train_loss, train_acc1, train_acc5 = train_one_epoch(
                model, criterion, None, optimizer_train, None, data_loader_train, None, logger,
                epoch, args.print_freq, world_size, scaler, args.accumulate_step, False)
            if lr_scheduler_train is not None and lr_scheduler_T0 <= epoch < lr_scheduler_Tmax:
                lr_scheduler_train.step()

        test_loss = test_acc1 = test_acc5 = None
        if not args.eval_final_only:
            with Timer(' Test', logger):
                logger.debug('[Test]')
                test_loss, test_acc1, test_acc5 = evaluate(model, criterion_eval, data_loader_test,
                                                           args.print_freq, logger, False)

        finetune_tb_record(tb_writer, train_loss, train_acc1, train_acc5, test_loss, test_acc1,
                           test_acc5, epoch)

        if not args.eval_final_only:
            logger.info(' Test Acc@1: {:.5f}, Acc@5: {:.5f}'.format(test_acc1, test_acc5))

            if max_acc1 < test_acc1:
                max_acc1 = test_acc1
                save_max = True

        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer_train': optimizer_train.state_dict(),
            'epoch': epoch,
            'max_acc1': max_acc1,
            'config': vars(args).copy(),
            'mstar_metadata': getattr(args, 'mstar_metadata', None),
            'scaler': scaler.state_dict() if scaler is not None else None,
            'phase': 'finetune', }
        if lr_scheduler_train is not None:
            checkpoint['lr_scheduler_train'] = lr_scheduler_train.state_dict()

        if args.save_latest:
            save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoint_latest.pth'))

        if save_max:
            save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoint_max_acc1.pth'))

    logger.info('Finetune finish.')
    # Also materialize the final checkpoint when resuming an already completed phase.
    save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoint_final.pth'))

    ##################################################
    #                   test
    ##################################################

    ##### reset utils #####

    # reset model

    selected = ('checkpoint_final.pth' if args.eval_final_only or args.epoch_finetune <= 0
                else 'checkpoint_max_acc1.pth')
    # All ranks must finish saving before the final checkpoint is loaded.
    if distributed:
        torch.distributed.barrier()
    checkpoint_path = os.path.join(args.output_dir, selected)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if tb_writer is not None:
        tb_writer.close()
    if not is_main_process():
        return
    del model, model_without_ddp
    model, inputs = create_model_and_inputs(args)
    model.eval()
    with torch.no_grad():
        model(inputs)
    init_mask(model, 1, 0)
    functional.reset_net(model)

    model.load_state_dict(checkpoint['model'])

    # reload data

    del data_loader_test

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                                   sampler=test_sampler, num_workers=args.workers,
                                                   pin_memory=False, drop_last=False)

    ##### test #####

    logger.info('Test start')
    if is_main_process():
        test(model, data_loader_test, inputs, args, logger)
    logger.info('All Done.')


if __name__ == "__main__":
    main()

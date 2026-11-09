import math
import os
import time
import torch
import torch.distributed
import errno
import datetime
from torch import Tensor, nn
from math import nan
from torch.utils.tensorboard import SummaryWriter

from models.submodules.sparse import ConvBlock


def is_distributed():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_rank():
    if not is_distributed():
        return 0
    return torch.distributed.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def safe_makedirs(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def search_tb_record(tb_writer: SummaryWriter, model: nn.Module, train_loss, train_acc1, train_acc5,
                     test_loss_c, test_acc1_c, test_acc5_c, test_loss_s, test_acc1_s, test_acc5_s,
                     epoch, interval=1):
    if is_main_process():
        tb_writer.add_scalar('search/train/loss', train_loss, epoch)
        tb_writer.add_scalar('search/train/acc1', train_acc1, epoch)
        tb_writer.add_scalar('search/train/acc5', train_acc5, epoch)
        if test_loss_c is not None:
            tb_writer.add_scalar('search/test/continuous/loss', test_loss_c, epoch)
            tb_writer.add_scalar('search/test/continuous/acc1', test_acc1_c, epoch)
            tb_writer.add_scalar('search/test/continuous/acc5', test_acc5_c, epoch)
            tb_writer.add_scalar('search/test/binary/loss', test_loss_s, epoch)
            tb_writer.add_scalar('search/test/binary/acc1', test_acc1_s, epoch)
            tb_writer.add_scalar('search/test/binary/acc5', test_acc5_s, epoch)

        neu_cnt, neu_total = 0, 0
        for name, module in model.named_modules():
            if isinstance(module, ConvBlock):
                if (epoch + 1) % interval == 0 or epoch == 0:
                    weight = module.conv.weight
                    tb_writer.add_histogram(f'weight/{name}', weight, epoch)
                if module.sparse_neurons:
                    n_c, n_t = module.neuron_mask.left()
                    neu_cnt, neu_total = neu_cnt + n_c, neu_total + n_t
                    tb_writer.add_scalar(f'left neurons/{name}', n_c / n_t, epoch)
                    if (epoch + 1) % interval == 0 or epoch == 0:
                        tb_writer.add_histogram(f'neuron mask/{name}',
                                                module.neuron_mask.mask_value, epoch)
        if neu_total != 0:
            tb_writer.add_scalar('left neurons', neu_cnt / neu_total, epoch)


def finetune_tb_record(tb_writer: SummaryWriter, train_loss, train_acc1, train_acc5, test_loss,
                       test_acc1, test_acc5, epoch):
    if is_main_process():
        tb_writer.add_scalar('finetune/train/loss', train_loss, epoch)
        tb_writer.add_scalar('finetune/train/acc1', train_acc1, epoch)
        tb_writer.add_scalar('finetune/train/acc5', train_acc5, epoch)
        if test_loss is not None:
            tb_writer.add_scalar('finetune/test/loss', test_loss, epoch)
            tb_writer.add_scalar('finetune/test/acc1', test_acc1, epoch)
            tb_writer.add_scalar('finetune/test/acc5', test_acc5, epoch)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, )):
    r"""
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def init_mask(model: nn.Module, neurons_mean: float, neurons_std: float = 0):
    mask_list = []
    for m in model.modules():
        if isinstance(m, ConvBlock):
            masks = m.init_mask(neurons_mean, neurons_std)
            for mask in masks:
                mask_list.append(mask)
    return mask_list


def set_pruning_mode(model: nn.Module, mode: bool = False):
    for m in model.modules():
        if isinstance(m, ConvBlock):
            m._pruning(mode)


def left_neurons(model: nn.Module):
    conn = 0
    total = 0
    for m in model.modules():
        if isinstance(m, ConvBlock):
            c, t = m.neuron_mask.left()
            conn, total = conn + c, total + t
    return conn, total


class Record:
    r'''
    Synchronous record
    '''
    def __init__(self, test: bool = False) -> None:
        self.value = torch.tensor([0], dtype=torch.float64, device='cuda')
        self.count = torch.tensor([0], dtype=torch.int64, device='cuda')
        self.global_value = 0.0
        self.global_count = 0
        self.test = test

    def sync(self) -> None:
        r'''
        reduce value and count, and update global ones
        '''
        if is_distributed() and not self.test:
            torch.distributed.all_reduce(self.value, torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(self.count, torch.distributed.ReduceOp.SUM)
        self.global_value += self.value.item()
        self.global_count += self.count.item()
        self.value[0] = 0.0
        self.count[0] = 0

    def update(self, value, count=1) -> None:
        r'''
        update local value and count
        '''
        self.value[0] += value * count
        self.count[0] += count

    def reset(self) -> None:
        self.value[0] = 0.0
        self.count[0] = 0
        self.global_value = 0.0
        self.global_count = 0

    @property
    def ave(self):
        if self.global_count == 0:
            return nan
        return self.global_value / self.global_count


class RecordDict:
    def __init__(self) -> None:
        self.__inner_dict = dict()

    def __init__(self, dic: dict, test: bool = False) -> None:
        self.__inner_dict = dict()
        self.test = test
        for key in dic.keys():
            self.__inner_dict[key] = Record(test)

    def __getitem__(self, key) -> Record:
        return self.__inner_dict[key]

    def __setitem__(self, key, value) -> None:
        assert (isinstance(value, Record))
        self.__inner_dict[key] = value

    def __str__(self) -> str:
        s = []
        for key, value in self.__inner_dict.items():
            s.append('{key}:{value}'.format(key=key, value=value.ave))
        return ', '.join(s)

    def sync(self):
        for value in self.__inner_dict.values():
            value.sync()

    def reset(self):
        for value in self.__inner_dict.values():
            value.reset()

    def add_record(self, key):
        self.__inner_dict[key] = Record(self.test)


class Timer:
    def __init__(self, timer_name, logger):
        self.timer_name = timer_name
        self.logger = logger

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start  # seconds
        self.logger.debug('{} spent: {}.'.format(
            self.timer_name, str(datetime.timedelta(seconds=int(self.interval)))))


class GlobalTimer:
    def __init__(self, timer_name, container):
        self.timer_name = timer_name
        self.container = container

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start  # seconds
        self.container[0] += self.interval


class CriterionWarpper(nn.Module):
    def __init__(self, criterion, TET=False, TET_phi=1.0, TET_lambda=0.0) -> None:
        super().__init__()
        self.criterion = criterion
        self.TET = TET
        self.TET_phi = TET_phi
        self.TET_lambda = TET_lambda
        self.mse = nn.MSELoss()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        if self.TET:
            loss = 0
            for t in range(output.shape[0]):
                loss = loss + (1. - self.TET_lambda) * self.criterion(output[t], target)
            loss = loss / output.shape[0]
            if self.TET_lambda != 0:
                loss = loss + self.TET_lambda * self.mse(
                    output,
                    torch.zeros_like(output).fill_(self.TET_phi))
            return loss
        else:
            return self.criterion(output.mean(0), target)


def unpack_len1_tuple(x: tuple or torch.Tensor):
    if isinstance(x, tuple) and x.__len__() == 1:
        return x[0]
    else:
        return x


def snn_to_ann_input(x: torch.Tensor):
    if len(x.shape) == 5:
        return x.flatten(0, 1)
    else:
        return x


class BaseMonitor:
    def __init__(self):
        self.hooks = []
        self.monitored_layers = []
        self.records = []
        self.name_records_index = {}
        self._enable = True

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.records[i]
        elif isinstance(i, str):
            y = []
            for index in self.name_records_index[i]:
                y.append(self.records[index])
            return y
        else:
            raise ValueError(i)

    def clear_recorded_data(self):
        self.records.clear()
        for k, v in self.name_records_index.items():
            v.clear()

    def enable(self):
        self._enable = True

    def disable(self):
        self._enable = False

    def is_enable(self):
        return self._enable

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def __del__(self):
        self.remove_hooks()


class SOPMonitor(BaseMonitor):
    def __init__(self, net: nn.Module):
        super().__init__()
        for name, m in net.named_modules():
            if name in net.skip:
                continue
            if isinstance(m, ConvBlock):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                # conv.weight [C_out, C_in, H_k, W_k]
                connects = torch.ones_like(m.conv.weight)
                if m.sparse_neurons:
                    mask = (m.neuron_mask.mask_value > 0).float().squeeze(0)
                else:
                    mask = None
                self.hooks.append(m.register_forward_hook(self.create_hook(name, connects, mask)))

    def cal_sop(self, x: Tensor, connects: Tensor, mask: Tensor, m: nn.Conv2d, pool):
        out = torch.nn.functional.conv2d(x, connects, None, m.stride, m.padding, m.dilation,
                                         m.groups)
        if pool is not None:
            out = pool(out)
        if mask is None:
            sop = out.sum()
        else:
            sop = (out * mask).sum()
        return sop.unsqueeze(0)

    def create_hook(self, name, connects, mask):
        def hook(m: ConvBlock, x: Tensor, y: Tensor):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(
                    self.cal_sop(
                        snn_to_ann_input(unpack_len1_tuple(x)).detach(), connects, mask, m.conv, m.pool))

        return hook

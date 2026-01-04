import math
import os

import torch
import torch.distributed as dist
from typing import Dict, Any

# 增强随机性
def get_ddp_generate(seed=100):
    local_rank=dist.get_rank()
    g =torch.Generator()
    g.manual_seed(seed+local_rank)
    return g

# 初始化环境参数

def init_distributed_mode(local_rank,args):

    os.environ["MASTER_ADDR"] = args.MASTER_ADDR
    os.environ["MASTER_PORT"] = args.MASTER_PORT

    args.rank = local_rank
    args.world_size = len(args.gpu)

    torch.cuda.set_device(args.gpu[local_rank])

    dist.init_process_group(backend='nccl', init_method='env://',world_size=args.world_size, rank=args.rank)

    # dist.init_process_group(backend='nccl',init_method=args.dist_url, rank=local_rank,  world_size=len(args.gpu))

    if  local_rank==0:
        print(f'begin validating')


    # # 如果是多机多卡的机器，WORLD_SIZE代表使用的机器数，RANK对应第几台机器，LOCAL_RANK代表当前机器上第几块GPU
    # # 如果是单机多卡的机器，WORLD_SIZE代表有几块GPU，RANK、LOCAL_RANK代表第几块GPU
    # # 在指令命令时--use_env 会传入这三个参数
    # if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    #     args.rank = int(os.environ["RANK"])
    #     args.world_size = int(os.environ['WORLD_SIZE'])
    #     args.gpu = int(os.environ['LOCAL_RANK'])
    #
    # elif 'SLURM_PROCID' in os.environ:
    #     args.rank = int(os.environ['SLURM_PROCID'])
    #     args.gpu = args.rank % torch.cuda.device_count()
    # else:
    #     print('Not using distributed mode')
    #     args.distributed = False
    #     return
    # args.distributed = True
    # torch.cuda.set_device(args.gpu)  # 当前进程指定GPU
    #
    # args.dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL
    #
    print('| distributed init (rank {}): {} {}'.format(
        local_rank, args.MASTER_ADDR,args.MASTER_PORT), flush=True)


    # # 创建进程组
    #
    # # backend='nccl'|'gloo'
    # # nccl：nccl 是 NVIDIA 提供的专门用于 GPU 之间高性能通信的库，适用于使用 NVIDIA GPU 的分布式训练。它充分利用了 NVIDIA GPU 的高性能计算能力
    # # 可以在多个 GPU 之间高效地传输大量数据，适合在大规模 GPU 集群中进行分布式训练。nccl 后端通常在单节点多 GPU 训练中表现较好 推荐使用
    #
    # # init_method
    # # 当参数为localhsot即是以一机多卡，当参数为IP地址为，即多机多卡，IP地址为通信的机器地址
    # # 后面的端口找一个空着没用的就行
    #
    # # rank
    # # 如果是多机多卡的机器，WORLD_SIZE代表使用的机器数，RANK对应第几台机器
    # # 如果是单机多卡的机器，WORLD_SIZE代表有几块GPU，RANK代表第几块GPU
    # # world_size
    # # 为GPU的数量
    #
    # dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                         world_size=args.world_size, rank=args.rank)
    #
    # dist.barrier()  # 等待每一块GPU走到这里之后再往下走

# 多GPU计算损失
def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value
    rt =value.clone()
    with torch.no_grad():
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)  # 对不同设备的value求和
        if average:
            rt /= world_size  # 得到多设备之间的均值
        return rt


def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0


class Scheduler:
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 param_group_field: str,
                 noise_range_t=None,
                 noise_type='normal',
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=None,
                 initialize: bool = True) -> None:
        self.optimizer = optimizer
        self.param_group_field = param_group_field
        self._initial_param_group_field = f"initial_{param_group_field}"
        if initialize:
            for i, group in enumerate(self.optimizer.param_groups):
                if param_group_field not in group:
                    raise KeyError(f"{param_group_field} missing from param_groups[{i}]")
                group.setdefault(self._initial_param_group_field, group[param_group_field])
        else:
            for i, group in enumerate(self.optimizer.param_groups):
                if self._initial_param_group_field not in group:
                    raise KeyError(f"{self._initial_param_group_field} missing from param_groups[{i}]")
        self.base_values = [group[self._initial_param_group_field] for group in self.optimizer.param_groups]
        self.metric = None  # any point to having this for all?
        self.noise_range_t = noise_range_t
        self.noise_pct = noise_pct
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.noise_seed = noise_seed if noise_seed is not None else 42
        self.update_groups(self.base_values)
        self.epoch = -1

    def state_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)

    def get_epoch_values(self, epoch: int):
        return None

    def get_update_values(self, num_updates: int):
        return None

    def get_last_lr(self):
        return self.get_epoch_values(self.epoch+1)

    def step(self, epoch: int, metric: float = None) -> None:
        self.metric = metric
        self.epoch = epoch
        values = self.get_epoch_values(epoch)
        if values is not None:
            values = self._add_noise(values, epoch)
            self.update_groups(values)

    def step_update(self, num_updates: int, metric: float = None):
        self.metric = metric
        values = self.get_update_values(num_updates)
        if values is not None:
            values = self._add_noise(values, num_updates)
            self.update_groups(values)

    def update_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group[self.param_group_field] = value

    def _add_noise(self, lrs, t):
        if self.noise_range_t is not None:
            if isinstance(self.noise_range_t, (list, tuple)):
                apply_noise = self.noise_range_t[0] <= t < self.noise_range_t[1]
            else:
                apply_noise = t >= self.noise_range_t
            if apply_noise:
                g = torch.Generator()
                g.manual_seed(self.noise_seed + t)
                if self.noise_type == 'normal':
                    while True:
                        # resample if noise out of percent limit, brute force but shouldn't spin much
                        noise = torch.randn(1, generator=g).item()
                        if abs(noise) < self.noise_pct:
                            break
                else:
                    noise = 2 * (torch.rand(1, generator=g).item() - 0.5) * self.noise_pct
                lrs = [v + v * noise for v in lrs]
        return lrs
# 这个学习率调度器根据设定的参数，在训练过程中动态地调整学习率。通过调用get_epoch_values或get_update_values方法，
# 可以获得当前周期或步数对应的学习率。这个调度器还支持学习率预热和噪声的添加，以增加训练的稳定性和泛化性能。
class CosineLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int = 200, # 初始学习率调整周期的长度（以步数或周期数表示）。
                 lr_min: float = 1.0e-4,
                 cycle_mul: float = 1., # 初始学习率调整周期的长度（以步数或周期数表示）。
                 cycle_decay: float = 0.1, # 学习率每个周期的衰减因子
                 cycle_limit: int = 1, # 学习率调整的周期数限制
                 warmup_t=0, # 学习率预热的步数或周期数
                 warmup_lr_init=1.0e-6, # 预热阶段的初始学习率。
                 warmup_prefix=False, # 是否在预热阶段之前进行周期计数。
                 t_in_epochs=True,  #  True表示周期数用于计算学习率，False表示步数用于计算学习率。
                 noise_range_t=None, # 噪声幅度的步数范围。
                 noise_pct=0.67,  # 噪声幅度的步数范围。
                 noise_std=1.0, #  噪声的标准差。
                 noise_seed=42, #  噪声的标准差。
                 k_decay=1.0,  #  余弦函数中的指数衰减因子
                 initialize=True) -> None:  # 是否在初始化时更新优化器的学习率。
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        # assert t_initial > 0
        assert lr_min >= 0
        self.t_initial = t_initial
        self.lr_min = lr_min
        self.cycle_mul = cycle_mul
        self.cycle_decay = cycle_decay
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.t_in_epochs = t_in_epochs
        self.k_decay = k_decay
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]
    # 根据当前的步数或周期数，计算学习率。
    # 在预热阶段，学习率线性地从初始值增加到目标值。
    # 在学习率调整周期中，根据余弦函数调整学习率，其中学习率的最大值随着周期数的增加而衰减。
    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.cycle_mul != 1:
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.cycle_mul), self.cycle_mul))
                t_i = self.cycle_mul ** i * self.t_initial
                t_curr = t - (1 - self.cycle_mul ** i) / (1 - self.cycle_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            gamma = self.cycle_decay ** i
            lr_max_values = [v * gamma for v in self.base_values]
            k = self.k_decay

            if i < self.cycle_limit:
                lrs = [
                    self.lr_min + 0.5 * (lr_max - self.lr_min) * (1 + math.cos(math.pi * t_curr ** k / t_i ** k))
                    for lr_max in lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_values]

        return lrs
    # 如果t_in_epochs为True，则返回给定周期的学习率。
    # 否则，返回None。
    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None
    # 如果t_in_epochs为False，则返回给定步数的学习率。
    # 否则，返回None。
    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None
    # get_cycle_length方法：
    # 计算指定周期数的学习率调整周期的总长度。
    def get_cycle_length(self, cycles=0):
        cycles = max(1, cycles or self.cycle_limit)
        if self.cycle_mul == 1.0:
            return self.t_initial * cycles
        else:
            return int(math.floor(-self.t_initial * (self.cycle_mul ** cycles - 1) / (1 - self.cycle_mul)))

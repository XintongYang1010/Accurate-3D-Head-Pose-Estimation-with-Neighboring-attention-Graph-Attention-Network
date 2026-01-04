import os
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import torch.multiprocessing as mp
from utils.facepose.train_eval_utils import train_epoch,test
from utils.log import *
from utils.init import init_distributed_mode,get_ddp_generate
from utils.data_load import FaceLandmarkData
from utils.utils import *

from config.config import args

from models.pointNet.pointnet import PtModel,PtLoss_euler

#  inplace 参数被设置为 True ，节省内存，inplace 操作可能会对梯度计算和模型的训练产生影响
def inplace_relu(m):
    # 首先用self.__class__将实例变量指向类，然后再去调用__name__类属性
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def train(local_rank,args):

    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    # 初始化各进程环境
    # local_rank = args.gpu[local_rank]

    init_distributed_mode(local_rank,args=args)

    '''DATA LOADING'''
    if local_rank ==0:
        print("Load dataset ...")
    # print(torch.cuda.current_device())
    # Dataset Random partition
    FaceLandmark = FaceLandmarkData(args,partition='trainval')
    train_size = int(len(FaceLandmark) * 0.7)

    test_size = len(FaceLandmark) - train_size
    torch.manual_seed(args.seed)

    # Prepare the dateset and dataloader
    train_dataset, test_dataset = torch.utils.data.random_split(FaceLandmark, [train_size, test_size])

    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,shuffle=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset,shuffle=True)

    # 将样本索引每batch_size个元素组成一个list
    # 测试集不需要设置
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, args.batch_size, drop_last=True)

    # 设置主进程进程下的子进程数，能够加速数据的加载，不会影响模型的训练进程
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])

    if local_rank == 0:
        print('Using {} dataloader workers every process'.format(nw))

    g = get_ddp_generate()
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=nw,
                                               shuffle=False,
                                               generator=g
                                               )

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             sampler=test_sampler,
                                             pin_memory=True,
                                             num_workers=nw,
                                             # collate_fn=val_data_set.collate_fn
                                              )
    # 实例化模型
    if args.model_name=='pointNet':
        model = PtModel(args).cuda()
        criterion = PtLoss_euler(args.Lamda)

    # 转为DDP模型,就可以在各个设备上通信
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) # BN 层同步

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu[local_rank]],find_unused_parameters=True)
    model.apply(inplace_relu)
    model.apply(weight_init)

    scaler = torch.cuda.amp.GradScaler()

    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]

    # 优化器是用来更新神经网络模型的权重（参数）以最小化损失函数的工具
    if args.optim == "sgd":
        optimizer = torch.optim.SGD(pg, lr=args.lr, momentum=0.9)

    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(pg, lr=args.lr, eps=1e-4)

    elif args.optim == "adam":
        optimizer = torch.optim.Adam(pg, lr=args.lr, eps=1e-4)

    # 习率调度器是用来调整训练过程中学习率的工具。
    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0001)
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=40, gamma=0.9)

    best_instance_acc = 999999

    '''TRANING'''
    if local_rank==0:
        exp_dir, logger, time, model_file = log(args,local_rank)            # 初始化日志
        save_model_file(args, model_file)

        print(f'{args.model_name} {args.gpu[local_rank]} Start training...')

    for epoch in range(args.epoch):

        train_sampler.set_epoch(epoch)

        train_mean_loss,train_euler_value,train_Deviation_value = train_epoch(args=args, model=model,
                                                        start_epoch=0,
                                                        criterion=criterion,
                                                        optimizer=optimizer,
                                                        data_loader=train_loader,device=device, scaler=scaler
                                                                                   ,epoch=epoch,gpu=args.gpu[local_rank])
        test_mean_loss,test_euler_value,test_Deviation_value = test(model=model,criterion=criterion,epoch=epoch,
                           data_loader=test_loader,
                           device=device,args=args)

        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if optimizer.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if optimizer.param_groups[0]['lr'] < 1e-5:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-5

        if local_rank == 0:

            # if bool:
            #     config = dict(
            #         learning_rate=args.lr,
            #         architecture=args.model_name,
            #         batch_size=args.batch_size,
            #         epoch=args.epoch,
            #         num_catecategory=args.num_category
            #     )
            #
            #     wandb.init(
            #         project="Point_faceland_train",
            #         name=time + args.model_name,
            #         notes="",
            #         config=config,
            #     )
            #     bool =False

            logger.info("[epoch {}]  train_mean_loss:{} {}  train:{}  test:{}".format(
                epoch, round(train_mean_loss, 3),train_euler_value,train_Deviation_value,test_euler_value,test_Deviation_value))

            print("[epoch {}]  train_mean_loss: {}  train:{} {}  test:{} {}".format(
                epoch, round(train_mean_loss, 3),train_euler_value,train_Deviation_value,test_euler_value,test_Deviation_value))

            if (test_euler_value <= best_instance_acc):
                best_instance_acc = test_euler_value
                best_epoch = epoch + 1

            if (test_euler_value <= best_instance_acc):
                savepath = str(exp_dir) + '/best_model.pth'
                logger.info('BestModel Saving at %s' % savepath)
                print('BestModel Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': test_euler_value,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            logger.info(f"conrrent_acc :{test_euler_value} best_acc :{best_instance_acc}")
            print( f"conrrent_acc :{test_euler_value} best_acc :{best_instance_acc}" )

    if local_rank==0:
        logger.info(args.other_info+' End of training...  ' )
        print('End of training...')
    dist.destroy_process_group()
    # wandb.finish()
    # 清理分布式训练环境



if __name__ == '__main__':

    mp.spawn(fn=train, args=(args,),  nprocs=len(args.gpu), join=True)
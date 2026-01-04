
from tqdm import tqdm
from utils.init import reduce_value, is_main_process
import  sys
from utils.facepose.angel import *
import torch
from torch.cuda.amp import GradScaler, autocast



scaler = GradScaler()
def train_epoch(args,model,criterion, start_epoch,optimizer, data_loader, device, epoch,scaler,gpu):
    train_mean_loss_list =[]
    train_difference_value_list=[]

    model =model.train()  # 开始训练

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout,smoothing=0.9)

    for index, (point, label ,heat_map,_) in enumerate(data_loader,start=start_epoch):


        heat_map = heat_map.to(device)  # seg: (Batch * point_num * landmark)
        label =label.to(device)
        point = point.permute(0, 2, 1).to(device)  # point: (Batch * num_point * num_dim)
        with autocast():
            pred,trans_feat = model(point)  # PAConv B 69 3   pt  B 207
            loss = criterion(pred,label, args, trans_feat).cuda()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        euler,Deviation_value=cal_euler_angle(args,pred, label,partition="train")

        train_mean_loss = reduce_value(loss.detach(), average=True)
        train_mean_loss_list.append(train_mean_loss)

        train_mean_difference_value = reduce_value(euler, average=True)
        train_difference_value_list.append(train_mean_difference_value)


        if is_main_process():
            data_loader.desc = "[{}  GPU: {} epoch {} ]  train_loss {}  {}".format(args.model_name,gpu,epoch,round(train_mean_loss.item(), 3), Deviation_value)

    train_mean_loss = torch.mean(torch.tensor(train_mean_loss_list)).cuda()
    train_mean_difference_value = torch.mean(torch.tensor(train_difference_value_list)).cuda()

    if not torch.isfinite(loss):
        print('WARNING: non-finite loss, ending training ', loss)
        sys.exit(1)


    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return train_mean_loss.item(),train_mean_difference_value,Deviation_value


@torch.no_grad()
def test(model,criterion, data_loader, device,args,epoch):
    classifier = model.eval()

    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    test_mean_loss_list =[]
    test_difference_value_list=[]
    with torch.no_grad():
        with torch.cuda.amp.autocast():

            for index, (point, label,heat_map,_) in enumerate(data_loader):

                point = point.to(device)  # point: (Batch * num_point * num_dim)

                label = label.to(device)  # landmark : (Batch * landmark * num_dim)

                heat_map = heat_map.to(device)  # seg: (Batch * point_num * landmark)
                point_normal = point.permute(0, 2, 1)  # point_normal : (batch * num_dim * num_point)


                pred,trans_feat = classifier(point_normal.contiguous())
                loss = criterion(pred, label, args,trans_feat)

        distence,Deviation_value=cal_euler_angle(args,pred, label)

        test_mean_loss = reduce_value(loss.detach(), average=True)
        test_mean_loss_list.append(test_mean_loss)

        test_mean_difference_value = reduce_value(distence, average=True)
        test_difference_value_list.append(test_mean_difference_value)

        if is_main_process():
            data_loader.desc = "[epoch {}]  test_loss {} {} ".format(epoch, round(test_mean_loss.item(), 3),Deviation_value)


    test_mean_loss = torch.mean(torch.tensor(test_mean_loss_list)).cuda()
    test_mean_difference_value = torch.mean(torch.tensor(test_difference_value_list)).cuda()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return test_mean_loss.item(),test_mean_difference_value,Deviation_value

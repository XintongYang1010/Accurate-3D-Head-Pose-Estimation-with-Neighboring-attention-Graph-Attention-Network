import os
from utils.utils import *
from models.pointNet.pointnet import PtModel
from tqdm import tqdm
from utils.data_load import FaceLandmarkData
from torch.utils.data import DataLoader, Subset
import torch
from utils.facepose.train_eval_utils import cal_euler_angle
from config.config import args
from utils.facepose.angel import *
def test(classifier, loader, args):

    path = 'result/' + args.model_name
    result_txt = os.path.join(path, args.log_name + '.txt')
    result_mean_txt = os.path.join(path,args.log_name+'_'+str(args.test_batch_size)+'.txt')
    os.makedirs(path,exist_ok=True)

    info_list=[]
    # 平均值
    mean_list=[]
    for epoch, (points, label ,heat_map,file_name) in tqdm(enumerate(loader), total=len(loader)):

        points, label = points.cuda(), label.cuda()
        points = points.transpose(2, 1).to(device)

        pred = classifier(points[:,0:3,:])[0]
        euler, Deviation_value,pitch_all,yaw_all,roll_all,euler_all = cal_euler_angle(args,pred, label,partition='test')
        mean_info = f"euler: {euler} ,Deviation_value: {Deviation_value} "
        mean_list.append(mean_info)

        for i in range(args.test_batch_size):

            info =  f'{file_name[i]} eluer: {euler_all[i]} pitch: {pitch_all[i]} yaw:{yaw_all[i]} roll: {roll_all[i]}'\
                    f' truth: {label[i][0] } {label[i][1] } {label[i][2]} pred: {pred[i][0]} {pred[i][1]} {pred[i][2]} '
            # print(info)
            info_list.append(info)

    # 按照欧氏距离排序
    sorted_info_list =  sorted(info_list, key=lambda x: float(x.split('eluer: ')[1].split()[0]))
    save(sorted_info_list,result_txt)

    #计算总体的均值
    size,avg_euler,avg_pitch,avg_yaw,avg_roll = cal_angel_all_mean(mean_list)

    info = f"the dataset of test is {size} mean_euler: {avg_euler} " \
           f"mean_pitch: {avg_pitch} mean_yaw: {avg_yaw} mean_roll: {avg_roll}"

    # 按照每个字符串中的 euler 值进行排序
    sorted_mean_list = sorted(mean_list, key=lambda x: float(x.split()[1]))
    # 添加最后的均值
    sorted_mean_list.append(info)

    get_perpoints(args,result_txt)

if __name__ == '__main__':

    root = args.root+args.dataset_name+"/"

    args.log_path = os.path.join('/home/hjc/Program/3D_face/pose/','log',  args.model_name,args.log_name)
    torch.cuda.set_device(7)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # Dataset Random partition
    FaceLandmark = FaceLandmarkData(args,partition='trainval')

    # 计算需要加载的数据的数量
    data_number = int(len(FaceLandmark) *1)

    subset = Subset(FaceLandmark, range(data_number))

    testDataLoader = torch.utils.data.DataLoader(subset, batch_size=args.test_batch_size, shuffle=False)
    '''MODEL LOADING'''

    if args.model_name == 'pointNet':
        classifier = PtModel(args).to(device)

    state_dict = torch.load(args.log_path + '/best_model.pth', map_location=torch.device('cpu'))

    classifier.load_state_dict(state_dict['model_state_dict'])

    with torch.no_grad():
        test(classifier.eval(), testDataLoader,args=args)











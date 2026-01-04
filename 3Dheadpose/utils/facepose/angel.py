

import numpy as np
import os
import math
from utils.utils import *

def cal_angel_all_mean(list):
    '''
    计算所有的数据的角度均值
    Args:
        list:

    Returns:

    '''
    # 初始化累加变量
    total_euler = 0.0
    total_pitch = 0.0
    total_yaw = 0.0
    total_roll = 0.0
    num_elements = len(list)

    # 提取值并累加
    for item in list:
        # 提取 euler 值
        euler = float(item.split('euler:')[1].split(',')[0].strip())

        # 提取 pitch, yaw, roll 值
        deviation_values = item.split('Deviation_value:')[1].strip()
        pitch = float(deviation_values.split('pitch:')[1].split()[0].strip())
        yaw = float(deviation_values.split('yaw:')[1].split()[0].strip())
        roll = float(deviation_values.split('roll:')[1].strip())

        total_euler += euler
        total_pitch += pitch
        total_yaw += yaw
        total_roll += roll

    # 计算平均值
    avg_euler = total_euler / num_elements
    avg_pitch = total_pitch / num_elements
    avg_yaw = total_yaw / num_elements
    avg_roll = total_roll / num_elements
    return  len(list),avg_euler,avg_pitch,avg_yaw,avg_roll

def cal_euler_angle(args,pred,label,partition="train"):
    B = pred.shape[0 ]
    euler_all = pitch_all = yaw_all = roll_all = 0

    euler_list =[]
    pitch_list = []
    yaw_list = []
    roll_list =[]

    for i in range(B):
        pitch = abs(pred[i][0] - label[i][0])
        yaw = abs(pred[i][1] - label[i][1])
        roll = abs(pred[i][2] - label[i][2])
        euler = math.sqrt(

            (pred[i][0] - label[i][0]) ** 2 + (pred[i][1] - label[i][1]) ** 2 + (pred[i][2] - label[i][2]) ** 2)

        if partition=="test":
            pitch_list.append(pitch)
            roll_list.append(roll)
            yaw_list.append(yaw)
            euler_list.append(euler)

        pitch_all =pitch_all+pitch
        yaw_all =yaw_all+yaw
        roll_all = roll_all + roll
        euler_all = euler_all +euler

    # batch的mean
    pitch_mean = pitch_all / B
    yaw_mean = yaw_all / B
    roll_mean = roll_all / B
    euler_mean = euler_all / B
    Deviation_value = f'pitch: {pitch_mean} yaw: {yaw_mean} roll: {roll_mean}'
    if partition =="train":
        return  euler_mean,Deviation_value

    else:
        return euler_mean,Deviation_value,pitch_list,yaw_list,roll_list,euler_list


# 随机生成欧拉角
def generate_euler_angles():
    pitch = np.random.randint(-60, 45)  # 仰俯角
    yaw = np.random.randint(-60, 60)  # 偏航角
    roll = np.random.randint(-20, 20)  # 滚转角
    return pitch, yaw, roll

# 旋转点云
def rotate_point_cloud(points, pitch, yaw, roll):
    # 将角度转换为弧度
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    roll = np.radians(roll)

    # 构造旋转矩阵
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(pitch), -np.sin(pitch)],
                    [0, np.sin(pitch), np.cos(pitch)]])

    R_y = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                    [0, 1, 0],
                    [-np.sin(yaw), 0, np.cos(yaw)]])

    R_z = np.array([[np.cos(roll), -np.sin(roll), 0],
                    [np.sin(roll), np.cos(roll), 0],
                    [0, 0, 1]])

    # 组合旋转矩阵
    R = np.dot(R_z, np.dot(R_y, R_x))

    # 旋转点云
    rotated_points = np.dot(points, R.T)
    return rotated_points


# 读取和保存点云文件
def cal_angel(args,input_folder, output_folder_angel, output_folder_label,decimal):
    # 确保输出文件夹存在
    os.makedirs(output_folder_angel, exist_ok=True)
    os.makedirs(output_folder_label, exist_ok=True)
    rotat_file_path = os.path.join(args.root, 'dataset',args.dataset_name, 'rotat.txt')
    with open(rotat_file_path,'a+') as rotat_file:

        # 752 1504 2256
        for root, dirs, files in os.walk(input_folder):
            for file_name in files:
                if file_name.endswith('.asc'):
                    name=file_name.split('.')[0]
                    newname=int(name)  + 0
                    input_file_path = os.path.join(root, file_name)

                    points = np.loadtxt(input_file_path)[:,0:3]

                    # 随机生成欧拉角
                    pitch, yaw, roll = generate_euler_angles()

                    # 旋转点云
                    rotated_points = rotate_point_cloud(points, pitch, yaw, roll)

                    # 保存旋转后的点云
                    output_file_path = os.path.join(output_folder_angel, str(newname)+'.asc')
                    np.savetxt(output_file_path, rotated_points, fmt=f'%{decimal}')

                    # 保存欧拉角
                    label_file_path = os.path.join(output_folder_label, str(newname)+'.asc')
                    with open(label_file_path, 'w') as f:
                        f.write(f"{pitch}\n{yaw}\n{roll}")
                    rotat_file.write(f"{name} {newname} {pitch} {yaw} {roll}")

def get_perpoints(args,result_txt):
    '''
    Args:
        result_txt: 获得点云的文件名  第一列
        ori_path: 原始点云
        rotat_path: 欧拉角
        output_path: 输出路径
        ground_truth: 真值

    Returns:
    '''
    result_path = os.path.join(args.root,"result",args.model_name,args.log_name)
    os.makedirs(result_path,exist_ok=True)

    with open(result_txt,"r") as read_file:
        for line in read_file:
            split = line.split()
            file_name = split[0]
            ori_file_name =  str(int(split[0].split('.')[0]) % 752)
            pitch = round(float(split[-3]),3)
            yaw = round(float(split[-2]),3)
            roll = round(float(split[-1]),3)


            ori_file = os.path.join(args.natural_face_path, ori_file_name+'.asc')
            pred_file = os.path.join(result_path, file_name.split('.')[0]+"_"+ori_file_name+'_pred.asc')
            ground_truth_file = os.path.join(args.rotat_face_path,file_name)

            points = np.loadtxt(ori_file)[:,0:3]
            pre_points = rotate_point_cloud(points,pitch,yaw,roll)

            save_pred(pre_points,pred_file)
            copy_and_rename_file(source_folder=ground_truth_file,destination_folder=result_path,
                                 new_filename=file_name.split('.')[0]+"_"+ori_file_name+"_truth."+file_name.split(".")[1])


input_folder = '/home/hjc/Program/3D_face/pose/dataset/mutil_expression/procession_Data/after_fps'
output_folder_angel = '/home/hjc/Program/3D_face/pose/dataset/mutil_expression/data_/angel3'
output_folder_label = '/home/hjc/Program/3D_face/pose/dataset/mutil_expression/data_/label3'



# load_and_save_point_clouds(input_folder, output_folder_angel, output_folder_label,decimal='0.3f')
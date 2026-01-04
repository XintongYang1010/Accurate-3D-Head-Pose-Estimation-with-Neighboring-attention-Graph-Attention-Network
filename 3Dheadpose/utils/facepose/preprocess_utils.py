from torch.utils.data import Dataset
from config.config import get_parser
from utils.augment_utils import *
from utils.facepose.angel import *
sys.path.append(os.path.dirname(sys.path[0]))


def save_asc_filenames_to_txt(folder_path, output_file):
    # 获取文件夹中所有 .asc 文件的文件名
    asc_files = [f for f in os.listdir(folder_path) if f.endswith('.asc')]

    # 将文件名保存到输出文件
    with open(output_file, 'w') as file:
        for asc_file in asc_files:
            file.write(f"{asc_file}\n")

# 规范化数字
def separate_numbers(file_path):
    import os
    import re
    with open(file_path, 'r') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        # 使用正则表达式匹配连接在一起的数，并在它们之间加入空格
        lines[i] = re.sub(r'(-?\d+\.\d+)(?=-?\d+\.\d+)', r'\1 ', lines[i])

    with open(file_path, 'w') as f:
        f.writelines(lines)

def to_numpy(args,save_path,sample_path,label_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("beginning to numpy and mat... ")
    # 读取文件夹中的所有 .txt 文件并存储数据
    def read_txt_files(folder_path):
        data_list = []
        txt_files = [f for f in os.listdir(folder_path) ]
        sorted_txt_files = sorted(txt_files)

        for txt_file in sorted_txt_files:
            txt_path = os.path.join(folder_path, txt_file)
            data = np.loadtxt(txt_path)
            data_list.append(data)
        return data_list, sorted_txt_files

    data_label, sorted_txt_files = read_txt_files(label_path)
    data_sample, sorted_txt_files = read_txt_files(sample_path)

    np.save(save_path + "/sample.npy", data_sample)
    np.save(save_path + "/label.npy", data_label)
    np.save(save_path+'/filename.npy',sorted_txt_files)

    if  args.model_name=='DHGCN':
        p2v_indices, part_distance = split_part(data_sample, args.split_num, args.max_distance)
        np.save(
            save_path+'/indices_splitnum_{}_md{}.npy'.format( args.split_num, args.max_distance),
            p2v_indices)
        np.save(save_path+'/distance_splitnum_{}_md{}.npy'.format(args.split_num,
                                                                              args.max_distance),
                part_distance)
    print('Split part done!')


    print("end to numpy and mat ...")

    with open(save_path + "name.txt", 'w') as data_file:
        for txt_file in sorted_txt_files:
            data_file.writelines(txt_file + "\n")

    if args.need_heatmap:

        print("beginning to heatmap ...")

        with h5py.File(save_path + "sample.mat", 'w') as mat_file:
            mat_file.create_dataset('point_all', data=data_sample)

        with h5py.File(save_path+ "landmark.mat", 'w') as mat_file:
            mat_file.create_dataset('landmark_all', data=data_label)
    print("end to heatmap ...")

class PointCoudProcessing(Dataset):
    def __init__(self, root,read_file,old_file_name,new_file_name,is_fps,is_normalize,num_point=3000,decimal_point='.6f'):
        self.root = root         # 数据集的根目录
        self.npoints =num_point  # 对原始数据集下采样至npoint个点
        self.is_fps=is_fps       # 是否下采样
        self.new_file_name =new_file_name
        self.old_file_name =old_file_name
        self.is_normalize =is_normalize   # 是否归一化
        self.decimal_point=decimal_point
        if not os.path.exists(new_file_name):
            os.makedirs(new_file_name)

        shape_ids = [line.rstrip() for line in open(read_file)]

        shape_names = ['_'.join(x.split('.')[0:-1]) for x in shape_ids]

        self.sample_path = [
            (shape_names[i], os.path.join(self.old_file_name, shape_ids[i])) for i
            in range(len(shape_ids))]

    def __len__(self):
        return len(self.sample_path)

    def _get_item(self, index):

        # 加载label文件
        fn = self.sample_path[index]

        point_set = torch.from_numpy(np.genfromtxt(fn[1] , delimiter=' ').astype(np.float32))
        # 点云下采样
        if self.is_fps:
            point_set = farthest_point_sample(point_set, self.npoints)

        # 是否归一化
        if self.is_normalize:
            point_set,m,centroid = pc_normalize(point_set)

            m_normalize_save(m, centroid,self.root + "m_centroid.txt"
                              , fn[0],self.decimal_point)

        #将点云下采样归一化的数据存储
        point_save(point_set,self.new_file_name +"/"+fn[0] + ".asc",self.decimal_point)
        # 返回点云和对应的标签
        return point_set

    def __getitem__(self, index):
        return self._get_item(index)

def cal_centroid(args,sample_path,new_sample):

    def compute_centroid(point_cloud):
        # 计算点云的质心
        centroid = np.mean(point_cloud, axis=0)
        return centroid

    def translate_point_cloud(point_cloud, translation_vector):
        # 将点云沿着平移向量平移
        translated_point_cloud = point_cloud + translation_vector
        return translated_point_cloud

    if not os.path.exists(new_sample):
        os.makedirs(new_sample)

    file_list = [file for file in os.listdir(sample_path) if file.endswith('.asc')]

    # 处理第一个文件
    first_file_path = os.path.join(sample_path, file_list[0])
    first_points = np.loadtxt(first_file_path)[:, 0:3]
    centroid = compute_centroid(first_points)

    for file_name in file_list:
        file_path = os.path.join(sample_path, file_name)

        # 读取点云数据
        sample = np.loadtxt(file_path)[:,:]
        points = sample[:, :3]
        info = sample[:,3:]
        # 平移到第一个质心的位置
        centroid_other = compute_centroid(points)
        translation_vector = centroid - centroid_other
        # 进行平移
        translated_point_cloud2 = translate_point_cloud(points, translation_vector)

        # 保存平移后的点云数据
        output_file_path = os.path.join(new_sample, file_name)
        result = np.concatenate((translated_point_cloud2, info), axis=1)
        np.savetxt(output_file_path, result, fmt="%"+args.decimal)
        print("finish"+ output_file_path+"calculate centroid ")
    print("处理完成！")

def FPS(args,root,read_file,old_file_name,new_file_name,num_point):
    is_fps = True
    is_normalize =False
    data = PointCoudProcessing(root,read_file,old_file_name,new_file_name,is_fps,is_normalize,num_point=num_point,decimal_point=args.decimal)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=48, shuffle=False)
    for point in DataLoader:
        print("!")

def Normalize(args,root,read_file,old_file_name,new_file_name):
    is_fps = False
    is_normalize = True
    data = PointCoudProcessing(root,read_file, old_file_name, new_file_name, is_fps, is_normalize,decimal_point=args.decimal)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=1240, shuffle=False)
    for point in DataLoader:
        print("!")

def delete_lastline(folder_path ):

    # 遍历文件夹下的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.asc'):  # 只处理txt文件
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # 删除最后一行的空格
            lines[-1] = lines[-1].rstrip()

            # 将修改后的内容写回文件
            with open(file_path, 'w') as file:
                file.writelines(lines)

def preprocess(args):
    root= '/home/hjc/Program/3D_face/pose/dataset/mutil_expression/test_data'
    # root = os.path.join(args.root,args.dataset_name)
    input_folder = os.path.join(root,'sample')
    output_folder_angel = os.path.join(root,"angel")
    output_folder_label = os.path.join(root,"label")

    file_name = root + "/name.txt"

    save_asc_filenames_to_txt(input_folder,file_name)

    if args.need_cal_centroid:
        print("Start calculate centroid...")
        after_cal_centroid_path = root + "/procession_Data/after_cal_centroid"
        cal_centroid(args, input_folder, after_cal_centroid_path)

    print("---------------------Start normalize---------------------")

    after_normalized_path = root + "/procession_Data/after_normalized"
    # 会因为这句报错，注释之后继续执行
    try:
        Normalize(args, root,file_name,after_cal_centroid_path,  after_normalized_path)

    except Exception as e:
        print(f"Error while Normalizing")

    # for filename in os.listdir(after_normalized_path ):
    #     if filename.endswith(".asc"):
    #         file_path = os.path.join(after_normalized_path , filename)
    #         separate_numbers(file_path)

    print("---------------------Start fps---------------------")

    after_fps_path = root + "/procession_Data/after_fps"
    FPS(args,root,file_name,after_normalized_path,after_fps_path,num_point=args.num_points)
    #
    # for filename in os.listdir(after_fps_path ):
    #     if filename.endswith(".asc"):
    #         file_path = os.path.join(after_fps_path , filename)
    #         separate_numbers(file_path)

    # delete_lastline(after_fps_path)

    cal_angel(args,after_fps_path, output_folder_angel, output_folder_label,args.decimal)

    # to_numpy(args,root,output_folder_angel ,output_folder_label)
    to_numpy(args, root,  after_fps_path , output_folder_label)
    print("dataset process success!")


if __name__=="__main__":
    args = get_parser()
    # collect_cie_file_paths( '/home/hjc/Program/3D_face/pose/dataset/mutil_expression/procession_Data/after_cal_centroid/','/home/hjc/Program/3D_face/pose/dataset/mutil_expression/name.txt')

    preprocess(args)

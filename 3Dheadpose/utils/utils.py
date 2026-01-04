import torch
import os
import shutil
import numpy as np
from sklearn.decomposition import PCA


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)

def process_txt_files(input_folder, output_folder):
    """
    Process all .txt files in a folder, keeping only the first three columns of each file.

    :param input_folder: str, path to the folder containing input .txt files
    :param output_folder: str, path to save the processed files
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(input_folder, file_name)
            try:
                # Load the data
                data = np.loadtxt(file_path)

                # Keep only the first three columns
                if data.shape[1] >= 3:  # Ensure the file has at least 3 columns
                    processed_data = data[:, :3]
                else:
                    print(f"Skipping {file_name}: Less than 3 columns.")
                    continue

                # Save the processed data
                output_file_path = os.path.join(output_folder, file_name)
                np.savetxt(output_file_path, processed_data, fmt="%.6f")
                print(f"Processed and saved: {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")


# RPS
def random_point_sample(points,point_nums,label_file,repetition):

    sampled_points = []
    exclusion_set = set()

    with open(label_file, 'r') as file:
        lines = file.readlines()

    points_list = []
    for line in lines:
        coordinates = line.strip().split()[0:3]  # 假设点坐标之间以空格分隔
        if len(coordinates) == 3:
            x, y, z = map(float, coordinates)
            points_list.append([x, y, z])

    # 将列表转换为 NumPy 数组
    label = np.array(points_list)

    while len(sampled_points) < point_nums:
        # 随机采样一个点
        random_index = np.random.randint(0, len(points))
        random_point = points[random_index]


        # # 如果采样点在 exclusion_set 中或者在 numpy1 中，则重新采样
        if repetition:
            if tuple(random_point) in exclusion_set or  ( points[random_index, 0:3] == label[:,0:3]).all(axis=1).any():
                continue
        else:
            if tuple(random_point) in exclusion_set:
                continue
        sampled_points.append(random_point)
        exclusion_set.add(tuple(random_point))

    return np.array(sampled_points)

# FPS
def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    point = point.to(device)
    N, D = point.shape  # N是数量 D是维数
    xyz =point[:, :3].to(device)

    # 先随机初始化一个centroids矩阵，
    # 后面用于存储npoint个采样点的索引位置
    centroids = torch.zeros((npoint,)).to(device)
    # 利用distance矩阵记录某个样本中所有点到某一个点的距离
    distance = torch.ones((N,)).to(device) * 1e10  # 初值给个比较大的值，后面会迭代更新
    # 利用farthest表示当前最远的点，也是随机初始化，范围为0~N
    #  = torch.tensor(np.random.randint(0, N))
    farthest =torch.randint(0,N,(1,))[0].to(device)
    # 直到采样点达到npoint，否则进行如下迭代

    for i in range(npoint):
        # 设当前的采样点centroids为当前的最远点farthest；
        centroids[i] = farthest.to(device)
        # 取出这个中心点centroid的坐标
        centroid =xyz[farthest, :].to(device)

        # 求出所有点到这个farthest点的欧式距离，存在dist矩阵中
        dist =torch.sum((xyz - centroid) ** 2,-1).to(device)
        # 建立一个mask，如果dist中的元素小于distance矩阵中保存的距离值，
        # 则更新distance中的对应值，
        # 即记录某个样本中每个点距离所有已出现的采样点的最小距离
        mask = dist < distance

        # 在这里添加检查采样点是否与a.txt中的点相同的条件

        distance[mask] = dist[mask]

        # 最后从distance矩阵取出最远的点为farthest，继续下一轮迭代
        farthest = torch.argmax(distance, -1).to(device)

    point = point[centroids.type(torch.long)]
    # 返回结果是npoint个采样点在原始点云中的索引
    return point

def process_point_clouds_in_folder(input_folder, output_folder):
    """
    Process all .txt point cloud files in a folder and apply PCA alignment.

    :param input_folder: str, path to the folder containing input .txt files
    :param output_folder: str, path to save the aligned point clouds
    """
    def pca_align_point_cloud(point_cloud):
        """
        Align a point cloud using PCA to ensure rotational invariance.

        :param point_cloud: np.array, shape (N, D), the input point cloud
        :return: np.array, shape (N, D), the aligned point cloud
        """
        # Ensure point cloud is zero-centered
        centered_cloud = point_cloud - np.mean(point_cloud, axis=0)

        # Perform PCA
        pca = PCA(n_components=point_cloud.shape[1])
        pca.fit(centered_cloud)

        # Align the point cloud by projecting onto principal components
        aligned_cloud = pca.transform(centered_cloud)
        return aligned_cloud

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(input_folder, file_name)
            try:
                # Load the point cloud
                point_cloud = np.loadtxt(file_path)

                # Apply PCA alignment
                aligned_point_cloud = pca_align_point_cloud(point_cloud)

                # Save the aligned point cloud
                output_file_path = os.path.join(output_folder, file_name)
                np.savetxt(output_file_path, aligned_point_cloud, fmt="%.6f")
                print(f"Processed and saved: {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")





import os
import numpy as np

# Function to read all .txt point cloud files from a folder
def read_point_cloud_files(folder_path):
    point_clouds = {}
    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            file_path = os.path.join(folder_path, file)
            point_cloud = np.loadtxt(file_path)
            point_clouds[file] = point_cloud
    return point_clouds

# Function to perform random sampling
def random_sampling(points, n_samples):
    """
    Perform Random Sampling on a point cloud.
    :param points: np.array, shape (N, D), original point cloud
    :param n_samples: int, number of points to sample
    :return: np.array, shape (n_samples, D), sampled points
    """
    n_points = points.shape[0]
    if n_points <= n_samples:
        # If the number of points is less than or equal to the sample size, return all points
        return points
    sampled_indices = np.random.choice(n_points, n_samples, replace=False)
    sampled_points = points[sampled_indices]
    return sampled_points

# Function to perform farthest point sampling (FPS)
def farthest_point_sampling(points, n_samples):
    """
    Perform Farthest Point Sampling (FPS) on a point cloud.
    :param points: np.array, shape (N, D), original point cloud
    :param n_samples: int, number of points to sample
    :return: np.array, shape (n_samples, D), sampled points
    """
    n_points, dim = points.shape
    sampled_points = np.zeros((n_samples, dim))
    sampled_indices = np.zeros(n_samples, dtype=int)

    # Initialize: Choose the first point randomly
    sampled_indices[0] = np.random.randint(n_points)
    sampled_points[0] = points[sampled_indices[0]]

    # Compute distances from the first point
    distances = np.linalg.norm(points - sampled_points[0], axis=1)

    for i in range(1, n_samples):
        # Choose the farthest point from the already sampled points
        farthest_point_index = np.argmax(distances)
        sampled_indices[i] = farthest_point_index
        sampled_points[i] = points[farthest_point_index]

        # Update distances to the closest sampled point
        distances = np.minimum(distances, np.linalg.norm(points - sampled_points[i], axis=1))

    return sampled_points


# Function to process all point clouds and save the downsampled versions
def process_point_clouds(input_folder, output_folder, n_samples, decimal_places):
    point_clouds = read_point_cloud_files(input_folder)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name, points in point_clouds.items():
        # Perform farthest point sampling
        # sampled_points = farthest_point_sampling(points, n_samples)
        sampled_points = random_sampling(points, n_samples)

        # Round the sampled points to the specified decimal places
        sampled_points = np.round(sampled_points, decimals=decimal_places)

        # Save the downsampled point cloud to the output folder
        output_path = os.path.join(output_folder, file_name)
        np.savetxt(output_path, sampled_points, fmt=f'%.{decimal_places}f')
        print(f"Processed and saved: {file_name}")


# Samping
# input_folder = "/home/hjc/Program/Pointnet_faceland/dataset/new/procession_Data/method_1/sample_fps_20000/"  # Replace with your input folder path
# output_folder = "/home/hjc/Program/Pointnet_faceland/dataset/new/procession_Data/method_1/random_3000/"  # Replace with your output folder path
# n_samples = 3000  # Number of points to sample
# decimal_places = 3  # Number of decimal places to round
#
# process_point_clouds(input_folder, output_folder, n_samples, decimal_places)

# # Example usage
# input_folder = "/path/to/input/folder"  # Replace with your input folder containing .txt files
# output_folder = "/path/to/output/folder"  # Replace with your desired output folder for aligned files
#
# process_point_clouds_in_folder(input_folder, output_folder)
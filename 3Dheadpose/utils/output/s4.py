import numpy as np
import matplotlib.pyplot as plt

def generate_percentage_based_matrices(size, bottleneck_max, wasserstein_max, highlight_positions, percentages):
    """
    Generate Bottleneck and Wasserstein matrices with values distributed across specified intervals by percentages,
    and highlighted maximum positions.

    Args:
        size: Size of the square matrices (e.g., 10 for 10x10).
        bottleneck_max: Maximum value for the Bottleneck matrix.
        wasserstein_max: Maximum value for the Wasserstein matrix.
        highlight_positions: List of positions to highlight with the maximum value.
        percentages: A list of percentages for each interval, summing to 100 (e.g., [10, 20, 30, 30, 10]).

    Returns:
        bottleneck_matrix, wasserstein_matrix: Matrices with values distributed by percentages.
    """
    assert sum(percentages) == 100, "Percentages must sum to 100."

    # Define boundaries for each interval
    num_intervals = len(percentages)
    bottleneck_intervals = np.linspace(0, bottleneck_max, num_intervals + 1)
    wasserstein_intervals = np.linspace(0, wasserstein_max, num_intervals + 1)

    # Generate values for each interval based on percentages
    bottleneck_values = []
    wasserstein_values = []
    for i, percentage in enumerate(percentages):
        count = int(size * size * (percentage / 100))
        bottleneck_values.extend(np.random.uniform(bottleneck_intervals[i], bottleneck_intervals[i + 1], count))
        wasserstein_values.extend(np.random.uniform(wasserstein_intervals[i], wasserstein_intervals[i + 1], count))

    # Shuffle and reshape into square matrices
    np.random.shuffle(bottleneck_values)
    np.random.shuffle(wasserstein_values)
    bottleneck_matrix = np.array(bottleneck_values[:size * size]).reshape(size, size)
    wasserstein_matrix = np.array(wasserstein_values[:size * size]).reshape(size, size)

    # Symmetrize the matrices
    bottleneck_matrix = (bottleneck_matrix + bottleneck_matrix.T) / 2
    wasserstein_matrix = (wasserstein_matrix + wasserstein_matrix.T) / 2

    # Ensure diagonal is 0
    np.fill_diagonal(bottleneck_matrix, 0)
    np.fill_diagonal(wasserstein_matrix, 0)

    # Highlight specific positions with the maximum value
    for pos in highlight_positions:
        bottleneck_matrix[pos] = bottleneck_max
        wasserstein_matrix[pos] = wasserstein_max

    return bottleneck_matrix, wasserstein_matrix


# Function to visualize matrices
def visualize_matrices(bottleneck_matrix, wasserstein_matrix):
    """
    Visualize Bottleneck and Wasserstein matrices with specified color schemes and settings.
    """
    plt.figure(figsize=(12, 6))

    # Bottleneck Distance Matrix
    plt.subplot(1, 2, 1)
    im_b = plt.imshow(bottleneck_matrix, cmap='viridis', interpolation='nearest')
    cbar_b = plt.colorbar(im_b, label="Bottleneck Distance")
    ticks_b = np.linspace(0, bottleneck_matrix.max(), 6)  # 6 evenly spaced ticks
    tick_labels_b = [f"{tick:.2f}" for tick in ticks_b]  # Format to 2 decimal places
    cbar_b.set_ticks(ticks_b)
    cbar_b.ax.set_yticklabels(tick_labels_b)
    cbar_b.ax.yaxis.get_ticklabels()[-1].set_color('red')  # Set top tick label color to red
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.title("Bottleneck Distance Matrix")
    plt.xlabel("Point Cloud")
    plt.ylabel("Point Cloud")

    # Wasserstein Distance Matrix
    plt.subplot(1, 2, 2)
    im_w = plt.imshow(wasserstein_matrix, cmap='viridis', interpolation='nearest')
    cbar_w = plt.colorbar(im_w, label="Wasserstein Distance")
    ticks_w = np.linspace(0, wasserstein_matrix.max(), 6)  # 6 evenly spaced ticks
    tick_labels_w = [f"{tick:.2f}" for tick in ticks_w]  # Format to 2 decimal places
    cbar_w.set_ticks(ticks_w)
    cbar_w.ax.set_yticklabels(tick_labels_w)
    cbar_w.ax.yaxis.get_ticklabels()[-1].set_color('red')  # Set top tick label color to red

    plt.title("Wasserstein Distance Matrix")
    plt.xlabel("Point Cloud")
    plt.ylabel("Point Cloud")
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.tight_layout()
    plt.show()


# Parameters
matrix_size = 10
bottleneck_max_value = 0.09
wasserstein_max_value = 3.62
# highlight_positions = [(7, 3),(6, 3),(5, 3),(4, 3),(4, 3), (3, 4), (3, 5), (3, 6), (3, 7)]  # Highlight positions
highlight_positions = [(4, 3),(3, 4)]  # Highl
percentages = [0,10,50,40,0]  # Percentage distribution for each interval

# Generate percentage-based matrices
bottleneck_matrix, wasserstein_matrix = generate_percentage_based_matrices(
    matrix_size, bottleneck_max_value, wasserstein_max_value, highlight_positions, percentages
)

# Visualize matrices
visualize_matrices(bottleneck_matrix, wasserstein_matrix)

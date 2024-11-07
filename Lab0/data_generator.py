import numpy as np
import matplotlib.pyplot as plt


class DataGenerator():
    def __init__(self):
        self.x = []
        self.y = []

    # K - number of classes/clusters
    # N - number of data points in each class
    # sigma - std of clusters
    def generate(self, K, N, sigma):
        
        data = []
        labels = []
        
        for i in range(K):
            mean = np.random.uniform(0, 1)
            dataArray = np.random.normal(mean, sigma, (N, 2))
            data.append(dataArray)
            cluster_labels = np.full(N, i)  # Label for current cluster is 'i'
            labels.append(cluster_labels)
            
        # Combine the data and labels
        self.x = np.vstack(data)  # Stack all cluster data vertically
        self.y = np.concatenate(labels)  # Concatenate all cluster labels
        
        return data, labels

    # x - data points
    # y - labels
    def plot_data(self):
        plt.figure(figsize=(6, 6))
        
        # Get the unique classes from the labels array y
        unique_classes = np.unique(self.y)
        
        # Loop through each unique class (label)
        for label in unique_classes:
            # By iterating over the labels, it maps column 0 = x values and column 1 y values for all the cases where y == the label value.
            plt.scatter(self.x[self.y == label, 0], self.x[self.y == label, 1], label=f'Class {label}', alpha=0.7)

        # Add labels, title, and legend
        plt.xlabel("X Dimension")
        plt.ylabel("Y Dimension")
        plt.title("2D Dataset with Multiple Classes")
        plt.legend()
        
        # Show the plot
        plt.show()

    def rotate(self, ang):
        W = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        
        # Rotate each point in self.x using matrix multiplication with W
        self.x = (W @ self.x.T).T  # Rotate and transpose back

    def export_data(self, file_prefix):
        # Save as separate .npy files
        np.save(f"{file_prefix}_x.npy", self.x)
        np.save(f"{file_prefix}_y.npy", self.y)

    def import_data(self, file_prefix):
        self.x = np.load(f"{file_prefix}_x.npy")
        self.y = np.load(f"{file_prefix}_y.npy")
        print("Data loaded successfully.")
        

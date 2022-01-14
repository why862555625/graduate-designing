import numpy as np
from sklearn.cluster import KMeans
from src.main.cluster_init_drawging import init_cluster_data
from sklearn.metrics import fowlkes_mallows_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(5)
x, y = init_cluster_data()

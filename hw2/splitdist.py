
import numpy as np
import matplotlib.pyplot as plt

accuracies_file = np.load("task1accuracies.npz")
baseline_accuracies = accuracies_file["baselineAccuracies"] 
tree_accuracies = accuracies_file["treeAccuracies"] 
plt.hist(baseline_accuracies,bins=10)
plt.hist(tree_accuracies,bins=10)
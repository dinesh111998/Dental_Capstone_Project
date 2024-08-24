import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(data):
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.show()
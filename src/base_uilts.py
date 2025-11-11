from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import root_mean_squared_error, r2_score

def plot_xy_trend(
    x, y, x_label_, y_label_, title_, add_refline=None
):
    """
    Ploting X-Y trend using matplotlib's scatter plot function
    """
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, alpha=0.5)
    
    if (add_refline is not None): 
        if (type(add_refline) is list):
            plt.plot(
                add_refline,
                add_refline,
                color='red',
                linestyle='--'
            )
        else:
            raise TypeError("add_refline should be a list")

    plt.title(title_)
    plt.ylabel(y_label_)
    plt.xlabel(x_label_)
    plt.grid(True)
    plt.show()




def get_performance_metrics(y_true, y_pred):
    """
    Calculate and display model prediction performance metrics: R2-score and RMSE
    Parameters:
        y_true (numpy.ndarray): 1D-array with true value of the target variable
        y_pred (numpy.ndarray): 1D-array with predicted value of the target variable
    Returns:
        dict: Dictionary containing R2-score and RMSE
    """
    print(f"R2-score:: {r2_score(y_true, y_pred):.3f}")
    print(f"Root Mean Squared Error: {root_mean_squared_error(y_true, y_pred):.4f}")
    
    return {
        'r2_score': r2_score(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred)
    }
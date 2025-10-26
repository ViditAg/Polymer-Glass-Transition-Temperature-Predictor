from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import root_mean_squared_error, r2_score

def plot_target_value_distribution(
        train_df,
        test_df,
        target_col
    ):
    """
    """
    plt.figure(figsize=(8,6))
    sns.histplot(
        test_df[target_col],
        bins=50,
        label='test data',
        stat='probability',  # Normalize to show probability density
        alpha=0.3,
        edgecolor='none'  # Remove edges
    )
    sns.histplot(
        train_df[target_col],
        bins=50,
        label='training data',
        stat='probability',  # Normalize to show probability density
        alpha=0.3,
        edgecolor='none'  # Remove edges
    )
    plt.xlabel('log(Tg)')
    plt.ylabel('Probability distribution')
    plt.title('Distribution of log(Tg) in Training vs Test Sets')
    plt.legend()
    plt.show()



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
    """
    print(f"R2-score:: {r2_score(y_true, y_pred):.3f}")
    print(f"Root Mean Squared Error: {root_mean_squared_error(y_true, y_pred):.3f}")
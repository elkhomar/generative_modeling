import numpy as np
import pandas as pd

def anderson_darling_distance(original_distribution, generated_distribution):
    original_df = pd.DataFrame(original_distribution.to("cpu").numpy()).T
    generated_df = pd.DataFrame(generated_distribution.to("cpu").numpy()).T

    def indicator(a, b):
        indicator = int(a <= b)  # 1 if a <= b, 0 otherwise
        return indicator

    def u_i(X,Y):
        # Ui is a ratio between 0 and 1 that represents the proportion of values in Y that are smaller than Xi
        n=len(X)
        X.sort()
        l=[]
        for i in range(n):
            l.append((1/(n+2))*(sum(indicator(y,X[i]) for y in Y)+1))
        return l
    
    def W(X,Y):
        # Wn is a statistic that measures the distance between the empirical distribution function of the sample and the theoretical distribution function
        n=len(X)
        u=u_i(X,Y)
        w_n=-n-(1/n)*sum(((2*i-1)*(np.log(u[i-1])+np.log(1-u[n-i-1]))) for i in range(1,n+1))
        return w_n

    def anderson_darling(x,y):
        # For the final Anderson-Darling distance, we average the Wn values for each feature
        d=x.shape[0]
        W_n=(1/d)*sum(W(x.iloc[i].tolist(),y.iloc[i].tolist()) for i in range(d))
        return W_n

    return anderson_darling(original_df,generated_df)


def normalised_kendall_tau_distance(original_distribution, generated_distribution):
    """Compute the Kendall tau distance."""

    original_df = pd.DataFrame(original_distribution.to("cpu").numpy()).T
    generated_df = pd.DataFrame(generated_distribution.to("cpu").numpy()).T

    n = len(original_df)
    assert len(generated_df) == n, "Both lists have to be of equal length"
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = np.argsort(original_df)
    b = np.argsort(generated_df)
    ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
    return ndisordered / (n * (n - 1))


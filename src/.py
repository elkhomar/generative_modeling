import numpy as np
import pandas as pd
import torch

def anderson_darling_distance(generated_distribution, original_distribution):
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
        w_n=-n-(1/n)*sum(((2*i-1)*(np.log(u[i-1])+np.log(1-u[n-i]))) for i in range(1,n+1)) # u[n-i-1] -> u[n-i]
        return w_n

    def anderson_darling(x,y):
        # For the final Anderson-Darling distance, we average the Wn values for each feature
        d=x.shape[0]
        W_n=(1/d)*sum(W(x.iloc[i].tolist(),y.iloc[i].tolist()) for i in range(d))
        return W_n

    return anderson_darling(generated_df, original_df)


def normalised_kendall_tau_distance(
    original_distribution, generated_distribution
):
    """Compute the Kendall tau distance."""
    
    original_df = pd.DataFrame(original_distribution.to("cpu").numpy())
    generated_df = pd.DataFrame(generated_distribution.to("cpu").numpy())

    n = len(original_df)
    assert len(generated_df) == n, "Both lists have to be of equal length"
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = np.argsort(original_df)
    b = np.argsort(generated_df)
    ndisordered = np.logical_or(
        np.logical_and(a[i] < a[j], b[i] > b[j]),
        np.logical_and(a[i] > a[j], b[i] < b[j])
    ).sum()
    return ndisordered / (n * (n - 1))

def absolute_kendall_error(X_og, X_gen):
    def pseudo_obs(x):
        n=x.shape[0]
        d=x.shape[1]
        M=np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if(j==i):
                    M[i,i] = False
                else:
                    bool=True
                    for c in range(d):
                        if x[j,c] >= x[i,c]: 
                            bool=False
                    M[i,j]=bool
        return np.sum(M, 0)/(n-1)

    Z_og=pseudo_obs(np.array(X_og.to("cpu").numpy()))
    Z_gen=pseudo_obs(np.array(X_gen.to("cpu").numpy()))
    return np.linalg.norm(np.sort(Z_og)-np.sort(Z_gen), 1)

df = torch.arange(4000).reshape(1000, 4)
df_fake = torch.arange(1000, 5000).reshape(1000, 4)
anderson_darling_distance(df_fake, df)


df = torch.randn(1000, 4)
df_fake = torch.randn(1000, 4)
absolute_kendall_error((df_fake-2)**2, (df-2)**2)

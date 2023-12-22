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


def absolute_kendall_error_torch(X_og, X_gen):
    d = X_og.shape[1]
    n = X_og.shape[0]
    def pseudo_obs(x):
        zi = [(1/(n-1))*torch.sum(
            (x[i, 0] > x[:, 0]) &
            (x[i, 1] > x[:, 1]) &
            (x[i, 2] > x[:, 2]) &
            (x[i, 3] > x[:, 3]))
              for i in range(n)]

        zi.sort()
        return torch.tensor(zi)

    zin = pseudo_obs(X_og)
    zin_tilde = pseudo_obs(X_gen)
    ld = (zin-zin_tilde).abs().sum()/n
    return ld


#df = torch.randn(1000, 4)
#df_fake = torch.randn(1000, 4)
#absolute_kendall_error((df_fake-2)**2, (df-2)**2)
#a = 10

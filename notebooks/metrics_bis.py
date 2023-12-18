import torch
import pandas as pd
import numpy as np

def indicatrice(a, b):
    # Comparaison pour déterminer si a est inférieur ou égal à b
    indicator = int(a <= b)  # Retourne 1 si la condition est vraie, sinon 0
    return indicator
                      
df = pd.DataFrame(torch.randn(4, 1000).numpy())
df_fake = pd.DataFrame(torch.randn(4, 1000).numpy())

def u_i(X,Y):
    n=len(X)
    X.sort()
    l=[]
    for i in range(n):
        l.append((1/(n+2))*(sum(indicatrice(y,X[i]) for y in Y)+1))
    return l
def W(X,Y):
    n=len(X)
    u=u_i(X,Y)
    w_n=-n-(1/n)*sum(((2*i-1)*(np.log(u[i-1])+np.log(1-u[n-i-1]))) for i in range(1,n+1))
    return w_n

def anderson_darling(x,y):
    d=x.shape[0]
    W_n=(1/d)*sum(W(x.iloc[i].tolist(),y.iloc[i].tolist()) for i in range(d))
    return W_n

r=anderson_darling(df,df_fake)
print(r)

def normalised_kendall_tau_distance(values1, values2):
    """Compute the Kendall tau distance."""
    n = len(values1)
    assert len(values2) == n, "Both lists have to be of equal length"
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = np.argsort(values1)
    b = np.argsort(values2)
    ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
    return ndisordered / (n * (n - 1))

def AndersonDarling3(data,predictions):
    N,P = data.shape
    ADDistance = 0
    for station in range(P):
        temp_predictions = predictions[:,station].reshape(-1)
        temp_data = data[:,station].reshape(-1)
        sorted_array = np.sort(temp_predictions)
        count = np.zeros(len(temp_data))
        count = (1/(N+2)) * np.array([(temp_data < order).sum()+1 for order in sorted_array])
        idx = np.arange(1, N+1)
        ADDistance += - N - np.sum((2*idx - 1) * (np.log(count) + np.log(1-count[::-1])))/N
    return ADDistance/P

df = torch.randn(4, 1000).numpy()
df_fake = torch.randn(4, 1000).numpy()

ad = AndersonDarling3(df.T,df_fake.T)
ad
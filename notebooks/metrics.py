import torch
import pandas as pd
import numpy as np

torch_data = torch.randn(4, 100) 
numpy_data = torch_data.numpy()
df = pd.DataFrame(np.abs(numpy_data)) 
#print(df)
def indicatrice(a, b):
    # Comparaison pour déterminer si a est inférieur ou égal à b
    indicator = int(a <= b)  # Retourne 1 si la condition est vraie, sinon 0
    return indicator

torch_data = torch.randn(4, 100) 
numpy_data = torch_data.numpy()
df_fake = pd.DataFrame(np.abs(numpy_data)) 

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


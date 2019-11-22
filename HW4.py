# %% [markdown]
# # Homework 4
# %% [markdown]
# **ÖNEMLİ:** Beyler aşağıda her bölümde yapılması gereken seyleri belirttim. Herkes tamamladığı kısımların üstünü çift tilda (\~~) ile çizerse geriye ne kalmış anlayabiliriz. Ayrıca **OutputTable** adında bir class olusturdum, output table olusturmak icin uzun uzun yazmanıza gerek yok kullanımına BFGS kodundan bakabilirsiniz.
# 
# Yazıyı okuyanlar isminin üstünü cizebilirse herkes okuyunca bu yazıyı silebiliriz.:
# *   Sefa
# *   Yunus
# *   Harun
# *   ~~Akın~~
# %% [markdown]
# ### TODOs
# 
# Steepest Descent:
# *   Algorithm code
# *   Function 1 - Solution set 1
# *   Function 1 - Solution set 2
# *   Function 2 - Solution set 1
# *   Function 2 - Solution set 2
# 
# Newton's Method:
# *   Algorithm code
# *   Function 1 - Solution set 1
# *   Function 1 - Solution set 2
# *   Function 2 - Solution set 1
# *   Function 2 - Solution set 2
# 
# DFP:
# *   Algorithm code
# *   Function 1 - Solution set 1
# *   Function 1 - Solution set 2
# *   Function 2 - Solution set 1
# *   Function 2 - Solution set 2  
# 
# BFGS:
# *   ~~Algorithm code~~
# *   Function 1 - Solution set 1
# *   Function 1 - Solution set 2
# *   Function 2 - Solution set 1
# *   Function 2 - Solution set 2

# %%
import pandas as pd
import numpy as np
from sympy import Symbol, lambdify


# %%
x1 = Symbol("x1")
x2 = Symbol("x2")

func1 = (5*x1 - x2)**4 + (x1 - 2)**2 + x1 - 2*x2 + 12
func2 = 100*(x2 - x1**2)**2 + (1 - x1)**2 


f1 = lambdify([[x1,x2]], func1, "numpy")
f2 = lambdify([[x1,x2]], func2, "numpy")

grad_f1 = lambdify([[x1,x2]], func1.diff([[x1, x2]]), "numpy")
grad_f2 = lambdify([[x1,x2]], func2.diff([[x1, x2]]), "numpy")

# %% [markdown]
# ### Useful Functions

# %%
def np_str(x_k):
    '''
    Used to convert numpy array to string with determined format
    '''
    return np.array2string(x_k, precision=3, separator=',')


# %%
class OutputTable:
    def __init__(self):
        self.table = pd.DataFrame([],columns=['k', 'x^k', 'f(x^k)', 'd^k', 'a^k', 'x^k+1'])
    def add_row(self, k, xk, fxk, dk, ak, xkp):
        self.table.loc[len(self.table)] = [k, np_str(xk), fxk, np_str(dk), ak, np_str(xkp)]
    def print_latex(self):
        print(self.table.to_latex(index=True,float_format='%.3f'))

# %% [markdown]
# ### Exact Line Search

# %%
def BisectionMethod(f, a=-100,b=100,epsilon=0.005) :
    iteration=0
    while (b - a) >= epsilon:
        x_1 = (a + b) / 2
        fx_1 = f(x_1)
        if f(x_1 + epsilon) <= fx_1:
            a = x_1
        else:
            b = x_1
        iteration+=1
    x_star = (a+b)/2
    return x_star

def ExactLineSearch(f, x0, d):
    alpha = Symbol('alpha')
    function_alpha = f(np.array(x0)+alpha*np.array(d))
    f_alp = lambdify(alpha, function_alpha, 'numpy')
    alp_star = BisectionMethod(f_alp)
    return alp_star

# %% [markdown]
# ## Steepest Descent Method
# %% [markdown]
# ## Newton's Method
# %% [markdown]
# ## DFP

# %%
def DFP(f, grad_f, x_0, epsilon):
    k = 0
    H = np.identity(len(x_0))
    print(H)


# %%
DFP(f1, f1, [0,2,2], 0.02)

# %% [markdown]
# Akin kardesime selam olsun. 4. Katta bulusmak uzere...
#                                 Sevgiler
#                                  Harun
# %% [markdown]
# ## BFGS

# %%
def BFGS(f, grad_f, x_0, epsilon):
    xk = np.array(x_0)
    k = 0
    H = np.identity(len(x_0))
    stop = False
    output = OutputTable()
    while(stop == False):
        d = -H @ np.transpose(grad_f(xk))
        if(np.linalg.norm(d) < epsilon):
            stop = True
        else:
            a = ExactLineSearch(f,xk,d)
            xkp = xk + a*d
            p = xkp - xk
            q = np.transpose(grad_f(xkp)) - np.transpose(grad_f(xk))
            A = ((1+ np.transpose(q) @ H @ q) / (np.transpose(q) @ p)) * (p @ np.transpose(p)) / (np.transpose(p) @ q)
            B = - (p @ np.transpose(q) * H + H @ q @ np.transpose(p)) / (np.transpose(q) @ p)
            Hkp = H + A + B
            output.add_row(k, xk, f(xk), d, a, xkp)
            k += 1
            xk = xkp
            H = Hkp
    output.add_row(k,xk,f(xk),d,None,np.array([]))
    return xk, f(xk), output


# %%
xs1, fs1, output1 = BFGS(f1, grad_f1, [6,30], 0.005)


# %%
xs1, fs1


# %%
output1.table


# %%
from scipy.optimize import minimize


# %%
minimize(fun=f1, x0=[0,0], method='BFGS', tol=0.005)


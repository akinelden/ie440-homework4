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

gf1 = lambdify([[x1,x2]], func1.diff([[x1, x2]]), "numpy")
gf2 = lambdify([[x1,x2]], func2.diff([[x1, x2]]), "numpy")

grad_f1 = lambda x_arr : np.array(gf1(x_arr)).reshape(1,2)
grad_f2 = lambda x_arr : np.array(gf2(x_arr)).reshape(1,2)

hf1 = lambdify([[x1,x2]], (func1.diff([[x1, x2]])).diff([[x1, x2]]), "numpy")
hf2 = lambdify([[x1,x2]], (func2.diff([[x1, x2]])).diff([[x1, x2]]), "numpy")

hess_f1= lambda x_arr : np.array(hf1(np.array(x_arr).reshape(2,)))
hess_f2= lambda x_arr : np.array(hf2(np.array(x_arr).reshape(2,)))


# %%
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

# plot the function
x = np.arange(0,3,0.01)
y = np.arange(0,3,0.01)
X,Y = meshgrid(x, y) # grid of point
Z = f2([X,Y]) # evaluation of the function on the grid

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
                      cmap=cm.RdBu,linewidth=0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.savefig("graph.png")
plt.show()

# %% [markdown]
# ### Useful Functions

# %%
np_str = lambda x_k : np.array2string(x_k.reshape(len(x_k)), precision=3, separator=',')

f_str = lambda x : "{0:.4f}".format(x)


# %%
class OutputTable:    
    def __init__(self):
        self.table = pd.DataFrame([],columns=['k', 'x^k', 'f(x^k)', 'd^k', 'a^k', 'x^k+1'])
    def add_row(self, k, xk, fxk, dk, ak, xkp):
        self.table.loc[len(self.table)] = [k, np_str(xk), f_str(np.asscalar(fxk)), np_str(dk), ak, np_str(xkp)]
    def print_latex(self):
        print(self.table.to_latex(index=False))

# %% [markdown]
# ### Exact Line Search

# %%
def BisectionMethod(f,epsilon, a=-100,b=100) :
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

def ExactLineSearch(f, x0, d, eps=0.0000000001):
    alpha = Symbol('alpha')
    function_alpha = f(np.array(x0)+alpha*np.array(d))
    f_alp = lambdify(alpha, function_alpha, 'numpy')
    alp_star = BisectionMethod(f_alp, epsilon=eps)
    return alp_star

# %% [markdown]
# ## Steepest Descent Method

# %%
def steepestDescentMethod(f, grad_f, x_0, epsilon):
    xk = np.array(x_0).reshape(2,1)
    k = 0
    stop = False
    output = OutputTable()
    while(stop == False):
        d = - np.transpose(grad_f(xk))
        if(np.linalg.norm(d) < epsilon):
            stop = True
        else:
            a = ExactLineSearch(f,xk,d)
            xkp = xk + a*d
            output.add_row(k, xk, f(xk), d, a, xkp)
            k += 1
            xk = xkp
    output.add_row(k,xk,f(xk),d,None,np.array([]))
    return xk, np.asscalar(f(xk)), output


# %%
xs1, fs1, outputs1 = steepestDescentMethod(f1, grad_f1, [10,10], 0.001)
xs1, fs1


# %%
print(outputs1.table.to_latex(index=False))


# %%
xs2, fs2, outputs2 = steepestDescentMethod(f1, grad_f1, [-25,-15], 0.001)
xs2, fs2


# %%
print(outputs2.table.to_latex(index=False))


# %%
xs3, fs3, outputs3 = steepestDescentMethod(f2, grad_f2, [2,-4], 0.01)
xs3, fs3


# %%
print(outputs3.table.to_latex(index=False))


# %%
xs4, fs4, outputs4 = steepestDescentMethod(f2, grad_f2, [-2,-3.5], 0.002)
xs4, fs4


# %%
outputs4.table

# %% [markdown]
# ## Newton's Method

# %%
def NewtonsMethod(x_0,epsilon,f,grad_f,Hessian_f):
    xk = np.array(x_0).reshape(2,1)
    k=0
    output = OutputTable()
    while(True):
        d_k=-np.linalg.inv(Hessian_f(xk))@np.transpose(grad_f(xk))
        alpha_k=ExactLineSearch(f,xk,d_k)
        xkp=xk+alpha_k*d_k
        if(np.linalg.norm(grad_f(xk)) < epsilon):
            break
        output.add_row(k, xk, f(xk), d_k, alpha_k, xkp)
        xk = xkp
        k += 1
    output.add_row(k,xk,f(xk),d_k,None,np.array([]))    
    return xk, np.asscalar(f(xk)), output


# %%
x_f1_s1,f1_s1, outputf1_1 = NewtonsMethod([-5,1], 0.01,f1,grad_f1,hess_f1)
x_f1_s1,f1_s1
print( outputf1_1.table.to_latex(index=False))


# %%
x_f1_s2,f1_s2, outputf1_2 = NewtonsMethod([-25,75], 0.001,f1,grad_f1,hess_f1)
x_f1_s2,f1_s2
print( outputf1_2.table.to_latex(index=False))


# %%
x_f2_s1,f2_s1, outputf2_1 = NewtonsMethod([-2,4], 0.01,f1,grad_f1,hess_f1)
print( outputf2_1.table.to_latex(index=False))
# %%

x_f2_s2,f2_s2, outputf2_2 = NewtonsMethod([-10,1], 0.001,f1,grad_f1,hess_f1)
print( outputf2_2.table.to_latex(index=False))
# %% [markdown]
# ## DFP

# %%
def DFP(f, grad_f, x_0, epsilon):
    xk = np.array(x_0).reshape(2,1)
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
            A = (p @ np.transpose(p)) / (p.transpose() @ q)
            B = - (H @ q @ np.transpose( H @ q)) / (q.transpose() @ H @ q)
            Hkp = H + A + B
            output.add_row(k, xk, f(xk), d, a, xkp)
            k += 1
            xk = xkp
            H = Hkp
    output.add_row(k,xk,f(xk),d,None,np.array([]))
    return xk, np.asscalar(f(xk)), output


# %%
xs1, fs1, output1 = DFP(f1, grad_f1, [0,0], 0.001)
xs1, fs1


# %%
output1.print_latex()


# %%
xs2, fs2, output2 = DFP(f1, grad_f1, [5,5], 0.0001)
xs2, fs2


# %%
output2.print_latex()


# %%
xs3, fs3, output3 = DFP(f2, grad_f2, [1.2,1.6], 1e-9)
xs3, fs3


# %%
output3.print_latex()


# %%
xs4, fs4, output4 = DFP(f2, grad_f2, [-2,-3], 1e-5)
xs4, fs4


# %%
output4.print_latex()

# %% [markdown]
# ## BFGS

# %%
def BFGS(f, grad_f, x_0, epsilon, line_search_tol = 0.0000001):
    xk = np.array(x_0).reshape(2,1)
    k = 0
    H = np.identity(len(x_0))
    stop = False
    output = OutputTable()
    while(stop == False):
        d = -H @ np.transpose(grad_f(xk))
        if(np.linalg.norm(d) < epsilon):
            stop = True
        if(k == -1):
            break
        else:
            a = ExactLineSearch(f,xk,d, line_search_tol)
            xkp = xk + a*d
            p = xkp - xk
            q = np.transpose(grad_f(xkp)) - np.transpose(grad_f(xk))
            A = ((1+ np.transpose(q) @ H @ q) / (np.transpose(q) @ p)) * (p @ np.transpose(p)) / (np.transpose(p) @ q)
            B = - (p @ np.transpose(q) @ H + H @ q @ np.transpose(p)) / (np.transpose(q) @ p)
            Hkp = H + A + B
            output.add_row(k, xk, f(xk), d, a, xkp)
            k += 1
            xk = xkp
            H = Hkp
    output.add_row(k,xk,f(xk),d,None,np.array([]))
    return xk, np.asscalar(f(xk)), output


# %%
xs1, fs1, output1 = BFGS(f1, grad_f1, [0,0], 0.01)
xs1, fs1


# %%
output1.print_latex()


# %%
xs2, fs2, output2 = BFGS(f1, grad_f1, [10,10], 0.01)
xs2, fs2


# %%
output2.print_latex()


# %%
xs3, fs3, output3 = BFGS(f2, grad_f2, [0,0], 0.01)
xs3, fs3


# %%
output3.print_latex()


# %%
xs4, fs4, output4 = BFGS(f2, grad_f2, [10,10], 0.001, line_search_tol=10**(-9))
xs4, fs4


# %%
output4.print_latex()

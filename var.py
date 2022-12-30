import model as md  

import pennylane as qml
from pennylane import numpy as nnp
import numpy as np
import torch
import random
import sympy
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import math as m
from tqdm import trange
import torch as th
from torch import nn
from scipy.special import rel_entr
import collections
import os

def expre_1(F,n):
    d = 2**n
    
    return np.sqrt(F-1/d)

def expre_2(F,n):
    d = 2**n
    d1 = 2**(n-1)
    dd = (d-1)*d1
    return np.sqrt(F-1/dd)


def alpha(n):
    d = 2**n
    return 2/d


def beta(n):
    d = 2**n
    return abs((2/(d**1-1))*(1-1/d))


def var(expr1,expr2,nq):
    return beta(nq)+alpha(nq)*expr1+2*expr2

def var_t(model,nq,epoch,circ):

    soma = 0
    F = 0
    exp_1 = []
    exp_2 = []
    dx = []
    for nl in [2,10,20,30,40,50]:
        
        f = model(nq,nl,1)
        ddd = 0
        
        while ddd<1/(2**nq):
            soma_1 = 0
            soma_2 = 0
            tp = trange(epoch)
            for i in tp:
                tp.set_description(f" circ:{circ} L:{nl}  ")
                w = np.random.random((nq,nl))
                state1=f(w)
                    
                w = np.random.random((nq,nl))
                state2=f(w)
                soma_1+= qml.math.fidelity(state1, state2)
                soma_2+= qml.math.fidelity(state1, state2)**2
            ddd = soma_1/epoch
            
              

        dx.append(nl)
        exp_1.append(expre_1(soma_1/epoch,nq))
        exp_2.append(expre_2(soma_2/epoch,nq))

    dx = np.array(dx)
    exp_1 = np.array(exp_1)
    exp_2 = np.array(exp_2)
    print(exp_1)
    print(exp_2)
    return dx, var(exp_1,exp_2,nq),exp_1,exp_2

def var_sim(model,nq,epoch,circ):
    
    
    var = []
   
    for nl in [2,10,20,30,40,50]:
        soma = 0
        f = model(nq,nl,2)
        tp = trange(epoch)
        ss = []
        for i in tp:
            tp.set_description(f" circ:{circ} L:{nl}  ")
            w = np.random.random((nq,nl))
            ss.append(f(w)[0])
       
        var.append(np.var(ss))
    
    return np.array(var)



##################################################


for nq in [4,5,6]:

    if not os.path.exists('./NumQubtis_{}'.format(nq)):
        os.mkdir('./NumQubtis_{}'.format(nq))

    if not os.path.exists('./NumQubtis_{}/graficos'.format(nq)):
        os.mkdir('./NumQubtis_{}/graficos'.format(nq))

    if not os.path.exists('./NumQubtis_{}/expr_1'.format(nq)):
        os.mkdir('./NumQubtis_{}/expr_1'.format(nq))

    if not os.path.exists('./NumQubtis_{}/expr_2'.format(nq)):
        os.mkdir('./NumQubtis_{}/expr_2'.format(nq))

    if not os.path.exists('./NumQubtis_{}/var_sim'.format(nq)):
        os.mkdir('./NumQubtis_{}/var_sim'.format(nq))

    list_model = [md.model1,md.model2,md.model3,md.model4,md.model5,md.model6,md.model7,md.model8,md.model9,md.model10,md.model11,md.model12]
    
    for mod in range(12):


        
        dx,var_t_,exp_1,exp_2 = var_t(list_model_1[mod],nq,5000,mod+1)
        
        var_s = var_sim(list_model_1[mod],nq,5000,mod+1)
        plt.title('Model {}'.format(mod+1))
        plt.plot(dx,var_t_,'--o',label='Var_t') 
        plt.plot(dx,var_s,'o',label='Var_s') 
        plt.xlabel('Number Layer')
        plt.legend()
        plt.xticks([2,10,20,30,40,50])
        plt.savefig('./NumQubtis_{}/graficos/model_{}.pdf'.format(nq,mod+1))
        plt.close()

        np.savetxt('./NumQubtis_{}/expr_1/expr_model_{}.txt'.format(nq,mod+1),exp_1)
        np.savetxt('./NumQubtis_{}/expr_2/expr_model_{}.txt'.format(nq,mod+1),exp_2)
        np.savetxt('./NumQubtis_{}/var_sim/var_sim_model_{}.txt'.format(nq,mod+1),var_s)


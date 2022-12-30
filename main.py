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

def expre(F,n):
    d = 2**n
    
    return np.sqrt(F-1/d)

def main_1(model,nq,epoch,num):

    soma = 0
    F = 0
    exp = []
    dx = []
    for nl in [2,10,20,30,40,50]:
        
        f = model(nq,nl,num)
        
        ddd = 0
        
        numero = 0
        while ddd<1/(2**nq):
            soma = 0
            tp = trange(epoch)
            for i in tp:
                tp.set_description(f" L:{nl}  ")
                w = np.random.random((nq,nl))
                state1=f(w)
                    
                w = np.random.random((nq,nl))
                state2=f(w)
                soma+= qml.math.fidelity(state1, state2)
            ddd = soma/epoch
            numero +=1
              
        
        dx.append(nl)
        exp.append(expre(soma/epoch,nq))

    dx = np.array(dx)
    exp = np.array(exp)
    return dx, exp

def main_2(model,nq,epoch,circ,num):
    
    
    med = []
    #dx = []
    for nl in [2,10,20,30,40,50]:
        soma = 0
        f = model(nq,nl,num)
        tp = trange(epoch)
        for i in tp:
            tp.set_description(f" circ:{circ} NL:{nl}  ")
            w = np.random.random((nq,nl))
            out = f(w)
            soma+=out[0]
        
        med.append(soma/epoch)
    med = abs(np.array(med)-1/(2**nq))
    return med


##########################################################################################################

for nq in [4,5,6]:

    if not os.path.exists('./te_NumQubtis_{}'.format(nq)):
        os.mkdir('./te_NumQubtis_{}'.format(nq))

    if not os.path.exists('./te_NumQubtis_{}/graficos'.format(nq)):
        os.mkdir('./te_NumQubtis_{}/graficos'.format(nq))

    if not os.path.exists('./te_NumQubtis_{}/expr'.format(nq)):
        os.mkdir('./te_NumQubtis_{}/expr'.format(nq))

    if not os.path.exists('./te_NumQubtis_{}/med'.format(nq)):
        os.mkdir('./te_NumQubtis_{}/med'.format(nq))

    list_model = [md.model1,md.model2,md.model3,md.model4,md.model5,md.model6,md.model7,md.model8,md.model9,md.model10,md.model11,md.model12]
   
    for mod in range(0,12):


        dx,exp = main_1(list_model[mod],nq,5000,1)
        med = main_2(list_model[mod],nq,5000,mod+1,2)
        plt.title('Model {}'.format(mod+1))
        plt.plot(dx,exp,'--o',label='Expr') 
        plt.plot(dx,med,'o',label='$\mu$')
        plt.xlabel('Number Layer')
        plt.legend()
        plt.xticks([2,10,20,30,40,50])
        plt.savefig('./te_NumQubtis_{}/graficos/model_{}.pdf'.format(nq,mod+1))
        plt.close()

        np.savetxt('./te_NumQubtis_{}/expr/expr_model_{}.txt'.format(nq,mod+1),exp)
        np.savetxt('./te_NumQubtis_{}/med/med_model_{}.txt'.format(nq,mod+1),med)



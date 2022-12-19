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

def main_1(model,nq,epoch):

    soma = 0
    F = 0
    exp = []
    dx = []
    for nl in [2,10,20,30,40,50]:
        
        f = model(nq,nl)
        
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

def main_2(model,nq,epoch,circ):
    
    
    med = []
    #dx = []
    for nl in [2,10,20,30,40,50]:
        soma = 0
        f = model(nq,nl)
        tp = trange(epoch)
        for i in tp:
            tp.set_description(f" circ:{circ} NL:{nl}  ")
            w = np.random.random((nq,nl))
            out = f(w)
            soma+=out[0]
        #dx.append(nl)
        med.append(soma/epoch)
    med = abs(np.array(med)-1/(2**nq))
    return med


##########################################################################################################

for nq in [4,5,6]:

    if not os.path.exists('./NumQubtis_{}'.format(nq)):
        os.mkdir('./NumQubtis_{}'.format(nq))

    if not os.path.exists('./NumQubtis_{}/graficos'.format(nq)):
        os.mkdir('./NumQubtis_{}/graficos'.format(nq))

    if not os.path.exists('./NumQubtis_{}/expr'.format(nq)):
        os.mkdir('./NumQubtis_{}/expr'.format(nq))

    if not os.path.exists('./NumQubtis_{}/med'.format(nq)):
        os.mkdir('./NumQubtis_{}/med'.format(nq))

    list_model_1 = [md.model1_1,md.model2_1,md.model3_1,md.model4_1,md.model5_1,md.model6_1,md.model7_1,md.model8_1,md.model9_1,md.model10_1,md.model11_1,md.model12_1]
    list_model_2 = [md.model1_2,md.model2_2,md.model3_2,md.model4_2,md.model5_2,md.model6_2,md.model7_2,md.model8_2,md.model9_2,md.model10_2,md.model11_2,md.model12_2]

    for mod in range(0,12):


        dx,exp = main_1(list_model_1[mod],nq,2000)
        med = main_2(list_model_2[mod],nq,2000,mod+1)
        plt.title('Model {}'.format(mod+1))
        plt.plot(dx,exp,'--o',label='Expr') 
        plt.plot(dx,med,'o',label='$\mu$')
        plt.xlabel('Number Layer')
        plt.legend()
        plt.xticks([2,10,20,30,40,50])
        plt.savefig('./NumQubtis_{}/graficos/model_{}.pdf'.format(nq,mod+1))
        plt.close()

        np.savetxt('./NumQubtis_{}/expr/expr_model_{}.txt'.format(nq,mod+1),exp)
        np.savetxt('./NumQubtis_{}/med/med_model_{}.txt'.format(nq,mod+1),med)



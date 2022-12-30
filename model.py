
import pennylane as qml
from pennylane import numpy as nnp
import numpy as np
import torch
import random
import sympy
import numpy as np

import matplotlib.pyplot as plt

import math as m
from tqdm import trange
import torch as th
from torch import nn
from scipy.special import rel_entr
import collections

def model1(nq,nl,num):
    dev = qml.device('default.qubit',wires=nq)
    @qml.qnode(dev)
    def f(w):
        
        #parametrization
        for i in range(nl):
            for j in range(nq):
                qml.RY(w[j][i],wires=j)
            for j in range(nq-1):
                qml.CNOT(wires=[j,1+j])
        if num == 1:
            return qml.state()
        elif num == 2:
            return qml.probs(wires=np.arange(nq))  

    return f



def model2(nq,nl,num):
    dev = qml.device('default.qubit',wires=nq)
    @qml.qnode(dev)
    def f(w):
        
        #parametrization
        for i in range(nl):
            for j in range(nq):
                qml.RY(w[j][i],wires=j)
            for j in range(nq-1):
                qml.CRY(m.pi/2,wires=[j,1+j])
        
        if num == 1:
            return qml.state()
        elif num == 2:
            return qml.probs(wires=np.arange(nq))  
    return f




def model3(nq,nl,num):
    dev = qml.device('default.qubit',wires=nq)
    @qml.qnode(dev)
    def f(w):
        
        #parametrization
        for i in range(nl):
            for j in range(nq):
                qml.Hadamard(wires=j)
            for j in range(nq):
                qml.RY(w[j][i],wires=j)
            for j in range(nq-1):
                qml.CRY(m.pi/2,wires=[j,1+j])
        
        if num == 1:
            return qml.state()
        elif num == 2:
            return qml.probs(wires=np.arange(nq))  
    return f




def model4(nq,nl,num):
    dev = qml.device('default.qubit',wires=nq)
    @qml.qnode(dev)
    def f(w):
        
        #parametrization
        for i in range(nl):
            for j in range(nq):
                qml.Hadamard(wires=j)
            for j in range(nq):
                qml.RY(w[j][i],wires=j)
            for j in range(nq-1):
                qml.CNOT(wires=[j,1+j])
        
        if num == 1:
            return qml.state()
        elif num == 2:
            return qml.probs(wires=np.arange(nq))  
    return f




def model5(nq,nl,num):
    dev = qml.device('default.qubit',wires=nq)
    @qml.qnode(dev)
    def f(w):
        
        #parametrization
        for i in range(nl):
            for j in range(nq):
                qml.RY(w[j][i],wires=j)
            for j in range(nq-1):
                qml.CZ(wires=[j,1+j])
        
        if num == 1:
            return qml.state()
        elif num == 2:
            return qml.probs(wires=np.arange(nq))  
    return f


def model6(nq,nl,num):
    dev = qml.device('default.qubit',wires=nq)
    @qml.qnode(dev)
    def f(w):
        for i in range(nl):
            for j in range(nq):
                qml.Hadamard(wires=j)
            for j in range(nq):
                qml.RY(w[j][i],wires=j)
            for j in range(nq-1):
                qml.CZ(wires=[j,j+1])
        if num == 1:
            return qml.state()
        elif num == 2:
            return qml.probs(wires=np.arange(nq))  
    return f



def model7(nq,nl,num):
    dev = qml.device('default.qubit',wires=nq)
    @qml.qnode(dev)
    def f(w):
        
        #parametrization
        for i in range(nl):
            for j in range(nq):
                qml.RY(w[j][i],wires=j)
            for j in range(nq):
            	for k in range(nq):
            		if j!=k:
                		qml.CNOT(wires=[j,k])
        
        if num == 1:
            return qml.state()
        elif num == 2:
            return qml.probs(wires=np.arange(nq))  
    return f




def model8(nq,nl,num):
    dev = qml.device('default.qubit',wires=nq)
    @qml.qnode(dev)
    def f(w):
        
        #parametrization
        for i in range(nl):
            for j in range(nq):
                qml.RY(w[j][i],wires=j)
            for j in range(nq):
            	for k in range(nq):
            		if j!=k:
                		qml.CRY(m.pi/2,wires=[j,k])
        
        if num == 1:
            return qml.state()
        elif num == 2:
            return qml.probs(wires=np.arange(nq))  
    return f





def model9(nq,nl,num):
    dev = qml.device('default.qubit',wires=nq)
    @qml.qnode(dev)
    def f(w):
        
        #parametrization
        for i in range(nl):
            for j in range(nq):
                qml.Hadamard(wires=j)
            for j in range(nq):
                qml.RY(w[j][i],wires=j)
            for j in range(nq):
            	for k in range(nq):
            		if j!=k:
                		qml.CRY(m.pi/2,wires=[j,k])
        
        if num == 1:
            return qml.state()
        elif num == 2:
            return qml.probs(wires=np.arange(nq))  
    return f




def model10(nq,nl,num):
    dev = qml.device('default.qubit',wires=nq)
    @qml.qnode(dev)
    def f(w):
        
        #parametrization
        for i in range(nl):
            for j in range(nq):
                qml.Hadamard(wires=j)
            for j in range(nq):
                qml.RY(w[j][i],wires=j)
            for j in range(nq):
            	for k in range(nq):
            		if j!=k:
                		qml.CNOT(wires=[j,k])
        
        if num == 1:
            return qml.state()
        elif num == 2:
            return qml.probs(wires=np.arange(nq))  
    return f




def model11(nq,nl,num):
    dev = qml.device('default.qubit',wires=nq)
    @qml.qnode(dev)
    def f(w):
        
        #parametrization
        for i in range(nl):
            for j in range(nq):
                qml.RY(w[j][i],wires=j)
            for j in range(nq):
            	for k in range(nq):
            		if j!=k:
                		qml.CZ(wires=[j,k])
        
        if num == 1:
            return qml.state()
        elif num == 2:
            return qml.probs(wires=np.arange(nq))  
    return f


def model12(nq,nl,num):
    dev = qml.device('default.qubit',wires=nq)
    @qml.qnode(dev)
    def f(w):
        #parametrization
        for i in range(nl):
            for j in range(nq):
                qml.Hadamard(wires=j)
            for j in range(nq):
                qml.RY(w[j][i],wires=j)
            for j in range(nq):
                for k in range(nq):
                    if j!=k:
                        qml.CZ(wires=[j,k])
        if num == 1:
            return qml.state()
        elif num == 2:
            return qml.probs(wires=np.arange(nq))  
    return f




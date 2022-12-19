
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

def model1_1(nq,nl):
    dev = qml.device('default.qubit',wires=nq)
    @qml.qnode(dev)
    def f(w):
        
        #parametrization
        for i in range(nl):
            for j in range(nq):
                qml.RY(w[j][i],wires=j)
            for j in range(nq-1):
                qml.CNOT(wires=[j,1+j])
        
        return qml.state()
    return f


def model1_2(nq,nl):
    dev = qml.device('default.qubit',wires=nq)
    @qml.qnode(dev)
    def f(w):
        
        #parametrization
        for i in range(nl):
            for j in range(nq):
                qml.RY(w[j][i],wires=j)
            for j in range(nq-1):
                qml.CNOT(wires=[j,1+j])
        
        return qml.probs(wires=np.arange(nq))  
    return f


def model2_1(nq,nl):
    dev = qml.device('default.qubit',wires=nq)
    @qml.qnode(dev)
    def f(w):
        
        #parametrization
        for i in range(nl):
            for j in range(nq):
                qml.RY(w[j][i],wires=j)
            for j in range(nq-1):
                qml.CRY(m.pi/2,wires=[j,1+j])
        
        return qml.state()
    return f


def model2_2(nq,nl):
    dev = qml.device('default.qubit',wires=nq)
    @qml.qnode(dev)
    def f(w):
        
        #parametrization
        for i in range(nl):
            for j in range(nq):
                qml.RY(w[j][i],wires=j)
            for j in range(nq-1):
                qml.CRY(m.pi/2,wires=[j,1+j])
        
        return qml.probs(wires=np.arange(nq))  
    return f

def model3_1(nq,nl):
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
        
        return qml.state()
    return f

def model3_2(nq,nl):
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
        
        return qml.probs(wires=np.arange(nq))
    return f



def model4_1(nq,nl):
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
        
        return qml.state()
    return f


def model4_2(nq,nl):
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
        
        return qml.probs(wires=np.arange(nq))
    return f


def model5_1(nq,nl):
    dev = qml.device('default.qubit',wires=nq)
    @qml.qnode(dev)
    def f(w):
        
        #parametrization
        for i in range(nl):
            for j in range(nq):
                qml.RY(w[j][i],wires=j)
            for j in range(nq-1):
                qml.CZ(wires=[j,1+j])
        
        return qml.state()
    return f

def model5_2(nq,nl):
    dev = qml.device('default.qubit',wires=nq)
    @qml.qnode(dev)
    def f(w):
        
        #parametrization
        for i in range(nl):
            for j in range(nq):
                qml.RY(w[j][i],wires=j)
            for j in range(nq-1):
                qml.CZ(wires=[j,1+j])
        
        return qml.probs(wires=np.arange(nq))
    return f

def model6_1(nq,nl):
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
    	return qml.state()
       
    return f

def model6_2(nq,nl):
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
    	return qml.probs(wires=np.arange(nq))
       
    return f


def model7_1(nq,nl):
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
        
        return qml.state()
    return f

def model7_2(nq,nl):
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
        
        return qml.probs(wires=np.arange(nq))  
    return f



def model8_1(nq,nl):
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
        
        return qml.state()
    return f


def model8_2(nq,nl):
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
        
        return qml.probs(wires=np.arange(nq))  
    return f




def model9_1(nq,nl):
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
        
        return qml.state()
    return f



def model9_2(nq,nl):
    dev = qml.device('default.qubit',wires=nq)
    @qml.qnode(dev)
    def f(w):
    	for i in range(nl):
    		for j in range(nq):
    			qml.Hadamard(wires=j)
    		for j in range(nq):
    			qml.RY(w[j][i],wires=j)
    		for j in range(nq):
    			for k in range(nq):
    				if j!=k:
    					qml.CRY(m.pi/2,wires=[j,k])
    	return qml.probs(wires=np.arange(nq))
    return f



def model10_1(nq,nl):
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
        
        return qml.state()
    return f



def model10_2(nq,nl):
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
        
        return qml.probs(wires=np.arange(nq))
    return f




def model11_1(nq,nl):
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
        
        return qml.state()
    return f


def model11_2(nq,nl):
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
        
        return qml.probs(wires=np.arange(nq))
    return f


def model12_1(nq,nl):
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
    	return qml.state()
    return f


def model12_2(nq,nl):
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
    	return qml.probs(wires=np.arange(nq))
    return f

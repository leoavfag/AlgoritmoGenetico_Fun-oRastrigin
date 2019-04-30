# -*- coding: utf-8 -*-
import numpy as np
import random
import math
import matplotlib.pyplot as pt

tamP = 20
Dim = 2
TamGer = 100    #Quantas gerações
Ch_Mut = 0.3    #Chance de mutação
Q_Mut = 1.05    # Quantidade de mutação
max_ = 5.12;
min_ = -5.12
A = 10          #Constante da função rastrigin
chancec = 0.9

def PopI(): 
    return np.random.uniform(min_,max_,(tamP,Dim)) #População INICIAL


def Rastrigin(POPI,**kwargs):
    A = kwargs.get('A')
    [row,col] = POPI.shape
    fx = np.zeros(row)
    for i in range(row):
        for j in range(col):
            fx[i] += POPI[i,j]**2 - A * np.cos(2*np.pi*POPI[i,j])
    return A*col + fx 


def inverte(fx):    #Função fitness, para nao ter problemas de divisao por zero
    hx = min(fx)    #Ocorre um deslocamento para tal
    desl = 10**-0.2 - hx
    return 1/( fx + desl)


def torneio(fx):
    ind = np.zeros(len(fx),dtype=int) #cria vetor com o tamanho do fx
    for i in range(len(fx)):
        i1 = random.randint(0,len(fx)-1)
        i2 = random.randint(0, len(fx)-1)
        if fx[i1]>fx[i2]:
            ind[i] = i1
        else:
            ind[i] = i2
    return ind

def cruzamento(X,chancec):
    l = X.shape[0]
    c = X.shape[1]
    Xfilho = np.zeros((l,c))
    filho=0
    alpha=0.9
    while filho <l:
        i1 = random.randint(0,l-1)
        i2 = random.randint(0,l-1)
        if random.random() < chancec:
            Xfilho[filho,:] = alpha*X[i1,:] + (1-alpha)* X[i2,:]
            filho+=1
            Xfilho[filho,:] = alpha*X[i2,:] + (1-alpha)* X[i1,:]
            filho+=1
    return Xfilho

def mutacao(X,chance): 
    [row,col] = X.shape         #recebe o numero de linhas e colunas
    Xn = np.zeros((row,col))    #cria um vetor com zeros    
    for i in range(row):        
        if random.random() <= chance:

            if random.random() > 0.5:
                Xn[i,:] = X[i,:]*Q_Mut
                for j in range(col):    
                    if Xn[i,j] > max_:
                        Xn[i,j]=max_
                    elif Xn[i,j] < min_:
                        Xn[i,j]=min_
            else:
                Xn[i,:] = X[i,:]*-Q_Mut
                for j in range(col):    
                    if Xn[i,j] > max_:
                        Xn[i,j]=max_
                    elif Xn[i,j] < min_:
                        Xn[i,j]=min_
        else:
            Xn[i:] = X[i,:]
    return Xn
    
vezes = 30
melhores = np.zeros((TamGer,Dim))
mc = np.zeros((vezes,Dim))

for i in range(vezes):
    a = PopI()
    z = Rastrigin(a,A=10)
    fx = inverte(z)
    for j in range(TamGer):
        
        mel = np.argmin(z)
        mfx = fx[mel]
        melhores[j,:] = a[mel, :]

        ele = torneio(fx)
        ind = np.array(a)[ele,:]
        
        Xc = cruzamento(ind,chancec)
        
        Xn = mutacao(Xc, Ch_Mut)
        
        z = Rastrigin(Xn, A=10) #refaz os passos para o novo X
        fx = inverte(z)
        n = np.argmin(z)
        
        Pioraux = np.argmin(fx)
        Xn[Pioraux,:] = a[mel,:]
        fx[Pioraux] = mfx

        z = Rastrigin(Xn,A=10)
        fx = inverte(z)
        n = np.argmin(z)
        print("z: ",z[n])

        
        if j == TamGer-1:
            if fx[n] > mfx:
                mc[i,:] = Xn[n]
            else:
                mc[i,:] = a[mel]
            z = Rastrigin(melhores, A=10)
            pt.plot(z)
            pt.title("Soluções ")
            pt.show()

        a = Xn
            

#------------------------Plotando os melhores Rastrigin

y = Rastrigin(mc,A=10)
pt.plot(y)
pt.title("Soluções finais")
pt.show()
print(y)
for i in range(len(mc)):
    print(i,"-> ",mc[i])
    print("Rastrigin: ",y[i])
    
    
    
    
    
    


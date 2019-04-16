# -*- coding: utf-8 -*-
import numpy as np
import random
import math

tamP = 20
Dim = 2
TamGer = 10 #Quantas gerações
Ch_Mut = 0.3 #Chance de mutação
max_ = 5.12
min_ = -5.12
A = 10 #Constante da função rastrigin


def Init():
    POPI = np.zeros((tamP,Dim)) #População INICIAL
    
    semente = random.randint(0,1000)   #Primeira semente aleatoria
    
    conjAlea = np.random.RandomState(seed = semente) #Gera conjunto aleatorio com a semente
    
    POPconj = conjAlea.uniform(min_,max_,tamP) #recebe um conjunto aleatorio de
                                                               #20 elementos entre -5.12 e 5.12
                                                               #do conjunto gerado com a semente
    
    POPconj2 = conjAlea.uniform(min_, max_,tamP)#recebe um conjunto aleatorio de
                                                               #20 elementos entre -5.12 e 5.12
                                                               #do conjunto gerado com a semente
    
    for gene in range(tamP):            #Preenche a população com os genes
        POPI[gene][0]= POPconj[gene]
        POPI[gene][1]= POPconj2[gene]
        
    return POPI

def Rastrigin(POPI,**kwargs): #calcula e retorna o valor da função e acordo com parametros recebidos
    A = kwargs.get('A')
    [row,col] = POPI.shape
    fx = np.zeros(row)
    for i in range(row):
        for j in range(col):
            fx[i] += POPI[i,j]**2 - A * np.cos(2*np.pi*POPI[i,j])
    return A*col + fx 

def inverte(fx):    #Função fitness, para nao ter problemas de divisao por zero
    hx = min(fx)    #Ocorre um deslocamento para tal
    desl = 10**-0.2 - hx;
    return 1/( fx + desl)

def torneio(fx):
    ind = np.zeros(len(fx)) #cria vetor com o tamanho do fx
    for i in range(len(fx)):
        i1 = random.randint(0,len(fx)-1)
        i2 = random.randint(0, len(fx)-1)
        if fx[i1]>fx[i2]:
            ind[i] = i1
        else:
            ind[i] = i2
    return ind


def mutacao(X,chance): 
    [row,col] = X.shape         #recebe o numero de linhas e colunas
    Xn = np.zeros((row,col))    #cria um vetor com zeros    
    for i in range(row):        
        #i1 = random.randint(0,row-1)  #sorteia um individuo
        if random.random() <= chance: #Gera um numero aleatorio de (0.0 , 1.0) e compara com a chance mutação
            if random.random() >0.5:
                Xn[i,:] = X[i,:]*1.05
                for j in range(col):    
                    if Xn[i,j] > max_:
                        Xn[i,j]=max_
                    elif Xn[i,j] < min_:
                        Xn[i,j]=min_
            else:
                Xn[i,:] = X[i,:]*-1.05
                for j in range(col):    
                    if Xn[i,j] > max_:
                        Xn[i,j]=max_
                    elif Xn[i,j] < min_:
                        Xn[i,j]=min_
            #print("Mutou: "+str(X[i])+" Mutado: "+str(Xn[i]))
        else:
            Xn[i:] = X[i,:]
    return Xn

def cruza(x,ind,fx): # cruza aleatoriamente os indices selecionados pelo fitness, mantem a parte inteira de uma e coloca a flutuante de outra
    [row,col] = x.shape
    xn = np.zeros((row,col))
    for i in range(row):
        i1 = random.randint(0,len(ind)-1)    #num aleatorio para pegar um indice dos selecionados(fitness) 
        i2 = random.randint(0, len(ind)-1)   #num aleatorio para pegar outro indice dos selecionados(fitness)
        for j in range(col):
            #x[ind[i1]] e x[ind[i2]]
            a = int(ind[i1])
            b = int(ind[i2])
            if fx[a]>fx[b]:
                inteira = int(x[a,j]) #pega a parte inteira do primeirok
                inteira2= int(x[b,j])
                flutuante = x[b,j] - inteira2 #pega a parte flutuante do segundo
                xn[i,j] = (inteira + flutuante) # soma as duas partes para criar o filho
            else:
                inteira = int(x[b,j]) #pega a parte inteira do primeiro
                inteira2= int(x[a,j]) #pega a parte inteira do segundo
                flutuante = x[a,j] - inteira2 #pega a parte flutuante do segundo
                xn[i,j] = (inteira + flutuante) # soma as duas partes para criar o filho
    
    return xn

def Geracoes(x,z,fx,**kwargs): #Cria um numero definido de gerações por um numero definido de vezes e armazena os melhores
    
    vezes = kwargs.get('vezes')
    [row,col] = x.shape
    melhores = np.zeros((vezes,col+1))
    for j in range(vezes):
        for i in range(TamGer):
           
            melhorind =  np.argmin(z)
            melhorfx = fx[melhorind]
            # torneio
            ind = torneio(fx)

            # AQUI PRECISO DO ALGORITMO DE CRUZAMENTO   
            Xc = cruza(x,ind,fx)

            Xn = mutacao(Xc, Ch_Mut) #<- Xc no lugar de x 
            
            z = Rastrigin(Xn, A=10) #refaz os passos para o novo Xc
            fx = inverte(z)
            auxind =  np.argmin(fx)         #Pega o indice do pior fx
            Xn[auxind, :] = x[melhorind, :] #Substitui na nova geracao aquele indice
            fx[auxind] = melhorfx
            print("MELHOR DA INTERAÇÂO "+str(j)+" GERAÇAO "+str(i)+": "+str(x[melhorind,:])+" fx:"+str(melhorfx) )
            x = Xn
            melhorindfin = np.argmin(z)
            
            if melhorfx > fx[melhorindfin]:
                melhores[j,0:2] = x[melhorind]
                melhores[j,2] = melhorfx
            else:
                melhores[j,0:2] = x[melhorindfin]
                melhores[j,2] = fx[melhorindfin]
                
            if i==TamGer-1:
                print(melhores[j,:])

    return melhores

x = Init()  # primeira geração
z = Rastrigin(x, A=10) # calcula valores rastrigin
fx = inverte(z) #calcula fitness

melhores = Geracoes (x, z ,fx, vezes = 30)
print("\n\n Melhores 30 de 100 em 100 gerações: \n")
for i in range(len(melhores)):
    print("Interação "+str(i)+"-> "+str(melhores[i,0:2])+" FX: "+str(melhores[i,2]))





#print(melhores)




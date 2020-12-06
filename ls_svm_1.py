# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 00:53:01 2018

@author: suraj prakash patil
@venue: department of technology, sppu
@topic: draft1- LS-SVM-Classify-From-Scratch- basic example
"""

import numpy as np


########### Training ##################

training_vec=np.matrix([(1,3),(2,1),(0,1)])
# types of class is 1 & -1
training_label=np.matrix([1,1,-1])

n=np.size(training_label)
n1=0
n2=0

for i in range(0,n):
    if(training_label[0,i]==1):
        n1=n1+1
    if(training_label[0,i]==-1):
        n2=n2+1
    
c1= 0.5 * (n/n1)
c2= 0.5 * (n/n2)

box=[]
for i in range(0,n1):
    box.append(1/c1)
for i in range(0,n2):
    box.append(1/c2)
box1=np.identity(n) * (box)
del box;    del c1;     del c2;

b=np.transpose(training_vec)
K= training_vec*b
del b

K=K+box1

b=np.transpose(training_label)
a= training_label*b

H= np.multiply(a,K)
H=K

A=np.zeros([n+1,n+1])

A[0,1:n+1]=training_label
A[1:n+1,1:n+1]=H
A[1:n+1,0]=training_label
A1=A
baux=np.zeros([n+1,1])
baux[1:n+1,0]=np.ones([1,n])

supp_vecs= training_vec

ainv= np.linalg.inv(A)
A=np.transpose(baux); B=ainv; 

x = [[sum(a * b for a, b in zip(A_row, B_col)) 
                        for B_col in zip(*B)]
                                for A_row in A]


del ainv,B,baux,H,K,box1

bias=x[0][0]

#alpha=np.multiply(b  , np.transpose( x[0][1:(n+1)]))
alpha=np.multiply(np.transpose(b), np.transpose( x[0][1:(n+1)]))


########### Testing ############

test_sample=[(0.5,1.1)]

F= (np.transpose(supp_vecs   * np.transpose( test_sample)) * np.transpose( alpha)  )  +  bias

print(F)
 
if(F>0):
    print('class is 1')

if(F<0):
    print('class is -1')



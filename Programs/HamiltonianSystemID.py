#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 11:07:42 2023
Example 1:
    from HamiltonianSystemID import HamiltonianSystemID
    wr,w0,r,EMconst = HamiltonianSystemID(600)
@author: Fredy Vides, Department of Statistics and Research, CNBS, Honduras
"""
def HamiltonianSystemID(S):
    from HamiltonianSystem import HamiltonianSystem
    from K4SymProjector import K4SymProjector
    from SpaRCSim import SpaRCSim
    from matplotlib.pyplot import figure,show,grid,tight_layout
    from numpy import array, kron, identity, zeros
    from numpy.linalg import norm
    from time import time
    
    t,X = HamiltonianSystem()
    K4gen = array([[0,1,-1,0],[1,0,0,-1]])
    L = X.shape[0]
    s = X.shape[1]
    l = 5
    tp = 3
    nz = int(s*l+(s*l)*(s*l+1)/2+(s*l)**3+1)
    g1 = kron(K4gen[:,:s],identity(l))
    g2 = kron(K4gen[:,s:],identity(l))
    dn = int(((s*l)**(tp+1)-s*l)/(s*l-1)+1)
    G1 = zeros((dn,dn))
    G2 = zeros((dn,dn))
    G1[:s*l,:s*l] = g1
    G1[s*l:((s*l)**(tp-1)+s*l),s*l:((s*l)**(tp-1)+s*l)] = kron(g1,g1)
    G1[((s*l)**(tp-1)+s*l):-1,((s*l)**(tp-1)+s*l):-1] = kron(g1,kron(g1,g1))
    G1[-1,-1] = 1
    G2[:s*l,:s*l] = g2
    G2[s*l:((s*l)**(tp-1)+s*l),s*l:((s*l)**(tp-1)+s*l)] = kron(g2,g2)
    G2[((s*l)**(tp-1)+s*l):-1,((s*l)**(tp-1)+s*l):-1] = kron(g2,kron(g2,g2))
    G2[-1,-1] = 1
    t0 = time()
    w0,data,r = K4SymProjector(X.T,"svd",l,S,1,tp,nz,g1,g2,G1,G2,1e-3,1e-4)
    wr = w0@r
    print("Elapsed time: ",time()-t0)
    y = SpaRCSim(wr,data[:s*l,-1],tp,L-S+l-1)
    fig0 = figure()
    ax00 = fig0.add_subplot(1,2,1)
    ax00.plot(X[:S,0],X[:S,1],'b')
    grid(color='k', linestyle='--', linewidth=0.5)
    tight_layout()
    ax00 = fig0.add_subplot(1,2,2)
    ax00.plot(X[(S-l):,0],X[(S-l):,1],'g')
    ax00.plot(y[0,:].T,y[l,:].T,'r--')
    grid(color='k', linestyle='--', linewidth=0.5)
    tight_layout()
    show()
    fig0.savefig('fig_HamiltonianSystem_ID_1.png',dpi=600,format='png')
    EMconst = norm(g1@wr-wr@G1)+norm(g2@wr-wr@G2)+norm(g1@g2@wr-wr@G1@G2)
    return wr,w0,r,EMconst
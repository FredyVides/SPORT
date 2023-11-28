"""
Created on Fri Apr 29 19:30:41 2022
Financial competition simulation
Example: 
    from FinancialCompetitionSystem import FinancialCompetitionSystem
    t,z,wr,g1,G1,G,EMconst,r = FinancialCompetitionSystem(.376,1,1,1.1,31)    
@author: Fredy Vides, Department of Statistics and Research, CNBS, Honduras
"""

def FinancialCompetitionSystem(r,a,b,c,S):
    from numpy import zeros, ones, linspace, identity, kron, where, hstack, savetxt
    from numpy.linalg import norm
    from Z5SymProjector import Z5SymProjector
    from SpaRCSim import SpaRCSim
    from networkx import DiGraph, draw_networkx
    from matplotlib.pyplot import plot, show, xlabel, ylabel, axis, grid, figure
    
    L = 425
    l = 1
    s = 5
    t = linspace(0,14,L)
    z = zeros((s,L))
    tp = 2
    nz = int(s*l+(s*l)*(s*l+1)/2+1)
    
    z[:,0] = [0.9 , 0.95, 0.7, 0.55, .5]
    C = MatGen(a,b,c,s)
    
    for k in range(1,L):
        z[:,k:k+1] = system_rhs(z[:,k-1:k],r,C)
        
    savetxt('CompetitionModelData.csv', z.T, delimiter =',')
    
    g1 = identity(5)
    g1 = g1[:,[4,0,1,2,3]]
    
    G1 = zeros((s*l+(s*l)**2+1,s*l+(s*l)**2+1))
    G1[:s*l,:s*l] = g1
    G1[s*l:-1,s*l:-1] = kron(G1[:s*l,:s*l],G1[:s*l,:s*l])
    G1[-1,-1] = 1
    
    
    
    w0,data,r = Z5SymProjector(z,"svd",l,S,1,tp,nz,g1,G1,1e-3,1e-4)
    wr = w0@r
    y = SpaRCSim(wr,data[:s*l,-1],tp,L-S+l-1)
    
    nodes = []
    edges = []
    st = []
    tt = []
    for j in range(s):
        f0 = where(C[j,:]>0)[0]
        nodes.append(j)
        tt = hstack((tt,f0))
        st = hstack((st,j*ones(len(f0))))
    
    st = st.astype('int')
    tt = tt.astype('int')
    for k in range(len(st)):
        edges.append((st[k],tt[k]))
        
    G = DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    fig1 = figure()
    draw_networkx(G, node_color = 'green')
    show()
    fig1.savefig('fig_Financial_competition_graph.png',dpi=600,format='png')
    
    fig1 = figure()
    plot(t,z.T,'g'),plot(t[:(S-l)],z[:,:(S-l)].T,'b--'),plot(t[(S-l):],y.T,'r--')
    xlabel('t (months)')
    ylabel('Representation ranking')
    axis('tight')
    grid('on')
    show()
    fig1.savefig('fig_Financial_competition_dynamics.png',dpi=600,format='png')
    
    EMconst = 0
    h1 = identity(g1.shape[0])
    H1 = identity(G1.shape[0])
    for k in range(5):
        EMconst = EMconst + norm(g1@wr-wr@G1)
        h1 = h1@g1
        H1 = H1@G1
        
    
    return t,z,wr,g1,G1,G,EMconst,r

def MatGen(a,b,c,n):
    from numpy import diag,ones
    C = diag(ones(n-1),1)
    C[-1,0] = 1
    return a*C.T+b*diag(ones(n))+c*C

def system_rhs(s,r,C):
    return s + r*s*(1-C@s)
"""
Created on Fri Apr 29 15:27:32 2022
RCIdentifier  Reservoir computer simulator
   Code by Fredy Vides
   For Paper, "Computing Sparse Reservoir Computers"
   by F. Vides
@author: Fredy Vides
"""

def SpaRCSim(w,x0,tp,N):
    from numpy import zeros
    from NLMap import NLMap
    
    lx = len(x0)
    r = zeros((lx,N+1))
    r[:,0] = x0
        
    for k in range(N):
        r[:,k+1] = w@NLMap(r[:,k],tp)
                
    return r
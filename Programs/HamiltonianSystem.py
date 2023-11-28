"""
Created on Fri Apr 29 19:30:41 2022
Lorenz system simulator
@author: Fredy Vides
"""

def HamiltonianSystem(q0 = 1.0, p0 = 0.0):
    from numpy import arange
    from scipy.integrate import odeint
    
    def Hamiltonian(state, t):
        q, p = state
        
        dq = p**3-9*p
        dp = q**3-9*q
        return [dq, dp]
    
    y0 = [q0, p0]
    t = arange(0.0, 40.0, 0.01)
    
    z = odeint(Hamiltonian, y0, t, rtol = 1e-10, atol = 1e-10)
    
    return t,z
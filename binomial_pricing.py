import numpy as np
def gen_underlying(current_price, num_movements, up):
    size = num_movements+1
    S = np.array([0.]*(size**2)).reshape((size,size)) #Generate mxm matrix
    for u in range(size):
        for d in range(size):
            S[d,num_movements-u] =  current_price*(up**(u-d)) if u+d<size else 0
    return S
def gen_underlying_normal(current_price,num_movements,mean,variance):
    up = np.exp(np.sqrt(variance+mean**2))
    return gen_underlying(current_price,num_movements,up)

def gen_bond(current_price, num_steps,total_years,i):
    size = num_steps +1
    B = np.array([0.]*(size**2)).reshape((size,size))
    for u in range(size):
        for d in range(size):
            B[d,num_steps-u] = current_price*(1+i)**((u+d)*total_years/(num_steps)) if u+d<size else 0
    return B

def gen_call (strike, underlying, bond):
    m = underlying.shape[0]
    C = np.array([0.]*(m**2)).reshape((m,m))
    for x in range(m):
        C[x,x] = max(0,underlying[x,x]-strike)
    for y in range(m,0,-1):
        for x in range(y-1):
            Cu,Cd = (C[x,x+m-y],C[x+1,x+1+m-y])
            Su,Sd = (underlying[x,x+m-y],underlying[x+1,x+1+m-y])
            Bt = bond[x,x+m-y]
            #Represent call as synthetic mixture of cn_s stock and cn_b bond
            n_s = (Cu-Cd)/(Su-Sd)
            n_b = (Cu - Su*n_s)/Bt
            C[x,x+1+m-y] = n_s*underlying[x,x+1+m-y]+n_b*bond[x,x+1+m-y]
    return C

def gen_put (strike, underlying, bond):
    m = underlying.shape[0]
    P = np.array([0.]*(m**2)).reshape((m,m))
    for x in range(m):
        P[x,x] = max(0,-underlying[x,x]+strike)
    for y in range(m,0,-1):
        for x in range(y-1):
            Pu,Pd = (P[x,x+m-y],P[x+1,x+1+m-y])
            Su,Sd = (underlying[x,x+m-y],underlying[x+1,x+1+m-y])
            Bt = bond[x,x+m-y]
            #Represent call as synthetic mixture of cn_s stock and cn_b bond
            n_s = (Pu-Pd)/(Su-Sd)
            n_b = (Pu - Su*n_s)/Bt
            P[x,x+1+m-y] = n_s*underlying[x,x+1+m-y]+n_b*bond[x,x+1+m-y]
    return P

def delta(payoff,underlying,position):
    m= payoff.shape[0]
    u,d = position
    dV = payoff[d,m-1-(u+1)] - payoff[d+1,m-1-u]
    dS = underlying[d,m-1-(u+1)] - underlying[d+1,m-1-u]
    return dV/dS
def gamma(payoff,underlying,position):
    m = payoff.shape[0]
    u,d = position
    d2V = delta(payoff,underlying,(u+1,d)) - delta(payoff,underlying,(u,d+1))
    dS2 = underlying [d,m-1-(u+1)] - underlying[d+1,m-1-u]
    return (d2V/dS2)
def theta(payoff,underlying,position,h):
    m= payoff.shape[0]
    u,d = position
    dV = payoff[d+1,m-1-(u+1)] - payoff[d,m-1-u]
    dt = 2*h
    return dV/dt
def charm(payoff,underlying,position,h):
    m= payoff.shape[0]
    u,d = position
    dD = delta(payoff,underlying,(u+1,d)) - delta(payoff,underlying,(u,d+1))
    dt = 2*h
    return dD/dt
def delta_hedge(underlying, call, put,position):
    #Returns the necessary number of call and put options to short/long to delta hedge the underlying
    #in the format
    #[num_call
    # num_put]
    m = underlying.shape[0]
    u,d = position
    delta_call = delta (call, underlying,position)
    delta_put = delta(put,underlying,position)
    value_call = call[d,m-1-u]
    value_put =  put[d,m-1-u]
    matrix = np.array([[value_call,value_put],[delta_call,delta_put]])
    soln = np.array([[0],[-1]])
    return np.linalg.inv(matrix).dot(soln)

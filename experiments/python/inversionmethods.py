import numpy as np
import cvxpy as cp
import math
from numba import njit
#import numba_special
from numpy import diag, eye, sqrt, exp, pi
from numpy.linalg import norm, svd, eigh, inv, solve
from scipy.linalg import lu_factor, lu_solve
#from scipy.special import erf
from math import erf
import matplotlib.pyplot as plt

def generate_data(taus, omegas, x, noise_level, Nsamples, kernel='Gaussian', gamma=0.01):
    # This function takes in two linspaces and then returns a kernel
    # taus:
    # omegas:
    # x:
    # noise_level:
    # Nsample:
    # kernel:
    # gamma:
    
    Ntau, Nomega = len(taus), len(omegas)
    data = np.zeros((Nsamples,Ntau))
    EE, TT = np.meshgrid(omegas, taus)
    print("EE", np.shape(EE))
    print("TT", np.shape(TT))
    # Construct Kernel
    if (kernel=='Gaussian'):
        A=TT-EE 
        for i in range(Ntau):
            for j in range (Nomega):
                A[i,j]= np.exp(-(A[i,j]**2)/(2*gamma**2))
    #K =  sqrt(1/(2* pi*s**2)) * K # normalize

    if (kernel=='DoubleLaplace'):
        A=np.abs(TT*EE) 
        for i in range(Ntau):
            for j in range (Nomega):
                A[i,j]= np.exp(-A[i,j]/gamma)

    if (kernel=='Boson' or kernel=='Fermion'):
        # gamma is a temperature here 
        A=TT*EE 
        sgn = 1
        if (kernel=='Boson'):
            sgn= -1
        for i in range(Ntau):
            for j in range (Nomega):
                A[i,j]= np.exp(-A[i,j]) / (1 + sgn*np.exp( - EE / gamma ))
    
    if (kernel=='ScalarParticle'):
        # gamma is a temperature here 
        print("ATTENTION: You picked scalar particle make sure that" )
        print("ATTENTION: omega integration bounds are set to [0, inf)" )
        print("ATTENTION: kernel mltiplied by w**2" )
        A = TT*EE
        B = (1./gamma - TT)*EE
        for i in range(Ntau):
            for j in range (Nomega):
                # Finite Temp
                #A[i,j]= omegas[j]**2 * ( np.exp(-A[i,j]) + np.exp(- B[i,j]) ) / ( 1 - np.exp( - EE[i,j] / gamma ) )
                # Zero Temp
                A[i,j]= omegas[j]**2 *np.exp(-A[i,j]) 
    
    # Construct Data
    data0 = A @ x
    print(np.shape(data0))
    for i in range(Nsamples):
        noise = np.random.randn(Ntau,)
        data[i] = (1 + noise_level * noise * taus/(taus[1]-taus[0]) ) * data0
    return A, data0, data

def TIK_solve(A, b, lam = -1, cond_upperbound = 1000, proj_flag=0, MaxIt=1000, errtol=1e-8, Nmu=1000):
    """
    # Tikhonov Least Squares via SVD defined by the objective function
    # minimize     |Ax-b|^2_2 + lam*|x|^2_2
    Input:
        A: The matrix to be inverted
        b: The data
        lam: the regularization parameter, 
            default (-1) - Use Generalized Cross Validation (GCV) to set lam
            https://pages.stat.wisc.edu/~wahba/ftp1/oldie/golub.heath.wahba.pdf
            https://en.wikipedia.org/wiki/Ridge_regression
            OLS (0) - Remove Tikhonov Regularization  
        cond_upperbound: Cut off for maximum condition number of matrix. 
            default (1000) - double precision
        errtol: Check error convergence for projected Tikhonov algorithm. 
            default (1e-8) - single precision
        proj_flag: enforce positivity (algorithm 1) 
            default (0) - do not use projection step.
            project (1) - enforce that the solution must be positive.
            https://www.math.kent.edu/~reichel/publications/modmeth.pdf
    """
    [Ntau,Nomega] = np.shape(A) # Nt rows and Nomega cols
    x=np.zeros((Nomega,))
    U,S,Vh = svd(A); V=Vh.T # svd produce A = U@diag(S)@Vh, thus we have to transpose Vh to get V 
    for i in range(1,Ntau): # pre-condition the inversion to a set condition number
        if cond_upperbound > S[0]/S[i] :
            r=i+1
    
    if(proj_flag):
        print("TIK: You are using Projected TIK")
        V=V[:,0:r]; U=U[:,0:r]; S = S[0:r]
        mu = S[0]*S[r-1] # 
        # Pre compute quantities for computational savings
        vec=np.zeros((Nomega,))
        B=U.T@b # project b vector into column space
        for i in range(r):
            vec += B[i] * ( S[i]   / ( mu +  S[i]**2) ) * V[:,i]            
        M = V @ np.diag( (mu - S**2)/( mu + S**2) ) @ V.T 
        # fixed point iteration 
        y=np.zeros((Nomega,))
        for i in range(MaxIt):
            y = M@y + vec
            x = np.abs(y) + y
            if ( norm(A@x-b)<errtol):
                print("TIK: Projection fixed point iteration converged on iteration: ",i)
                break
            if(i==MaxIt-1):
                print("TIK: Projection fixed point iteration failed to converge!")

    else:
        # project b vector into column space
        B=U.T@b
        if (lam < 0): 
            # Compute the GCV using Wahba method for setting regularizer mu
            mus=np.logspace(-16, 4, num=Nmu)
            GCV=np.zeros((Nmu,1))
            print("TIK: You are using Generalized Cross Validation Tikhonov")
            for j in range(Nmu):
                tmp = np.zeros((3,1)); mu=mus[j]
                for i in range(r):
                    tmp[0]+=mu**2/(S[i]**2+mu)**2*B[i]**2
                    tmp[2]+=mu/(S[i]**2+mu)
                for i in range(r,Ntau):
                    tmp[1]+=B[i]**2
                GCV[j]=(tmp[0]+tmp[1])/tmp[2]**2
                if (j == 0):
                    lower=GCV[j]
                    GCVI=0
                if (GCV[j] < lower):
                    lower=GCV[j]
                    GCVI=j
                #elif (GCV[j] > lower):
                #    break
            lam=mus[GCVI]  
        # exress x as a windowed sum of the column space basis vectors,
        # https://math.stackexchange.com/questions/2065192
        for i in range(r):
            x += B[i] * ( S[i]   / ( lam +  S[i]**2) ) * V[:,i]

    return x

def TIK_solve_via_dual(K, G, alpha=-1, R=None, cond_upperbound=1000):
  """
  Input:
    K: (m,d) numpy array
    G: (m,1) numpy array
    alpha: postive scalar
    R: (d,d) symmetric positive definite numpy array  (default of None corresponds to R = identity)
  Output:
    (d,1) numpy array --- "recovered probability vector"
  """
  m, d = K.shape
  y = cp.Variable(m)

  # --- Set value of alpha using GVC --- #
  U,S,Vh = svd(K); V=Vh.T 
  # pre-condition the inversion to a set condition number
  for i in range(m):
      if cond_upperbound > S[0]/S[i] :
          r=i+1
  B=U.T@G
  if (alpha < 0): 
      # Compute the GCV using Wahba method for setting regularizer mu
      Nmu=1000; 
      mus=np.logspace(-16, 4, num=Nmu)
      GCV=np.zeros((Nmu,1))
      print("Dual TIK: You are using Generalized Cross Validation Tikhonov")
      for j in range(Nmu):
          tmp = np.zeros((3,1)); mu=mus[j]
          for i in range(r):
              tmp[0]+=mu**2/(S[i]**2+mu)**2*B[i]**2
              tmp[2]+=mu/(S[i]**2+mu)
          for i in range(r,m):
              tmp[1]+=B[i]**2
          GCV[j]=(tmp[0]+tmp[1])/tmp[2]**2
          if (j == 0):
              lower=GCV[j]
              GCVI=0
          if (GCV[j] < lower):
              lower=GCV[j]
              GCVI=j
          #elif (GCV[j] > lower):
          #    break
      alpha=mus[GCVI] 
            
  fid = (1/2*alpha) * cp.sum_squares(y) - y.T @ G

  if R == None:
    reg = 0.5 * cp.sum_squares(K.T @ y)
  else:
    R_inv = np.linalg.inv(R)
    reg = 0.5 * cp.sum_squares(R_inv @ K.T @ y)

  obj = cp.Minimize(fid + reg) # minimize the dual objective function
  prob = cp.Problem(obj)
  prob.solve()

  if R == None:
    sol = K.T @ y.value
  else:
    sol = R_inv @ R_inv @ K.T @ y.value

  sol = sol / np.sum(sol) # we need convert to a prob vector

  print("Solution:", prob.status)

  return sol

@njit # just in time compile to speed up program
def BGM_solve(A, b, C, omegas):
    # Backus-Gilbert Method 
    # lam: regularization bameter
    # This code uses Hansen, Lupo, Tantelo https://doi.org/10.1103/PhysRevD.99.094508
    # But it uses the unsmeared/original method not the new method proposed in the paper.
    
    [Ntau,Nomega] = np.shape(A) # Nt rows and Nomega cols
    q_list=[];
    obj_list=[];
    lams = np.linspace(1e-2,.99,98)
    x=np.zeros((Nomega,))
    R=A@np.ones((Nomega,1)) #Row Sum
    for i in range(Nomega):
        W=np.zeros((Ntau,Ntau))
        for j in range(Ntau):
            for k in range(Ntau):
                for l in range(Nomega):
                    W[j,k]+=(l-i)**2 * A[j,l]*A[k,l]; # eqn 17
                    #W(j,k)= W(k,j);?
        #Q(i,:)=q;
        # Full solution
        #q=( inv(W+lam*Sinv) @R ) / (R.T @ inv(W+lam*Sinv) @ R); # eqn 22
        # numerical recipes approach
        #scan for value of lambbda which optimizes the functional (1-lam)*W + lam*C
        objfxn=[];
        for lam in lams:
            y = solve( (1-lam)*W + lam*C, R)
            q = y / (R.T @ y ); # eqn 22
            val = (1.-lam)*np.sum( (omegas - omegas[i])**2 * (q.T @ A)**2) + lam*q.T @ C @ q
            #print( np.shape( (omegas - omegas[i])*(q.T @ A) ), np.shape(q.T @ C @ q), np.shape(val) )
            objfxn.append(val[0,0])
        obj_list.append(objfxn)
        minval = objfxn[0]
        for index in range( len(objfxn) ):
            if ( objfxn[index] <= minval):
                minval = objfxn[index]
                minindex = index
        print(lams[minindex])
        y = solve( (1-lams[minindex])*W + lams[minindex]*C, R)
        q = y / (R.T @ y );
        q_list.append(q)
        #print("q.T@R", q.T@R); #should be normalized - use to check if working
        x[i]=(b.T @ q)[0];
    return x, q_list, obj_list

@njit # just in time compile to speed up program
def SmearedBGM_solve(A, b, Cov, omegas, sigma=.2):
    # https://journals.aps.org/prd/abstract/10.1103/PhysRevD.99.094508
    # lam: regularization bameter
    # This code uses Hansen, Lupo, Tantelo https://doi.org/10.1103/PhysRevD.99.094508
    # But it uses the unsmeared/original method not the new method proposed in the paper.
    
    [Ntau,Nomega] = np.shape(A) # Nt rows and Nomega cols
    x=np.zeros((Nomega,))
    R=A@np.ones((Nomega,1)) #Row Sum of Kernel
    obj_list=[];
    g_list=[]; 
    lams = np.linspace(1e-2,.99,98)
    for i in range(Nomega):
        W=np.zeros((Ntau,Ntau))
        for j in range(Ntau):
            for k in range(Ntau):
                for l in range(Nomega):
                    W[j,k]+= A[j,l]*A[k,l] * (l-i)**2 ; # eq 32

        objfxn=[];
        Z = 1./2. * ( 1. + erf(omegas[i] / (sqrt(2.)*sigma)) ) # eq A2
        Smearing = 1./( sqrt(2*pi)*sigma*Z ) * exp( -(omegas - omegas[i])**2 / (2*sigma**2) ); # eq A1
        f = A@np.reshape(Smearing, (-1,1)) # eq 30
        for lam in lams:
            z = solve( (1-lam)*W + lam/(b[0]**2)*Cov, (1-lam)*f)
            y = solve( (1-lam)*W + lam/(b[0]**2)*Cov, R)
            g = z + y*( 1 - R.T@z) / (R.T @ y ); # eqn 22
            val = (1.-lam)*np.sum( (Smearing - g.T @ A)**2 ) + lam/(b[0]**2)*g.T @ Cov @ g
            objfxn.append(val[0,0])
        obj_list.append(objfxn)
        minval = objfxn[0]
        for index in range( len(objfxn) ):
            if ( objfxn[index] <= minval):
                minval = objfxn[index]
                minindex = index
        print("smeared", lams[minindex])
        z = solve( (1-lams[minindex])*W + lams[minindex]*Cov / b[0]**2, (1-lams[minindex])*f)
        y = solve( (1-lams[minindex])*W + lams[minindex]*Cov / b[0]**2, R)
        g = z + y*( 1 - R.T@z) / (R.T @ y ); # eqn 22
        g_list.append(g)
        x[i]=(b.T @ g)[0];
    return x, g_list, obj_list

def KL_via_dual(K, G, rho, alpha, cond_upperbound):
    """
    This code takes the dual approach to solve primal problem
    min || K @ sol - G ||^2 + alpha_1 * KL( sol | m)
    Input:
        K:   (m,d) numpy array, kernel to be inverted
        G:   (m,1) numpy array, convolved data
        rho: (d,1) numpy array, prior (strictly postive entries and in unit simplex)
        alpha: postive scalar
    Output:
        sol: (d,1) solution, numpy array --- "recovered probability vector"
    """
    #if (np.sum(rho) != 1 ):
    #    print("not probability vector!")
    #    rho /= np.sum(rho)
    if (np.any(rho <= 0) ):
        print("rho less than zero!")
        rho[ rho <0 ] = 1e-2

    m, d = K.shape
    V, S, Uh = svd(K); U=Uh.T 
    for i in range(m):
        if cond_upperbound > S[0]/S[i] :
            r=i+1
    V=V[:,0:r]; U=U[:,0:r]; S = diag(S[0:r])
    K_reduced = V@S@U.T
    #K_reduced = np.copy(K)
    y = cp.Variable((m,1))
    fid = (1./2.) * cp.sum_squares(y) - y.T @ G
    b = (1./alpha) * K_reduced.T @ y + np.log(rho)
    reg = alpha*cp.log_sum_exp(b)

    obj = cp.Minimize(fid + reg) # minimize the dual objective function
    prob = cp.Problem(obj)
    prob.solve( solver=cp.SCS) #, verbose=True )

    w = np.exp(K_reduced.T @ y.value )
    sol =  (rho * w) / rho.T @ w
    sol = sol / np.sum(sol) # we need convert to a prob vector

    print("Solution:", prob.status)
    return sol
    
@njit
def calcerror(A, b, eigvals_inv_matrix, m, U, y):
    x = m*np.exp(U @ y) # combining eqns (3.33) a = Uy and (3.29) x = m * exp(a) 
    return np.sum((A@x-b).T @ eigvals_inv_matrix @ (A@x-b))

@njit
def Bryans_alg(A, b, C, m, alpha, cond_upperbound, MaxIt, MaxFail):
    """
    Bryan's Algorithm published - https://link.springer.com/article/10.1007/BF02427376
    This code follows the convention of Asakawa et al. https://arxiv.org/abs/hep-lat/0011040
    Input:
      A: matrix to be inverted.
      b: data
      C: data convariance matrix.
      m: default model or prior.
      alpha: weight of the entropic prior.
      cond_upperbound: Cut off for maximum condition number of matrix. 
         default (1000) - double precision
      MatIt: maximum number of Levenberg Marquart steps to take
         default () - 
      MaxFail: maximum number of failed attempted Levenberg-Marquart steps.
         default () - 
    Output:
        x: solution
    """
    [Ntau,Nomega] = np.shape(A)
    # Decompose the convariance matrix
    # see https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html#numpy.linalg.eigh for conventions 
    eigvals, R = eigh(C) # C = R Lambda R.T 
    eigvals_inv_matrix = diag(1/eigvals)

    # Rotate into diagonal covariance space (eqn 3.39)
    Arot = np.copy(R.T @ A)
    brot = np.copy(R.T @ np.reshape(b, (Ntau,1)) )
    m = np.copy(np.reshape(m, (Nomega,1)))
    
    # Precompute the M for computational savings (eqn 3.37)
    # svd produce A = V@diag(S)@Uh, thus we have to transpose Vh to get V 
    # this convention disagrees with all the other conventions, but it is 
    # how Asakawa writes it (see the text below eqn 3.30).
    V,S,Uh = svd(Arot); U=Uh.T 
    # pre-condition the inversion to a set condition number
    for i in range(Ntau):
        if cond_upperbound > S[0]/S[i] :
            r=i+1
    V=V[:,0:r]; U=U[:,0:r]; S = diag(S[0:r])
    #DDL = eigvals_inv_matrix 
    M = S @ V.T @ (eigvals_inv_matrix) @ V @ S

    y = np.zeros((r,1))
    err_list=[ calcerror(Arot, brot, eigvals_inv_matrix, m, U, y) ];
    
    #initialize algorithm constants
    up=1e-3; c0=.2; i=0;
    while (i < MaxIt):
        errflag=1; fail=0; mu=0; i+=1
        
        # ------------------------------------------------------- #
        # Compute RHS of eqn 3.36, where b has been relabeled `y' #
        # ------------------------------------------------------- #
        x = m*np.exp(U @ y) # eqn (3.29) x = m * exp(a) & eqn (3.33) a = Uy 
        DL = eigvals_inv_matrix @ ( Arot @ x - brot ) # derivative of Likelihood (L) w.r.t. A@x
        g = S @ V.T @ DL # compute g from 3.34 
        RHS = -alpha * y - g # compute the RHS of eqn 3.36

        # ---------------------------- #
        # Levenberg-Marquart iteration #
        # ---------------------------- #
        T = U.T @ diag(x.flatten()) @ U # pre-compute T from eqn 3.37
        while ( errflag and (i <= MaxIt) ): 
            LHS = (alpha + mu) * eye(r) + M @ T # compute the LHS of eqn 3.36
            delta = solve(LHS,RHS) # leave it up to numpy on how to solve it
            new_y = y + delta
            err = calcerror(Arot, brot, eigvals_inv_matrix, m, U, new_y)
            #print("err", delta.T @ T @ delta, c0)
            #errflag = (delta.T @ T @ delta < c0 ) # eqn 3.38
            errflag = (err_list[-1] < err) # my prefered error!
            if ( errflag ): #unsuccessful update - raising mu
                mu*=up;
                fail+=1;
            else:
                mu=0
                i+=1;
            if (fail > MaxFail):
                #print("alpha: %.3e"%(alpha) + " reached fail limit");
                break
        if (fail > MaxFail): #fail limit reached close program
            break
        else:  #update
            err_list.append(err);
            y=new_y;
    
    x = m*np.exp(U @ y); # eqn (3.29) x = m * exp(a) & eqn (3.33) a = Uy
    #x /= np.sum(x)
    return x, err_list


def MEM_solve(A, b, C, m, alpha_min=1e-7, alpha_max=1e2, Nalpha=50, cond_upperbound = 1e6, MaxIt=1000, MaxFail=1000, dual=0):
    # Nalpha: Total number of alphas to check
    plt.figure()
    x_list = [];
    P_list = [];
    alphas = np.geomspace(alpha_min, alpha_max ,Nalpha)
    N = len(alphas); 
    for alpha, num in zip( alphas, range(len(alphas)) ):
        if(dual):
            x = KL_via_dual(A, b, m, alpha, cond_upperbound)
        else:
            x, err_list = Bryans_alg(A, b, C, m, alpha, cond_upperbound, MaxIt, MaxFail)
        if (not any( np.isnan(x) )): # Check if there are NaN's in the reconstruction 
            x_list.append(x); # if no NaN's save the reconstruction
            xtmp= np.reshape(x,(-1,1))
            btmp= np.reshape(b,(-1,1))
            L = 1./2. * ( A@xtmp - btmp ).T @ ( A@xtmp - btmp ) # eqn 3.6
            tmp = np.nan_to_num(x*np.log(x/m), np.nan==0.0, np.inf==0.0, -np.inf==0.0)
            S = - np.sum(tmp) # eqn 3.13 assuming that A and m are normalized 
            LAMBDA = A.T @ C @ A
            tmp = np.sqrt(x)
            for i in range(len(x)):
                for j in range(len(x)):
                    LAMBDA[i,j] = tmp[i]*LAMBDA[i,j]*tmp[j] 
            #test = np.sqrt(x) *  (A.T@ C @ A) * np.sqrt(x).T # eqn 3.20 - A.T@C@A is the 2nd derivative of L w.r.t. A
            #print("LAMBDA", np.allclose(LAMBDA,test))
            L_eigvals, trash = eigh(LAMBDA)
            # Compute eqn 3.19  
            PalphaHm= 1 #1./alpha # Jeffery's Rule
            Palpha = PalphaHm*np.exp (0.5*np.sum( np.log(alpha / (alpha + L_eigvals))) + alpha*S - L)
            # Print out the cost functions to see where the major contributions are.
            print("P: %.2e | term 1, a*S, L: %.2e %.2e %.4e"%(Palpha, 0.5*np.sum( np.log( alpha / (alpha + L_eigvals))), alpha*S, L) )
            #print("P(alpha)", Palpha)
            plt.plot(x)
            P_list.append([alpha, Palpha])
    P_list = np.array(P_list)
    P_list[:,1] = P_list[:,1] / np.sum(P_list[:,1])
    x_avg = x_list[0]*P_list[0,1]
    for i in range(1, len(P_list)):
        x_avg += x_list[i]*P_list[i,1]
    return x_avg, P_list

def SD_solve( A, b, cond_upperbound=1e3, tol=1e-16, proj_flag=0, MaxIt=10000 ):
    # Projected Steepest Descent - steepest descent can take a long time to converge. 
    # stopping the algorithm early is a form of regularization here.
    # tol: the convergence tolerance.
    # proj_flag: project each steepest descent step into positive function space.
    # Max_It: maximum nuber of gradient descent steps that can be taken.
    
    # if you have time to implement a more general optimization algorithm
    # https://www.sciencedirect.com/science/article/pii/S0895717705005297
    
    [Ntau,Nomega] = np.shape(A) # Nt rows and Nomega cols
    b = np.reshape(b, (Ntau,1))
    x = np.ones((Nomega,1))
    # svd produce A = U@diag(S)@Vh, thus we have to transpose Vh to get V 
    U,S,Vh = svd(A); V=Vh.T 
    # pre-condition the inversion to a set condition number
    # Assess which singular vectors will be too sensitive to noise. 
    for i in range(Ntau):
        if cond_upperbound > S[0]/S[i] :
            r=i
    A_reduced = U[:,0:r+1] @ diag(S[0:r+1]) @ V[:,0:r+1].T 

    # precompute quantities for computational speed
    tmp0 = A_reduced.T @ A_reduced
    tmp1 = A_reduced.T @ b 
    # initialize residual vector
    r= tmp0 @ x - tmp1
    err = [];
    for i in range(MaxIt):
        z= ( norm(r)/ norm(A_reduced @r) )**2
        x -= z*r 
        r = tmp0 @ x - tmp1 
        err.append(norm(r))
        if (proj_flag): # positivity projection step
            x[ x < 0.0 ] = 0.0
        if ( norm(A_reduced @ x - b)<tol):
            print("SD converged iteration: ",i)
            break
        if(i==MaxIt-1):
            print("SD failed to converge!")
    return x, err

def CG_solve( A, b, tol=1e-16, flag=0, MaxIt=0 ):
    # Conjugate Gradient Method - cite painlessCGM
    # Regularization is introduced by the 'tol' term.
    # At min(Nomega,Ntau) iterations sol converges to the OLS (unregularized)
    # Defaults: Algorithm stops when MSE agrees at double precision.
    [N,M] = np.shape(A)
    b = np.reshape(b, (N,1))
    x = np.ones((M,1))
    if(MaxIt==0):
        MaxIt=np.fmax(N,M);
    tmp0 = A.T @ A
    tmp1 = A.T @ b
    r = tmp1 - tmp0 @ x
    p = np.copy(r)
    
    err=[];
    for i in range(MaxIt):
        alph =  norm(r)**2 / ( r.T @ tmp0 @ p );
        x +=alph*p;
        r -=alph*tmp0 @ p;
        beta =-1*(r.T@tmp0@p) / (p.T@tmp0@p) ;
        p    = r+beta*p;
        err.append(norm(r))
        if ( norm(A@x-b) <tol):
            print("CGM converged iteration: ",i ,  norm(A*x-b))
            break
        if(i==MaxIt-1):
            print("CGM failed to converge!")
    return x, err

# Methods Coming ?soon?
# !! priority !! Smooth Bayesian Reconstruction Method https://arxiv.org/abs/1705.03207
# Pade Approach Tripolt https://www.sciencedirect.com/science/article/pii/S0010465518304041?via%3Dihub
# Shu-Ding Stochastic Approach https://arxiv.org/abs/1510.02901
# Variational Method https://iopscience.iop.org/article/10.1088/0954-3899/36/6/064027
#                    https://journals.aps.org/prd/abstract/10.1103/PhysRevD.84.094504



import numpy as np

def KL(x, mu): 
  # assume mu is postive 
  if np.sum(x) > 1.01 or np.sum(x) < 0.99 or np.any(x<0) == True:
    return np.inf
  x_ = x[x>0]
  mu_ = mu[x>0]
  return np.dot(x_, np.log(x_ / mu_))


def set_x(A, mu, y): 
  a = A.T @ y
  M = np.max(a) # to prevent overflow
  x = np.diag(mu) @ np.exp(a - M)
  x = x / np.sum(x) # x is always in the unit simplex as enforced by the optimality conditions 
  return x


def dual_val(A, mu, b, lmda, y):
  return  (lmda /2) * np.linalg.norm(y) ** 2 + np.log(np.dot(mu, np.exp(A.T @ y))) - np.dot(b,y) # d(y) in fact the negative of the dual objective viewed from a max POV


def primal_val(A, mu, b, lmda, x):
  return (1/(2*lmda)) * np.linalg.norm(A @ x - b) ** 2 + KL(x, mu)





#def GlobalNewton(truth, A, mu, b, lmda=10e-12, eps=10e-5, max_iters=100, verbose=True, rho=1e-11, p=2.1, beta=0.1, s= 0.25, testing=False):
def GlobalNewton(A, mu, b, lmda=10e-12, eps=10e-5, max_iters=100, verbose=True, rho=1e-11, p=2.1, beta=0.1, s= 0.25, testing=False):
  m, n = A.shape
  y = np.zeros(m)
  x = set_x(A, mu, y) # we think of x simply as a function of y useful in computing d_grad and d_hess
  print("first set_x just happened.")
  # but x here happens to converge to the solution of the primal as y converges to a sol of min d(y)
  
  gaps = []
  grad_steps = []
  d_grad = A @ x + lmda * y - b # Jacobian of d(y)
  k = 0

  Gnorm = np.linalg.norm(d_grad)
  Gnorms = []

  while Gnorm > eps and k < max_iters:

    if lmda != 0:
      gap = primal_val(A, mu, b, lmda, x) + dual_val(A, mu, b, lmda, y)
      gaps.append(gap)
  
    if k % 1000 == 0 and verbose == True:
      print('-' * 10)
      print(f'Iteration: {k+1}')
      if lmda != 0:
        print(f'Primal value: {primal_val(A, mu, b, lmda, x)}')
        print(f'Dual value: {dual_val(A, mu, b, lmda, y)}')
        print(f'Primal-Dual Gap: {gap}')
      print(f'Iter: {k+1}, Dual Jacobian Norm: {np.linalg.norm(d_grad)}')

    # Sol to Newton equation
    d_hess = A @ (np.diag(x) - x @ x.T) @ A.T + lmda * np.identity(m)
    d = np.linalg.solve(d_hess, -d_grad) # d_hess is postive definite thus will always have a sol


    if np.dot(d_grad, d) > -rho * np.linalg.norm(d) ** p: # check if we have sufficient decrease wrt the norm of d
      if k % 1000 == 0 and verbose == True:
        print(f'Insuff decrease: grad step taken')
        print(f'dk norm {np.linalg.norm(d)}')
        print(f'dot prod {np.dot(d_grad, d)}')
      grad_steps.append(k)
      d = -d_grad # Take a gradient step
    t = 1
    
    while dual_val(A, mu, b, lmda, y + t*d) > dual_val(A, mu ,b, lmda, y) + (t*s) * np.dot(d_grad, d):
      t *= beta

    y = y + t*d

    x = set_x(A, mu, y)
    d_grad = A @ x + lmda * y - b
    k += 1
    Gnorm = np.linalg.norm(d_grad)
    Gnorms.append(Gnorm)

  if testing == True:
    return {np.linalg.norm(truth - x) / np.sqrt(n)}

  print('-' * 10)
  print("Summary:")
  print(f"Total iters: {k}")
  print(f"Grad steps: {len(grad_steps)}")
  print(f"Jacobian Norm: {Gnorms[-1]}")
  if lmda != 0:
    print(f'Primal-Dual Gap: {gap}')
    print(f'Primal value: {primal_val(A, mu, b, lmda, x)}')
    print(f'Dual value: {dual_val(A, mu, b, lmda, y)}')
  #print(f'RMS(x-x0): {np.linalg.norm(truth - x) / np.sqrt(n)}')
  print(f'RMS(Ax-b): {np.linalg.norm(A @ x - b) / np.sqrt(m)}')

  return x #, y, gaps, grad_steps, Gnorms


# mu = np.ones(x0.shape[0]) / x0.shape[0] # set uniform prior mu
# m, n = A.shape
# lmda=0

# x, y, gaps, armijo_steps, Gnorms = GlobalNewton(truth=x0, A, mu, b, lmda=0, eps=1e-6, max_iters=1000, 
#                                                 verbose=True, rho=1e-14, p=1.0001, beta=0.11, s= 0.25)
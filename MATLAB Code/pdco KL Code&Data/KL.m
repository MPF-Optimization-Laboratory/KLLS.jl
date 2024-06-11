function [obj,grad,hess] = KL(x, p)
%        [obj,grad,hess] = KL(x,p)
%        computes the objective value, gradient and diagonal Hessian
%        of a separable convex function, for use with pdsco.m.
%        This is an example objective function.

% Nick: The original entropy function did not handle the case where
% values were not in the unit simplex and so we omit this case as well

   logx = log(x);
   logp = log(p);

   obj  = sum( x.*logx );
   grad = 1 + logx - logp;
   hess = 1 ./ x;
   hess = diag(sparse(hess));  % Turn diag into a matrix for pdco2.

%-----------------------------------------------------------------------
% End function entropy
%-----------------------------------------------------------------------

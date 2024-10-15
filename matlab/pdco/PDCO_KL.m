function [x,y,z,inform,tracer_test] = PDCO_KL(A, b, mu, d2)
    % Matt Edit 10/15: added scaling via d2 to inputs
    % deleted printing.
    % d2 commented below recovers usual setup
    
    print = 0;

    A(end+1, :) = 1; % enforce simplex constraint
    b(end+1, :) = 1;

    d2(end+1,:) = 10e-3; % default value of delta, chosen by matt.
    % this is simplex constraint tolerance
    

    [m, n] = size(A);
     %{
    % these choices of d2 and d1 give the standard framework

    d2 = sqrt(lmda) * ones(m, 1);
    d2(end) = delta;

     %}

    d1=0;

    % ---------------------------------------
    
    %The following two code sets the appropriate parameters 
    % to run pdco on the KL problem. The parameter choices are 
    % largely copied from those set by Saunders in his implementation 
    % of pdco for the entropy function

    xsize = 100/n;               % A few elements of x are much bigger than 1/n.
    xsize = min(xsize,1);      % Safeguard for tiny problems.
    zsize = 1; 
    x0min = xsize;             % Applies to scaled x1, x2
    z0min = zsize;             % Applies to scaled z1, z2
    
    en    = ones(n,1);
    bl    = zeros(n,1); % lower bound on x
    bu    = ones(n, 1); % upper bound on x
    
    x0    = en*xsize;        
    y0    = zeros(m,1); 
    z0    = en*z0min;       
    
    options = pdcoSet;

    % MODIFICATION MADE BELOW
    options.MaxIter      =   1000;
    options.FeaTol       =  1e-10; %ORIGNIALLY 1e-6
    options.OptTol       =  1e-10; %ORIGNIALLY 1e-6

    options.x0min        = x0min;  % This applies to scaled x1, x2.
    options.z0min        = z0min;  % This applies to scaled z1, z2.
    options.mu0          =  1e-0;  % 09 Dec 2005: BEWARE: mu0 = 1e-5 happens
                                   %    to be ok for the entropy problem,
                                   %    but mu0 = 1e-0 is SAFER IN GENERAL.
    options.Method       =     3;  % 1=Chol  2=QR  3=LSQR
    options.LSMRatol1    =  1e-3;
    options.LSMRatol2    =  1e-6;
    options.wait         =     1;
    
    % ---------------------------------------

    options.Print = print;
    options.wait = 0;


    [x,y,z,inform,tracer_test] = pdco(@KL, A, b, bl, bu, d1, d2,options,x0,y0,z0,xsize,zsize, mu); 

end


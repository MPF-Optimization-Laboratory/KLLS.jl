function [x] = EMDA(A, b, lmda, num_iters, plt, L)
% solves min over x in unit simplex of 1/(2 lmda) |Ax-b|^2
    
    [~, n] = size(A);

    values = zeros(num_iters);

    x = ones(n, 1) / n; % starting point in int \Delta_n


    % uncomment below lines to obtain the correct Lipschitz constant
    %s = svds(A, 1, "largest");
    %L = (1/lmda) * (s^2 + norm(A'*b)); 

    
    for k = 1:num_iters
        
        % From the paper, L should be the Lipschitz constant of f(x)
        % used in the learning rate t. However, it seems to be too large
        % and yields no objective decrease. So, we take L as an input.

        t = sqrt(2*log(n)) / (L*sqrt(k));

        grad = (1 / lmda) * (A'*A*x - A'*b);

        u = - t * grad;

        M = max(u);

        v = exp(u - M);

        z = x .* v;

        x = z ./ sum(z);

        values(k) = norm(A*x - b);

    end

    if plt
        plot(1:num_iters, values);
    end

   

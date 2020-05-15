function [u_min, J_min] = conj_grad(A, c, u0, eps, J, grad)
    % A, c - parameters of quadratic minimization
    % J, grad - parameters in jeneral case 

    n_max = 1000;
    dim = numel(u0);
    u = u0;
    n = 0;
    
    if nargin == 4
        J = @(x) c' * x + x' * A * x;
        grad = @(x) c + (A + A') * x;
        p = -grad(u0);
        while (sq_norm(grad(u)) >= eps^2) && (n < n_max)
             u_prev = u;
             alpha = - (grad(u_prev)' * p) / (2 * p' * A * p);
             u = u_prev + alpha * p;
             gu = grad(u);
             beta = (gu' * A * p) / (p' * A * p);
             p = -gu + beta * p;
             n = n + 1;
             if dim == 2
                 plot([u_prev(1) u(1)], [u_prev(2) u(2)],'o--g');
                 hold on;
             end
        end
    else
        p = -grad(u0);
        while (sq_norm(grad(u)) >= eps^2) && (n < n_max)
             u_prev = u;
             options = optimoptions(@fminunc,'Display','off');
             alpha = fminunc(@(x) J(u_prev + x * p), 0, options);
             u = u_prev + alpha * p;
             gu = grad(u);
             beta = sq_norm(gu) / sq_norm(grad(u_prev));
             p = -gu + beta * p;
             n = n + 1;
             if dim == 2
                 plot([u_prev(1) u(1)], [u_prev(2) u(2)],'o--g');
                 hold on;
             end
        end     
    end
    

    if n == n_max
        disp("Algorithm diverged");
        u_min = [];
        J_min = [];
        return;
    else
        u_min = u;
        J_min = J(u);
        disp(['Algorithm converged in ', num2str(n), ' iteration(s)']);
        disp("u* = ");
        disp(u_min);
        disp("J* = ");
        disp(J_min);
        return;
    end
end
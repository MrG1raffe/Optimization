function [u_min, J_min] = cond_grad(A, c, rho, u0, eps, J, grad)
    % [val; point] = rho(l) - support function of U
    % A, c - parameters of quadratic minimization
    % J, grad - parameters in jeneral case 
    % u0 - initial approximation
    % eps - J tolerance
    n_max = 1000;
    dim = numel(u0);
    u = u0;
    u_prev = u0;
    n = 0;
    if nargin == 5
        J = @(x) c' * x + x' * A * x;
        grad = @(x) c + (A + A') * x;
        while n == 0 || (abs(J(u) - J(u_prev)) >= eps && n < n_max)
            u_prev = u; 
            [~, v] = rho(-grad(u_prev));
            delta = v - u;
            if prod(delta == 0)
                break;
            end
            if prod(A == 0, 'all')
                alpha = 1;
            else
                alpha = min(1, -(grad(u_prev)' * delta) / (2 * delta' * A * delta));
            end
            u = u_prev + alpha * delta;
            n = n + 1;
            if dim == 2
                plot([u_prev(1) u(1)], [u_prev(2) u(2)],'o--r');
                hold on;
            end
        end
    else
        while n == 0 || (abs(J(u) - J(u_prev)) >= eps && n < n_max)
            u_prev = u;
            [~, v] = rho(-grad(u_prev));
            delta = v - u;
            if prod(delta == 0)
                break;
            end
            alpha = fminbnd(@(x) J(u_prev + x * delta), 0, 1);
            u = u_prev + alpha * delta;
            n = n + 1;
            if dim == 2
                plot([u_prev(1) u(1)], [u_prev(2) u(2)],'o--r');
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
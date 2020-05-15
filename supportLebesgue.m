function res = supportLebesgue(f, opts, n)
    res = @rho;
    function [val, point] = rho(y)
        nonlcon = @(x) const(f(x));
        maxim = fmincon(@(x) -(y' * x), zeros(n, 1), [], [], [], [], [], [], nonlcon, opts);
        val = y' * maxim;
        point = maxim;
    end
end
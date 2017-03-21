function [x_avg] = stoc_gd(samples, labels, gamma, max_it, d_bound)
% initialization
[d,N] = size(samples);
x = zeros(d,1);
x_avg = zeros(d,1);
rnd_pm = randperm(N);

for k = 1:max_it

    % randomly choose a sample
    idx = rnd_pm(mod(k,N)+1);
    sample = samples(:,idx);
    label  = labels(idx);

    % Stochastic Gradient Descent
    eta = d_bound / sqrt(k);
    x = (1-eta*gamma)*x - eta * feval(@sub_grad, x, sample, label);

    % averaging x
    x_avg = x_avg * (k - 1);
    x_avg = x_avg + x;
    x_avg = x_avg/k;
end
end
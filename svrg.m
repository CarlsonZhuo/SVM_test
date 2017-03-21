function [x] = svrg(samples, labels, gamma, max_it, mb)

if nargin < 5
    mb = 10;  
end 

% initialization
[d, N] = size(samples);
x      = zeros(d,1);
xold   = x;
rnd_pm = [randperm(N)];
m      = fix(2*N/mb);
eta    = 0.35;  
gold   = zeros(d,1); 
max1   = fix(max_it/(2*N))+1;

for k = 1:max1
    
    if k > 0     
        gold = feval(@full_grad, xold, samples, labels); 
        gold = gold/N;
    end
    numx = 0;
    xx = zeros(d,1);
    
    for j = 1:m 
        
        % randomly choose a sample
        idx = rnd_pm(mod(k,N)+1);
        sample = samples(:,idx);
        label  = labels(idx);

        %%% randomly choose minibatch   
        idx = ceil(N*rand);
        if idx <= N-mb+1
            ix = idx:idx+mb-1;
        else
            ix = [1:(idx+mb-N-1), idx:N];
        end
        I = sort(rnd_pm(ix));
        sample = samples(:,I);
        label  = labels(I);    

        % Gradient
        if mb > 1            
            if k==1
                gg = feval(@msub_grad, x, sample, label);
                gg = gg/mb;
            else
                gg = feval(@msub_grad,x,sample,label) - feval(@msub_grad,xold,sample,label);
                gg = gg/mb + gold;
            end
        else
            if k==1
                gg = feval(@sub_grad, x, sample, label);
            else
                gg = feval(@sub_grad,x,sample,label) - feval(@sub_grad,xold,sample,label);
                gg = gg + gold;
             end
        end

        % SVRG        
        x = (1-eta*gamma)*x - eta * gg;
        if j > fix(m/2)
            xx = xx + x;
            numx = numx + 1;
        end
    end 
    xold = xx/numx;
    x = xold;
    %xold = x;
end
end
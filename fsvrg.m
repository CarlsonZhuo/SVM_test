function [x] = fsvrg(samples, labels, gamma, max_it, mb)

if nargin < 5
    mb = 10;  
end 

% initialization
[d, N] = size(samples);
x = zeros(d,1);
xold   = x;
rnd_pm = [randperm(N)];
m      = fix(2*N/mb);
eta    = 0.80;  
gold   = zeros(d,1); 
max1   = fix(max_it/(2*N))+1;
theta  = 0.95; 

for k = 1:max1
    
    if k > 0     
        gold = feval(@full_grad, xold, samples, labels); 
        gold = gold/N;
    end
    ww = zeros(d,1); 
    w  = xold;
    numw = 0;
    
    for j = 1:m 
        
        % Randomly choose sample
        idx = rnd_pm(mod(k,N)+1);
        sample = samples(:,idx);
        label  = labels(idx);

        % Randomly choose minibatch   
        idx = ceil(N*rand);
        if idx <= N-mb+1
            ix = idx:idx+mb-1;
        else
            ix = [1:(idx+mb-N-1),idx:N];
        end
        I = sort(rnd_pm(ix));
        sample = samples(:,I);
        label  = labels(I);    

        % Stochastic Sub-gradient Descent 
        if mb > 1            
            if k==1
                gg = feval(@msub_grad, w, sample, label);
                gg = gg/mb;
            else
                gg = feval(@msub_grad, w, sample,label) - feval(@msub_grad,xold,sample,label);
                gg = gg/mb + gold;
            end
        else
            if k==1
                gg = feval(@sub_grad, w, sample, label);
            else
                gg = feval(@sub_grad, w, sample,label) - feval(@sub_grad,xold,sample,label);
                gg = gg + gold;
             end
        end

        % FSVRG        
        x  = (1-eta*gamma)*x - eta * gg; 
        w  = x*theta + xold*(1-theta); 
        ww = ww + j*w;
    end  
    xold = ww/(m*(m+1)/2); 
    x    = xold;    
end
end
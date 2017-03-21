function g = full_grad(x, sample, label)
% calculate full subgradient for hinge loss
% 
d   = size(sample,1);
ind = find(label.*(x'*sample) <1.0);
gg  = -repmat(label(ind)',1,d)'.*sample(:,ind);
g   = sum(gg, 2);
end
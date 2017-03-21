function g = msub_grad(x, sample, label)
% calculate subgradient for hinge loss

d   = size(sample,1);
ind = find(label.*(x'*sample) <1.0);
if sum(label.*(x'*sample) <1.0) > 0
    gg = -repmat(label(ind)',1,d)'.*sample(:,ind);
    g  = sum(gg, 2);
else
    g = zeros(d,1);
end
end
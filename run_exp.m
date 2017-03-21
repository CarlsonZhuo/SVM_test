
clear all; 
close all;

epochs = 15;
runs   = 3; 

%%% SGD
opt_method = 'SGD';  % Stochastic subgradient descent

tic
accuracy1 = zeros(runs,1);
for i = 1:runs
    accuracy1(i) = classification_multi(opt_method, epochs);
    disp(accuracy1(i))
end
time1 = toc

acc1 = mean(accuracy1)
std1 = std(accuracy1)


runs = 10; 
%%% SVRG
opt_method = 'svrg';  % Stochastic Variance Reduced Gradient 

tic
accuracy2 = zeros(runs,1);
for i= 1:runs
    accuracy2(i) = classification_multi(opt_method, epochs);
    disp(accuracy2(i))
end
time2 = toc

acc2 = mean(accuracy2)
std2 = std(accuracy2)



%%% FSVRG
opt_method = 'fsvrg'; % Fast Stochastic Variance Reduced Gradient 

tic
accuracy3 = zeros(runs,1);
for i= 1:runs
    accuracy3(i) = classification_multi(opt_method, epochs);
    disp(accuracy3(i))
end
time3 = toc

acc3 = mean(accuracy3)
std3 = std(accuracy3)



%%% show results
b  = bar([acc1,acc2,acc3]);
ch = get(b,'children'); 
set(ch,'FaceVertexCData',[1;2;3])
set(gca,'XTickLabel',{'SGD','SVRG','FSVRG'})
axis([0.45 3.55 0.85 0.86]) 


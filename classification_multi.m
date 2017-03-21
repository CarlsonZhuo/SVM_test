function pred_accuracy = classification_multi(opt_method, epochs)

%%% training for multiple-classes: One-vs-Rest

D = load('20news_w100');

% Training Parameters
ratio_train = 0.8;
ratio_test  = 1-ratio_train;
N = [4605, 3519, 2657, 5461];
N_train = ceil(N * ratio_train);
N_test  = N - N_train;
N_s     = [0, 4605, 8124, 10781];
N_train_all = sum(N_train);
N_test_all  = sum(N_test);
N_all = sum(N); d = 100;

% for all
gamma  = 0.00001;
x      = zeros(d, 4);
max_it = epochs * N_train_all;

% for SGD
d_bound_sgd = 1;  %%%%%%%  tuning parameter  %%%%%%%%%


%%% Stochastic training
%tic
for c = 1:4
    disp(['Training for Class ', num2str(c)]);
    % preparing data
    train_samples = zeros(d, N_train(c));
    train_samples(:,1:N_train(c)) = D.documents(:,N_s(c)+1:N_s(c)+N_train(c));
    for i = 1:4
        if i~=c
            train_samples = [train_samples, D.documents(:,N_s(i)+1:N_s(i)+N_train(i))];
        end
    end
    train_labels(1:N_train(c)) = 1;
    train_labels(N_train(c)+1:N_train_all) = -1;
    
    if strcmp(opt_method, 'fsvrg')
        x(:,c) = fsvrg(train_samples, train_labels, gamma, max_it);
    elseif strcmp(opt_method, 'svrg')
        x(:,c) = svrg(train_samples, train_labels, gamma, max_it);
    else
        x(:,c) = stoc_gd(train_samples, train_labels, gamma, max_it, d_bound_sgd);
    end
end
%toc
disp('Training finished.');


%%% Testing
err_ct   = 0;
pred_lbl = zeros(N_test_all, 1);
test_ct  = 0;
for c = 1:4
    for i = 1:N_test(c)
        test_ct = test_ct + 1;
        test_s = D.documents(:, N_s(c)+N_train(c)+i);
        pred_val = zeros(1,4);
        for p = 1:4
            pred_val(p) = test_s' * x(:,p);
        end
        [pred_val_max, pred_lbl(test_ct)] = max(pred_val);
        if (pred_lbl(test_ct) ~= c)
            err_ct = err_ct + 1;
        end
    end
end

pred_accuracy = 1 - err_ct/N_test_all;
disp(['Prediction accuracy: ', num2str(pred_accuracy)]);

end
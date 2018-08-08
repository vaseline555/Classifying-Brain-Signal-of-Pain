clear
close all
clc

%% Load Data
load('classification_data.mat')

%%
% 1. Generate training data
c_pos = find(Gtrain == 1); c_neg = find(Gtrain == -1);
train_pos = Xtrain(c_pos,:)'; train_neg = Xtrain(c_neg,:)';
nr_train_pos = size(train_pos,2); nr_train_neg = size(train_neg,2);
nr_train = nr_train_pos + nr_train_neg;

t_pos = find(Gtest == 1); t_neg = find(Gtest == -1);
test_pos = Xtest(t_pos,:)'; test_neg = Xtest(t_neg,:)';
nr_test_pos = size(test_pos,2); nr_test_neg =  size(test_neg,2); 
nr_test = nr_test_pos + nr_test_neg;

tr_data = horzcat(Xtrain, Gtrain);
te_data = horzcat(Xtest, Gtest);

tr_in = tr_data(:, 1:2);
tr_out = zeros(nr_train, 2);
tr_out(tr_data(:, 3) == 1, 1) = 1;
tr_out(tr_data(:, 3) == -1, 2) = 1;
te_in = te_data(:, 1:2);
te_out = zeros(nr_test, 2);
te_out(te_data(:, 3) == 1, 1) = 1;
te_out(te_data(:, 3) == -1, 2) = 1;

%%
% 2. Plot
figure(1);
subplot(121); hold on;
pos_idx = find(tr_data(:, 3) == 1);
neg_idx = find(tr_data(:, 3) == -1);
plot(tr_data(pos_idx, 1), tr_data(pos_idx, 2), 'ro')
plot(tr_data(neg_idx, 1), tr_data(neg_idx, 2), 'bx')
axis equal; grid on; title('Training Data');
subplot(122); hold on;
pos_idx = find(te_data(:, 3) == 1);
neg_idx = find(te_data(:, 3) == -1);
plot(te_data(pos_idx, 1), te_data(pos_idx, 2), 'ro')
plot(te_data(neg_idx, 1), te_data(neg_idx, 2), 'bx')
axis equal; grid on; title('Test Data');


%%
% 3. Deep Neural Network
nodes = [2 200 500 50 2];
dnn   = init_dnn(nodes);

opts.MaxIter     = 100;
opts.BatchSize   = nr_train/2;
opts.Verbose     = false; 
opts.StepRatio   = 0.1;
opts.Layer       = length(nodes)-1;
opts.DropOutRate = 0.5;
opts.WeightCost  = 0.001;
opts.Object      = 'CrossEntorpy'; % 'CrossEntorpy' / 'Square'
opts.Layer       = 0;
tic

% Pretrain
dnn = pretrain_dnn(dnn, tr_in, opts);
dnn = set_linearMapping(dnn, tr_in, tr_out);

% Fine-tune
dnn = train_dnn(dnn, tr_in, tr_out, opts);
dnn_train_toc = toc;

% Compute error
trpred = v2h(dnn, tr_in);
hidden_no_train = size(trpred, 2)
trnegidx = find(trpred(:, 1) < trpred(:, 2));
trpredvec = ones(nr_train, 1);
trpredvec(trnegidx) = -1;
tic;
tepred = v2h(dnn, te_in);
hidden_no_test = size(tepred, 2)
dnn_test_toc = toc;
tenegidx = find(tepred(:, 1) < tepred(:, 2));
tepredvec = ones(nr_test, 1);
tepredvec(tenegidx) = -1;



dnn_train_acc = mean(trpredvec==tr_data(:, 3));
dnn_test_acc  = mean(tepredvec==te_data(:, 3));
fprintf('======== Deep Neural Network ========\n');
fprintf('DNN train: %.3f sec \n', dnn_train_toc);
fprintf('DNN test: %.3f sec \n', dnn_test_toc);
fprintf('DNN Train Accuracy: %.1f%% AdaBoost Test Accuracy: %.1f%%  \n', dnn_train_acc*100, dnn_test_acc*100);


T_P = length(find(Gtest == 1 & tepredvec == 1));
T_nP = length(find(Gtest == -1 & tepredvec == -1));
F_P = length(find(Gtest == -1 & tepredvec == 1));
F_nP = length(find(Gtest == 1 & tepredvec == -1));

True_Painful_Bc = [T_P; F_nP];
True_NonPainful_Bc = [F_P; T_nP];

EstimatedClass = {'Est_Painful'; 'Est_NonPainful'};
Error_type_table_SVM = table(True_Painful_Bc, True_NonPainful_Bc, 'RowNames', EstimatedClass)
%%
% Do some plotting
xgrid = linspace(0, 1, 100);
ygrid = linspace(0, 1, 100);
[mesh_X, mesh_Y] = meshgrid(xgrid, ygrid);
points = [mesh_X(:) mesh_Y(:)];
nr_points = length(points);
% Boosted Random Fern
brf_grid = v2h(dnn, points);
gridnegidx = find(brf_grid(:, 1) < brf_grid(:, 2));
gridpredvec = ones(nr_points, 1);
gridpredvec(gridnegidx) = -1;
gridpredvec = gridpredvec - min(gridpredvec);
gridpredvec = gridpredvec / max(gridpredvec);
brf_grid_rs = reshape(gridpredvec, 100, 100);

figure(2); clf; colormap jet
imagesc(ygrid, xgrid, brf_grid_rs); hold on;
pos_idx = find(tr_data(:, 3) == 1);
neg_idx = find(tr_data(:, 3) == -1);
plot(tr_data(pos_idx, 1), tr_data(pos_idx, 2), 'ko', 'MarkerFaceColor', 'r');
plot(tr_data(neg_idx, 1), tr_data(neg_idx, 2), 'k^', 'MarkerFaceColor', 'b');
axis equal; axis off; grid on; title('Training Data');


figure(3); clf; colormap jet
imagesc(ygrid, xgrid, brf_grid_rs); hold on;
pos_idx = find(te_data(:, 3) == 1);
neg_idx = find(te_data(:, 3) == -1);
plot(te_data(pos_idx, 1), te_data(pos_idx, 2), 'ko', 'MarkerFaceColor', 'r');
plot(te_data(neg_idx, 1), te_data(neg_idx, 2), 'k^', 'MarkerFaceColor', 'b');
axis equal; axis off; grid on; title('Test Data');

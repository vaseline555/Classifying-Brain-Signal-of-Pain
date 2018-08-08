clear
close all
clc

load('classification_data.mat')

%% Gaussian Kernel Function
figure(1)
SVMStruct = svmtrain(Xtrain, Gtrain, 'autoscale', true, 'boxconstraint', 9e-2, 'kernel_function', 'rbf', 'showplot', true);
%%%pause(5);
testResult = svmclassify(SVMStruct, Xtest, 'showplot', true);
Performance = Evaluate(Gtest, testResult)

%% Polynomial Kernel Function
figure(2)
SVMStruct = svmtrain(Xtrain, Gtrain, 'autoscale', true, 'boxconstraint', 4e-1, 'kernel_function', 'polynomial', 'showplot', true);
%%%pause(5);
testResult = svmclassify(SVMStruct, Xtest, 'showplot', true);
Performance = Evaluate(Gtest, testResult)

%% Sigmoid Kernel Function
figure(3)
SVMStruct = svmtrain(Xtrain, Gtrain, 'autoscale', true, 'boxconstraint', 4e-3, 'kernel_function', @mysigmoid, 'showplot', true);
%%%pause(5);
testResult = svmclassify(SVMStruct, Xtest, 'showplot', true);
Performance = Evaluate(Gtest, testResult)


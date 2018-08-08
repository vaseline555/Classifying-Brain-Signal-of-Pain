clear
close all
clc

%% Linear generative model
load('classification_data.mat')

c_1 = find(Gtrain == 1);
c_2 = find(Gtrain == -1);

class1 = Xtrain(c_1,:)';
class2 = Xtrain(c_2,:)';

Prior = length(class1)/length(Xtrain);

mu_1 = mean(class1,2);
mu_2 = mean(class2,2);

Sum_1 = [0 0; 0 0];

for i = 1:length(class1);
    Variance_1 = (class1(:,i) - mu_1)*(class1(:,i) - mu_1)';
    Sum_1 = Sum_1 + Variance_1;
end

S_1 = Sum_1/length(class1);

Sum_2 = [0 0; 0 0];
for i = 1:length(class2);
    Variance_2 = (class2(:,i) - mu_1)*(class2(:,i) - mu_1)';
    Sum_2 = Sum_2 + Variance_2;
end

S_2 = Sum_2/length(class2);
    
Sigma_ML = (length(class1)/length(Xtrain))*S_1 + (length(class2)/length(Xtrain))*S_2;

W = (inv(Sigma_ML))*(mu_1-mu_2);
W_0 = -0.5*(mu_1')*(inv(Sigma_ML))*(mu_1) + 0.5*(mu_2')*(inv(Sigma_ML))*(mu_2) + log(Prior/(1-Prior));

Posterior_1 = (1./(1 + exp(-((W')*(Xtrain')) - W_0)))';
Posterior_2 = (1./(1 + exp(((W')*(Xtrain')) + W_0)))';

Est_class1 = Xtrain(Posterior_1 > 0.5,:)';
Est_class2 = Xtrain(Posterior_1 < 0.5,:)';

x = 0.3;
y = 0.5;
point1 = [];
while y < 1.5
    X = [x; y];
    difference = (1/(1 + exp(-((W')*(X)) - W_0))) - (1/(1 + exp(((W')*(X)) + W_0)));
    point1 = [point1; x y difference];
    
    y = y + 0.0001;
end

x = 0.7;
y = 0.5;
point2 = [];
while y < 1.5
    X = [x; y];
    difference = (1/(1 + exp(-((W')*(X)) - W_0))) - (1/(1 + exp(((W')*(X)) + W_0)));
    point2 = [point2; x y difference];
    
    y = y + 0.0001;
end

p1 = point1(point1(:,3) < 0,:);
p2 = point2(point2(:,3) < 0,:);

x = linspace(-0.1,1.1,1000);
y = ((p2(1,2)-p1(1,2))/(p2(1,1)-p1(1,1)))*(x-p1(1,1)) + p1(1,2);

figure;
scatter(class1(1,:), class1(2,:), 5, 'ro');
hold on
scatter(class2(1,:), class2(2,:), 5, 'bo');
hold on
plot(x,y)

%% Applying on test dataset
c_1_test = find(Gtest == 1);
c_2_test = find(Gtest == -1);

class1_test = Xtest(c_1_test,:)';
class2_test = Xtest(c_2_test,:)';

Posterior_1 = (1./(1 + exp(-((W')*(Xtest')) - W_0)))';
Posterior_2 = (1./(1 + exp(((W')*(Xtest')) + W_0)))';

Est_class1 = Xtest(Posterior_1 > 0.5,:)';
Est_class2 = Xtest(Posterior_1 < 0.5,:)';
    
figure;
scatter(class1_test(1,:), class1_test(2,:), 5, 'ro');
hold on
scatter(class2_test(1,:), class2_test(2,:), 5, 'bo');
hold on
plot(x,y)

Mis_c1_SigmaML = find((Gtest(:,1) == 1) & (Posterior_1(:,1) < 0.5));
Mis_c2_SigmaML = find((Gtest(:,1) == -1) & (Posterior_1(:,1) > 0.5));

Accuracy_SigmaML = 1 - ((length(Mis_c1_SigmaML) + length(Mis_c2_SigmaML))/length(Xtest));

%% Quadratic generative model
clear

load('classification_data.mat')

c_1 = find(Gtrain == 1);
c_2 = find(Gtrain == -1);

class1 = Xtrain(c_1,:)';
class2 = Xtrain(c_2,:)';

Prior = length(class1)/length(Xtrain);

mu_1 = mean(class1,2);
mu_2 = mean(class2,2);

Sum_1 = [0 0; 0 0];
for i = 1:length(class1);
    Variance_1 = (class1(:,i) - mu_1)*(class1(:,i) - mu_1)';
    Sum_1 = Sum_1 + Variance_1;
end

S_1 = Sum_1/length(class1);

Sum_2 = [0 0; 0 0];
for i = 1:length(class2);
    Variance_2 = (class2(:,i) - mu_1)*(class2(:,i) - mu_1)';
    Sum_2 = Sum_2 + Variance_2;
end

S_2 = Sum_2/length(class2);

W1 = inv(S_1) - inv(S_2);
W2 = ((mu_1')*(inv(S_1))) - ((mu_2')*(inv(S_2)));
W3 = ((inv(S_1))*(mu_1)) - ((inv(S_2))*(mu_2));
W_0 = ((mu_1')*(inv(S_1))*(mu_1)) - ((mu_2')*(inv(S_2))*(mu_2)) - (2*log(Prior/(1-Prior)))...
    - (2*(log(((det(S_2))^0.5)/((det(S_1))^0.5))));
% a = (-0.5*(Xtrain)*W1*(Xtrain')) + (0.5*W2*(Xtrain')) + (0.5*(Xtrain)*W3) - (0.5*W_0);

for i = 1:length(Xtrain);    
    x = Xtrain(i,1);
    y = Xtrain(i,2);
    X = [x y];
    a = (-0.5*(X)*W1*(X')) + (0.5*W2*(X')) + (0.5*(X)*W3) - (0.5*W_0);
    Posterior_1(i,1) = (1./(1 + exp(-a)));
    Posterior_2(i,1) = (1/(1 + exp(a)));
end

Est_class1 = Xtrain(Posterior_1 > 0.5,:)';
Est_class2 = Xtrain(Posterior_1 < 0.5,:)';

x = -0.1;
y = 0.6;
Prob = [];
while x < 1.10001   
    X = [x y];
    a = (-0.5*(X)*W1*(X')) + (0.5*W2*(X')) + (0.5*(X)*W3) - (0.5*W_0);
    
    Prob = [Prob; (1/(1 + exp(-a)))];
    x = x + 0.00001;
end

x = -0.1:0.00001:1.1;
P = find(Prob < 0.5001 & Prob > 0.4999);
P1 = [x(P(1)),y]; 
P2 = [x(P(end)),y];

x = -0.1;
y = 0.4;
Prob = [];
while x < 1.10001   
    X = [x y];
    a = (-0.5*(X)*W1*(X')) + (0.5*W2*(X')) + (0.5*(X)*W3) - (0.5*W_0);
    
    Prob = [Prob; (1/(1 + exp(-a)))];
    x = x + 0.00001;
end

x = -0.1:0.00001:1.1;
P = find(Prob < 0.5001 & Prob > 0.4999);
P3 = [x(P(1)),y]; 
P4 = [x(P(end)),y];

x = -0.1;
y = 0.1;
Prob = [];
while x < 1.10001   
    X = [x y];
    a = (-0.5*(X)*W1*(X')) + (0.5*W2*(X')) + (0.5*(X)*W3) - (0.5*W_0);
    
    Prob = [Prob; (1/(1 + exp(-a)))];
    x = x + 0.00001;
end

x = -0.1:0.00001:1.1;
P = find(Prob < 0.5001 & Prob > 0.4999);
P5 = [x(P(1)),y]; 
P6 = [x(P(end)),y];

PointX = [P1(1,1); P2(1,1); P3(1,1); P4(1,1); P5(1,1); P6(1,1)];
PointY = [P1(1,2); P2(1,2); P3(1,2); P4(1,2); P5(1,2); P6(1,2)];

f = fit(PointX, PointY, 'poly2');

figure;
scatter(class1(1,:), class1(2,:), 5, 'ro');
hold on
scatter(class2(1,:), class2(2,:), 5, 'bo');
hold on
plot(f,PointX,PointY)

%% Applying on test dataset
c_1_test = find(Gtest == 1);
c_2_test = find(Gtest == -1);

class1_test = Xtest(c_1_test,:)';
class2_test = Xtest(c_2_test,:)';

Posterior_1 = [];
Posterior_2 = [];
for i = 1:length(Xtest);    
    x = Xtest(i,1);
    y = Xtest(i,2);
    X = [x y];
    a = (-0.5*(X)*W1*(X')) + (0.5*W2*(X')) + (0.5*(X)*W3) - (0.5*W_0);
    Posterior_1(i,1) = (1/(1 + exp(-a)));
    Posterior_2(i,1) = (1/(1 + exp(a)));
end

Est_class1 = Xtest(Posterior_1 > 0.5,:)';
Est_class2 = Xtest(Posterior_1 < 0.5,:)';

figure;
scatter(class1_test(1,:), class1_test(2,:), 5, 'ro');
hold on
scatter(class2_test(1,:), class2_test(2,:), 5, 'bo');
hold on
plot(f,PointX,PointY)

% figure;
% scatter(Est_class1(1,:), Est_class1(2,:), 5, 'ro');
% hold on
% scatter(Est_class2(1,:), Est_class2(2,:), 5, 'bo');
% hold on
% plot(f,PointX,PointY)

Mis_c1_diffSigma = find((Gtest(:,1) == 1) & (Posterior_1(:,1) < 0.5));
Mis_c2_diffSigma = find((Gtest(:,1) == -1) & (Posterior_1(:,1) > 0.5));
Xpred = horzcat(Mis_c1_diffSigma, Mis_c2_diffSigma);
Accuracy_diffSigma = 1 - ((length(Mis_c1_diffSigma) + length(Mis_c2_diffSigma))/length(Xtest));
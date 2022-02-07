


% Initialization



input = 400;  % 20x20 Input Images of Digits
hidden = 50;   % 50 hidden units
output= 10;          % 10 labels, from 1 to 10   
                          

% Load Training Data

load('data.mat');
m = size(X, 1);


fprintf('\nInitializing Neural Network Parameters ...\n')


initial_Theta1 = 0.2*randn(hidden,(input+1));%随机初始化参数
initial_Theta2 = 0.2*randn(output,(hidden+1));

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];








%---------------------------training--------------- 

fprintf('\nTraining Neural Network... \n')


lambda = 1;%正则化系数

count=3000;%迭代次数
[J,grad] = myCostfunction(initial_nn_params, input, hidden,output,X,y, lambda);%计算初始梯度和代价
[nn_params,cost]=fmin(J,grad,initial_nn_params,count,input,hidden,output,X,y,lambda);%调用优化算法

% Obtain Theta1 and Theta2 back from nn_params


Theta1 = reshape(nn_params(1:hidden * (input + 1)),hidden, (input + 1));

Theta2 = reshape(nn_params((1 + (hidden * (input + 1))):end),output, (hidden + 1));

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);





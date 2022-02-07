function p = predict(Theta1, Theta2, X)
% 利用训练好的参数预测



m = size(X, 1);
output = size(Theta2, 1);


p = zeros(size(X, 1), 1);

h1 = tanh([ones(m, 1) X] * Theta1');
h2 = [ones(m, 1) h1] * Theta2';
[d, p] = max(h2, [], 2);




end

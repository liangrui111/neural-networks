function [J grad] = myCostfunction(nn_params, input, hidden,output,X,y, lambda)

Theta1 = reshape(nn_params(1:hidden * (input + 1)),hidden, (input + 1));

Theta2 = reshape(nn_params((1 + (hidden * (input + 1))):end),output, (hidden + 1));




m = size(X, 1);

% convert y£¨0-9£© to vector
c = 1:output;
yt = zeros(output,m); 
for i = 1:m
    yt(:,i) = (c==y(i));   
end         


J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%--------------------caculate cost------------------------ 

% compute hx
a1 = [ones(m, 1) X];    
a1=a1';
z2 = Theta1*a1;      
a2 = tanh(z2);       

a2 = [ones(m, 1) a2'];   
a2=a2';
z3 = Theta2*a2;   
hx=z3;



% regularization term
regTerm = lambda / 2 / m * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

% J with regularization
J = 1 /2/ m * sum(sum((hx-yt).*(hx-yt))) + regTerm; 


%----------------------------------- Backpropagation-----------------------

% Accumulate the error term

delta_3=hx-yt;

delta_2 = Theta2'*delta_3 .* tanhGradient(a2);

delta_2 = delta_2(2:end,:);   

% Accumulate the gradient 

D2 = delta_3*a2';    

D1 = delta_2*a1';    

% Obtain the (unregularized) gradient for the neural network cost function
Theta2_grad = 1/m * D2;
Theta1_grad = 1/m * D1;

%---Regularize gradients

temp1 = Theta1;
temp2 = Theta2;
temp1(:,1) = 0; % set first column to 0
temp2(:,1) = 0; % set first column to 0
Theta1_grad = Theta1_grad + lambda/m * temp1;
Theta2_grad = Theta2_grad + lambda/m * temp2;

% -------------------------------------------------------------



% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
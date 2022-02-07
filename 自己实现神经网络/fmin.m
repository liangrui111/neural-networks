function [nn_params, cost]=fmin(J,grad,nn_params,count, input, hidden,output,X,y, lambda)
%共轭梯度下降prp算法

S='Iteration ';
learing_rate=0.01;
g(:,1)=grad;
p(:,1)=-g(:,1);

for k=2:(count+1)
nn_params=nn_params+learing_rate*p(:,k-1);
[cost,g(:,k)]=myCostfunction(nn_params,input, hidden,output,X,y, lambda);
fprintf('%s %4i | Cost: %4.6e\r', S, k-1, cost);
beta(:,k-1)=g(:,k)'*(g(:,k)-g(:,k-1))/(g(:,k-1)'*g(:,k-1));

p(:,k)=-g(:,k)+beta(:,k-1)*p(:,k-1);


end


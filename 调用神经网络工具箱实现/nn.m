% intput=400;
% hidden=50;
% output=10;
% % Load Training Data
% 
% load('data.mat');
% 
% m = size(X, 1);
% X=X';
% 
% 
% % convert y（0-9） to vector
% c = 1:output;
% yt = zeros(output,m); 
% for i = 1:m
%     yt(:,i) = (c==y(i));   
% end    
% 
% 
% 
% P=X;%神经网络输入
% T=yt;%神经网络输出目标
% %定义神经网络，采用正切和线性激活函数，采用powell-beale共轭梯度法
% net=newff(P,T,[50],{'tansig' 'purelin'} ,'traincgb');
% 
% 
% net.trainParam.epochs=200;%迭代200次
% net.trainParam.goal=1e-5;
% 
% [net,tr]=train(net,P,T);%训练网络
% yp=sim(net,X);%仿真预测
% yp=yp';
% 
% [d, p] = max(yp, [], 2);%返回最大值索引
% 
% fprintf('\nTrainning Set Accuracy: %f\n', mean(double(p == y)) * 100);













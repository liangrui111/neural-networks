function g = tanhGradient(z)

g = zeros(size(z));

g =1-tanh(z).*tanh(z);





end

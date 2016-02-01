function [J, grad] = cost(theta, X, y, lambda)

m = length(y); % number of training examples

bob = X * theta;
%calculating the cost
J = (sum(-y .* log(sigmoid(bob)) - (1-y).*(log(1 - (sigmoid(bob))))))/m + (lambda/(2*m))*sum(theta(2:end) .^ 2);
%calculating the gradient
grad = (1/m)*(X'*(sigmoid(bob) - y));
grad(2:end) = theta(2:end) .* (lambda/m) + grad(2:end);

grad = grad(:);

end

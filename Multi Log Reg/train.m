function [all_theta] = train(X, y, num_labels, lambda)

m = size(X, 1); %number of training examples
n = size(X, 2);	%number of input values

all_theta = zeros(num_labels, n + 1);

X = [ones(m, 1) X];
K = 1:num_labels;

     initial_theta = zeros(n + 1, 1);
     options = optimset('GradObj', 'on', 'MaxIter', 50);
%go through each answer, and train one vs all logistic regression using fmincg.
for c = 1:num_labels
     [theta] = ...
         fmincg (@(t)(cost(t, X, (y == c), lambda)), initial_theta, options);
     all_theta(c, :) = theta';
end

end

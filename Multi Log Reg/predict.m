function p = predict(all_theta, X)

m = size(X, 1);
num_labels = size(all_theta, 1);

X = [ones(m, 1) X];
       

[x, p] = max(sigmoid(all_theta * X', 2), [], 1);
p = mod(p, 10);


end

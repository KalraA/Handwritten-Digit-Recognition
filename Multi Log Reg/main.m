input_layer_size  = 20*20;  % 20x20 Input Images of Digits
num_labels = 10;       %the digits 0->9


load('digits.mat'); % this loads the X array of 5000 images 400 px each, and the y array, containing a number between 0-9 corresponding to each image

m = size(X, 1); %Number of training examples

lambda = 0.1; %the regularizaiton constant
%training the algorithm.
[all_theta] = train(X, y, num_labels, lambda);

pred = predict(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100); %Tells accuracy of algo


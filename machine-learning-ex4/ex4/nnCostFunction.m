function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% fprintf('Dim(y): [%d, %d]\n', size(y));
Y = zeros(num_labels, m);
for i=1:m
    Y(y(i), i) = 1;
end
% fprintf('Dim(Y): [%d, %d]\n', size(Y));

% Hypothesis value: tempH
A1 = [ones(m, 1) X];
% fprintf(' dimensions A1: (%d %d)\n', [size(A1,1) size(A1,2)]);
Z2 = Theta1 * A1';
% fprintf(' dimensions Z2: (%d %d)\n', [size(Z2,1) size(Z2,2)]);
A2 = sigmoid(Z2);
A2 = [ones(size(A2, 2), 1)' ; A2];
% fprintf(' dimensions A2: (%d %d)\n', [size(A2,1) size(A2,2)]);
Z3 = Theta2 * A2;
% fprintf(' dimensions Z3: (%d %d)\n', [size(Z3,1) size(Z3,2)]);
tempH = sigmoid(Z3);

% fprintf("Dim(tempH): [%d %d]\n", size(tempH));
% disp(tempH);

% nullifies when y=0
log0 = log(tempH);

% nullifies when y=1
log1 = log((1 - tempH));

tempS = (Y.*log0) + ((1 - Y).*log1);
% fprintf("Dim(tempS): [%d %d]\n", size(tempS));

J = (-1)*sum(sum(tempS))/m;

% We do not regularize for bias terms; first columns of theta matrices
theta1 = Theta1;
theta1(:, 1) = zeros(size(Theta1, 1), 1);
theta2 = Theta2;
theta2(:, 1) = zeros(size(Theta2, 1), 1);
all_theta = [theta1(:); theta2(:)];

J = J + ((lambda/(2*m))*sum(all_theta.*all_theta));


% backpropagation:
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));
for i=1:m
    delTemp3 = tempH(:,i) - Y(:,i);
    delTemp2 = (Theta2'*delTemp3).*sigmoidGradient([0; Z2(:,i)]);
    delTemp2 = delTemp2(2:end);
    % fprintf("Dim(delTemp2): [%d %d]\n", size(delTemp2));
    % fprintf("Dim(A1(i,:)): [%d %d]\n", size(A1(i,:)));
    Delta2 = Delta2 + (delTemp3*A2(:,i)');
    Delta1 = Delta1 + (delTemp2*A1(i,:));   % One row of A1 represents the i th input value. No need to transpose, as it is a row vector (transpose of col. vector)
end

Theta1_grad = Delta1 ./ m;
Theta2_grad = Delta2 ./ m;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

% adding regularization to grad
grad = grad + (lambda/m)*all_theta;


end

function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


% Hypothesis exponent value, tempH dimension: mxn * nx1 = mx1
tempH = X * theta;
% Hypothesis value, tempH dimension: mxn * nx1 = mx1
tempH = sigmoid(tempH);

% nullifies when y=0
log0 = log(tempH);

% nullifies when y=1
log1 = log((1 - tempH));

tempS = (y.*log0) + ((1 - y).*log1);

J = (-1)*sum(tempS)/m;

% Vectorized formula for gradient of J
grad = (X'*(tempH - y))/m;

% Since we do not regularize for j=0
theta(1,:) = zeros(1, size(theta,2));

% fprintf('Cost at initial theta (zeros): %f\n', cost);
% fprintf(' %f \n', theta(1));

J = J + ((lambda/(2*m))*sum(theta.*theta));

grad = grad + (lambda/m)*theta;

% =============================================================

grad = grad(:);

end

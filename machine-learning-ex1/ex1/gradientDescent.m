function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    tempH = X*theta;
%    tempD0 = (tempH - y).*X(:,1);
%    tempD1 = (tempH - y).*X(:,2);
%    tempS0 = sum(tempD0);
%    tempC0 = (alpha * tempS0/m);
%    tempS1 = sum(tempD1);
%    tempC1 = (alpha * tempS1/m);
%    theta(1) = theta(1) - tempC0;
%    theta(2) = theta(2) - tempC1;

    tempD = (tempH - y).*X;
    tempS = sum(tempD);
    tempC = (alpha/m)*(tempS');
    theta = theta - tempC;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end

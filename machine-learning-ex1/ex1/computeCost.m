function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% Hypothesis value, tempH dimension: mxn * nx1 = mx1
tempH = X * theta;

% Delta value, tempD dimension: mx1
tempD = tempH - y;

% Delta square value, tempD2 dimension: mx1
tempD2 = tempD.*tempD;

% Summation of all m elements, J dimension: scalar
J = sum(tempD2);

% Divide by 2*m factor
J = J/(2*m);


% =========================================================================

end

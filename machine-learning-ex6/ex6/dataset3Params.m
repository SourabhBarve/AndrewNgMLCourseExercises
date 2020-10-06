function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C1 = [0.01,0.03,0.1,0.3,1,3,10,30,100];
sigma1 = [0.01,0.03,0.1,0.3,1,3,10,30,100];
error = ones(size(C1,2), size(sigma1,2));
minimum_error = 100000;
for i=1:size(C1,2)
    for j=1:size(sigma1,2)
        model= svmTrain(X, y, C1(i), @(x1, x2) gaussianKernel(x1, x2, sigma1(j)));
        predictions = svmPredict(model, Xval);
        error(i,j) = mean(double(predictions ~= yval));
        if (error(i,j) < minimum_error)
            minimum_error = error(i,j);
            C = C1(i);
            sigma = sigma1(j);
            end
%         fprintf("\nC:%f , sigma:%f, error:%f\n", C1(i), sigma1(j), error(i,j));
    end
end


% fprintf("\nBest C and sigma:\nC:%f , sigma:%f, error:%f\n", C, sigma, minimum_error);
% C = 1;
% sigma1 = 0.1;

% =========================================================================

end

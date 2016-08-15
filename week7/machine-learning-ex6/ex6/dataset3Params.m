function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
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

v = [0.01 0.03 0.1 0.3 1 3 10 30];

%dummy value of 0 assigned
min_error_val = 0;
min_c = 0;
min_s = 0;

for c=1:length(v),
	for s = 1:length(v),
		%use this combo of c and s to get a prediction by training it on the training set X and y
		model= svmTrain(X, y, v(c), @(x1, x2) gaussianKernel(x1, x2, v(s)));
		%use the model to predict values on the cross validation set
		predictions = svmPredict(model,Xval);
		prediction_error = mean(double(predictions ~= yval));
		if c == 1 && s == 1,
			min_error_val = prediction_error;
			min_c = 1;
			min_s = 1;
		else,
			if min_error_val > prediction_error,
				min_error_val = prediction_error;
				min_c = c;
				min_s = s;
			end
		end
	end
end

C = v(min_c);
sigma = v(min_s);
% =========================================================================

end

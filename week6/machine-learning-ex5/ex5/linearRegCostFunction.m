function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X * theta;
J1 = h - y;
%g1 is for later use in gradient computation
g1 = J1;

J1 = J1' * J1;
J1 = J1 / (2*m);
J2 = theta(2:end);
J2 = J2' * J2;
J2 = J2 * ( (lambda / (2* m ) ) );
J = J1 + J2;

%computing gradient
grad1 = g1' *  X;
grad1 = grad1';
grad1 = grad1 / m;

grad2 = theta(2:end);
grad2 = (lambda / m) *  grad2;
grad2 = [0; grad2];
grad = grad1 + grad2;








% =========================================================================

grad = grad(:);

end

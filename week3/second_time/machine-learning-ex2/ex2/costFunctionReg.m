function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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


z = X  * theta;
z = sigmoid(z);
logZ = log(z);
logOneMinusZ = log(1-z);
J = y .* logZ;
oneMinusY = 1-y;
J = J + (oneMinusY .* logOneMinusZ);
J = sum(J);
J = -J;
J = J / m;

tempTheta = theta;
tempTheta(1,1) = 0;
thetaSqr = tempTheta' * tempTheta;
thetaSqr = thetaSqr * ( lambda / ( 2 * m )); 
J = J + thetaSqr;

grad = z - y;
grad = X' *  grad;
grad = grad / m;
grad = grad + ( (lambda / m ) *  tempTheta );


% =============================================================

end

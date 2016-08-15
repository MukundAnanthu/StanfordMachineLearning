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



%adding x_0 to X
X = [ones(size(X,1),1) X];




%computing layer 2 units
t2 = Theta1 * X';
z2 = t2;
z2 = [zeros(1,m); z2];
t2 = sigmoid(t2);



%adding bias unit to layer 2
t2 = [ones(1,m); t2];

%computing layer3 units from layer 2
t3 = Theta2 * t2;
t3 = sigmoid(t3);

v = 1:num_labels;
yMat = [];


for i = 1:m,
	yMat = [yMat (v' == y(i,1))];	
end;

cost1 = yMat .* log(t3);
cost2 = (1- yMat) .* log(1-t3);
cost3 = cost1 + cost2;
cost4 = sum(cost3);
J = sum(cost4);
J = J / m;
J = -J;

%regularization 
reg1 = Theta1(:,2:end) .^ 2;
reg1 = sum(sum(reg1));

reg2 = Theta2(:,2:end) .^ 2;
reg2 = sum(sum(reg2));

reg = reg1 + reg2;
reg = reg * lambda;
reg = reg / ( 2 * m );

J = J + reg;

%backpropagation


smallDelta3 = t3 - yMat; %note that t3=a3=h_theta=output

smallDelta2 = ( ( Theta2' * smallDelta3 ) .* sigmoidGradient(z2) );

%remove bias unit from smallDelta2
smallDelta2 = smallDelta2(2:end,:);

delta_forTheta2 = smallDelta3 * (t2)';
delta_forTheta1 = smallDelta2 * X; %a1 is nothing but the input vector

D2 = delta_forTheta2 / m;
D1 = delta_forTheta1 / m;

Theta1_grad = D1;
Theta2_grad = D2;

%regularizing
D1(:,2:end) = D1(:,2:end) + ( (lambda / m ) * Theta1(:,2:end));
D2(:,2:end) = D2(:,2:end) + ( (lambda / m ) * Theta2(:,2:end));

Theta1_grad = D1;
Theta2_grad = D2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

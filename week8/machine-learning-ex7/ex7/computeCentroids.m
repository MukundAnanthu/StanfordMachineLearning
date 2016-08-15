function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

%iterating over each centroid
for i = 1:K,
	num_examples_assigned = 0;
	sum = zeros(1,n);
	%iterating over each training example
	for j = 1:m,
		%if the jth training example is assigned to the ith centroid
		if i == idx(j),
			sum = sum + X(j,:);
			num_examples_assigned = num_examples_assigned + 1;
		end
	end
	%compute the mean
	mean = sum / num_examples_assigned;
	%re-assign the ith centroid
	centroids(i,:) = mean;
end


% =============================================================


end


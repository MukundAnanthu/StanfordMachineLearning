function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

%number of training examples computed
m = size(X,1);




for i = 1:m,
	%assigning dummy values
	min_dist = -1;
	min_centroid_index = 0;	
	%taking i th training example
	p = X(i,:);
	for j = 1:K,
		%calculating squared dist
		temp = p - centroids(j,:);
		temp = temp * temp';
		%if min_dist hasn't been set so far
		if min_dist == -1,
			min_dist = temp;
			min_centroid_index = j;
		else,
			if min_dist > temp,
				min_dist = temp;
				min_centroid_index = j;
			end
		end
	end
	%assign centroid to the training example
	idx(i) = min_centroid_index;
end



% =============================================================

end


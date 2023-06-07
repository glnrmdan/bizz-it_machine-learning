import tensorflow.keras.backend as K


def euclidean_distance(vectors):
	# unpack the vectors into separate lists
	(feats_A, feats_B) = vectors
	# compute the sum of squared distances between the vectors
	sum_squared = K.sum(K.square(feats_A - feats_B), axis=1,
		keepdims=True)
	# return the euclidean distance between the vectors
	return K.sqrt(K.maximum(sum_squared, K.epsilon()))

def cosine_similarity(vectors):
    (featsA, featsB) = vectors
    sum_product = K.sum(featsA*featsB)
    sum_squared_featsA = K.sqrt(K.sum(featsA**2, keepdims=1, axis=1))
    sum_squared_featsB = K.sqrt(K.sum(featsB**2, keepdims=1, axis=1))
    sum_mul_feats = sum_squared_featsA * sum_squared_featsB

    return sum_product / sum_mul_feats
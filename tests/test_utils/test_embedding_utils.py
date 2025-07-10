from src.utils.embedding_utils import fourier_transform, pca_transform


def test_fourier_transform_length():
    data = [1, 2, 3, 4]
    coeffs = fourier_transform(data)
    # For length 4 -> (n/2 + 1)*2 = 6 coefficients
    assert isinstance(coeffs, list)
    assert len(coeffs) == 6


def test_pca_transform_shape():
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    transformed = pca_transform(data, n_components=2)
    assert isinstance(transformed, list)
    assert len(transformed) == 3
    assert len(transformed[0]) == 2

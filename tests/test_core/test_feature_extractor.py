from src.core.feature_extractor import FeatureExtractor


def test_extract_features_and_modular():
    extractor = FeatureExtractor()
    seq = [1, 2, 4]
    features = extractor.extract_features(seq)
    assert isinstance(features, list)
    assert len(features) == 3
    assert features[1]["diff_n"] == 1
    assert features[2]["diff_n"] == 2

    mods = extractor.modular_features(seq, 3)
    assert mods == [1, 2, 1]

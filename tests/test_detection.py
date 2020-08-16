import pytest

from data_describe.compat import presidio_analyzer
from data_describe.compat import _DATAFRAME_TYPE
from data_describe.privacy.detection import sensitive_data
from data_describe.backends.compute._pandas.detection import (
    identify_pii,
    create_mapping,
    redact_info,
    identify_column_infotypes,
    encrypt_text,
    hash_string,
)


def test_identify_pii():
    example_text = "This string contains a domain, gmail.com"
    response = identify_pii(example_text)
    assert isinstance(response, list)
    assert isinstance(response[0], presidio_analyzer.recognizer_result.RecognizerResult)
    assert len(response) == 1
    assert isinstance(response[0].entity_type, str)
    assert isinstance(response[0].start, int)
    assert isinstance(response[0].end, int)
    assert isinstance(response[0].score, float)
    assert response[0].entity_type == "DOMAIN_NAME"


def test_identify_column_infotypes(compute_backend_column_infotype):
    results = identify_column_infotypes(compute_backend_column_infotype, sample_size=1)
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], str)
    assert results[0] == "DOMAIN_NAME"


def test_identify_infotypes(compute_backend_pii_df):
    results = sensitive_data(
        compute_backend_pii_df, sample_size=1, detect_infotypes=True, redact=False
    )
    assert isinstance(results, dict)
    assert len(results) == 2
    assert isinstance(results["domain"], list)
    assert isinstance(results["name"], list)
    assert results["domain"][0] == "DOMAIN_NAME"
    assert results["name"][0] == "PERSON"


def test_create_mapping():
    example_text = "This string contains a domain gmail.com"
    response = identify_pii(example_text)
    word_mapping, text = create_mapping(example_text, response)
    assert isinstance(word_mapping, dict)
    assert isinstance(text, str)
    assert example_text != text


def test_redact_info():
    example_text = "This string contains a domain gmail.com"
    result_text = redact_info(example_text)
    assert isinstance(result_text, str)
    assert example_text != result_text
    assert result_text == "This string contains a domain <DOMAIN_NAME>"


def test_sensitive_data_cols(compute_backend_pii_df):
    redacted_df = sensitive_data(compute_backend_pii_df, redact=True, cols=["name"])
    assert redacted_df.shape == (1, 1)
    assert redacted_df.loc[1, "name"] == "<PERSON>"


def test_type_df_type(compute_backend_pii_text):
    with pytest.raises(TypeError):
        sensitive_data(compute_backend_pii_text)


def test_column_type(compute_backend_pii_df):
    with pytest.raises(TypeError):
        sensitive_data(compute_backend_pii_df, cols="this is not a list")


def test_sample_size(compute_backend_pii_df):
    with pytest.raises(ValueError):
        sensitive_data(
            compute_backend_pii_df, redact=False, detect_infotypes=True, sample_size=9
        )
    with pytest.raises(ValueError):
        sensitive_data(compute_backend_pii_df, redact=True, encrypt=True)


def test_sensitive_data_redact(compute_backend_pii_df):
    redacted_df = sensitive_data(compute_backend_pii_df, redact=True)
    assert redacted_df.shape == (1, 2)
    assert redacted_df.loc[1, "domain"] == "<DOMAIN_NAME>"
    assert redacted_df.loc[1, "name"] == "<PERSON>"
    assert isinstance(redacted_df, _DATAFRAME_TYPE)


def test_sensitive_data_detect_infotypes(compute_backend_pii_df):
    results = sensitive_data(
        compute_backend_pii_df, redact=False, detect_infotypes=True, sample_size=1
    )
    assert isinstance(results, dict)
    assert len(results) == 2
    assert isinstance(results["domain"], list)
    assert isinstance(results["name"], list)
    assert results["domain"][0] == "DOMAIN_NAME"
    assert results["name"][0] == "PERSON"


def test_encrypt_text():
    text = "gmail.com"
    encrypted = encrypt_text(text)
    assert text != encrypted
    assert isinstance(encrypted, str)


def test_encrypt_data(compute_backend_pii_df):
    encrypted_df = sensitive_data(compute_backend_pii_df, redact=False, encrypt=True)
    assert isinstance(encrypted_df, _DATAFRAME_TYPE)
    assert isinstance(encrypted_df.loc[1, "name"], str)
    assert isinstance(encrypted_df.loc[1, "domain"], str)


def test_hash_string():
    hashed = hash_string("John Doe")
    assert isinstance(hashed, str)
    assert len(hashed) == 64

"""Tests for HuggingFaceEndpoint."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.load import dumpd
from pydantic import SecretStr

from langchain_huggingface.llms.huggingface_endpoint import (
    HuggingFaceEndpoint,
    _is_huggingface_hosted_url,
)


class SerializedPayloadHandler(BaseCallbackHandler):
    """Capture serialized callback payloads."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        """Capture the model payload sent when an LLM starts."""
        self.payloads.append(serialized)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (None, False),
        ("", False),
        ("http://localhost:8010/", False),
        ("http://127.0.0.1:8080", False),
        ("http://my-tgi.internal/", False),
        ("https://api.inference-api.azure-api.net/", False),
        ("https://abc.huggingface.co/inference", True),
        ("https://xyz.hf.space/", True),
    ],
)
def test_is_huggingface_hosted_url(
    url: str | None,
    expected: bool,  # noqa: FBT001
) -> None:
    """URL helper: local/custom vs HF-hosted."""
    assert _is_huggingface_hosted_url(url) is expected


@patch(
    "huggingface_hub.AsyncInferenceClient",
)
@patch("huggingface_hub.InferenceClient")
def test_local_endpoint_does_not_pass_api_key(
    mock_inference_client: MagicMock,
    mock_async_client: MagicMock,
) -> None:
    """With a local endpoint_url we don't pass api_key so the client doesn't hit HF."""
    mock_inference_client.return_value = MagicMock()
    mock_async_client.return_value = MagicMock()

    HuggingFaceEndpoint(  # type: ignore[call-arg]
        endpoint_url="http://localhost:8010/",
        max_new_tokens=64,
    )

    mock_inference_client.assert_called_once()
    call_kwargs = mock_inference_client.call_args[1]
    assert call_kwargs.get("api_key") is None
    assert call_kwargs.get("model") == "http://localhost:8010/"

    mock_async_client.assert_called_once()
    async_call_kwargs = mock_async_client.call_args[1]
    assert async_call_kwargs.get("api_key") is None


@patch("huggingface_hub.AsyncInferenceClient")
@patch("huggingface_hub.InferenceClient")
def test_huggingface_hosted_endpoint_keeps_api_key(
    mock_inference_client: MagicMock,
    mock_async_client: MagicMock,
) -> None:
    """HF-hosted endpoint_url still gets the token."""
    mock_inference_client.return_value = MagicMock()
    mock_async_client.return_value = MagicMock()

    HuggingFaceEndpoint(  # type: ignore[call-arg]
        endpoint_url="https://abc.huggingface.co/inference",
        max_new_tokens=64,
        huggingfacehub_api_token="hf_xxx",  # noqa: S106
    )

    call_kwargs = mock_inference_client.call_args[1]
    assert call_kwargs.get("api_key") == "hf_xxx"


@patch("huggingface_hub.AsyncInferenceClient")
@patch("huggingface_hub.InferenceClient")
def test_api_token_is_excluded_from_callback_serialization(
    mock_inference_client: MagicMock,
    mock_async_client: MagicMock,
) -> None:
    """The raw token must not reach model-start callback payloads."""
    api_token = "hf_callback_secret"  # noqa: S105
    mock_inference_client.return_value.text_generation.return_value = "generated"
    mock_async_client.return_value = MagicMock()
    handler = SerializedPayloadHandler()

    model = HuggingFaceEndpoint(  # type: ignore[call-arg]
        repo_id="test/repo",
        huggingfacehub_api_token=api_token,
    )

    assert model.invoke("prompt", config={"callbacks": [handler]}) == "generated"
    assert isinstance(model.huggingfacehub_api_token, SecretStr)
    assert model.huggingfacehub_api_token.get_secret_value() == api_token
    assert mock_inference_client.call_args.kwargs["api_key"] == api_token
    assert api_token not in repr(model)
    assert api_token not in str(dumpd(model))
    assert len(handler.payloads) == 1
    assert api_token not in str(handler.payloads)
    assert "huggingfacehub_api_token" not in model.model_dump()
    assert "client" not in model.model_dump()
    assert "async_client" not in model.model_dump()


@patch("huggingface_hub.AsyncInferenceClient")
@patch("huggingface_hub.InferenceClient")
def test_huggingfacehub_environment_token_is_masked(
    mock_inference_client: MagicMock,
    mock_async_client: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The primary environment token must be masked while remaining usable."""
    api_token = "hf_environment_secret"  # noqa: S105
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", api_token)
    mock_inference_client.return_value = MagicMock()
    mock_async_client.return_value = MagicMock()

    model = HuggingFaceEndpoint(repo_id="test/repo")  # type: ignore[call-arg]

    assert isinstance(model.huggingfacehub_api_token, SecretStr)
    assert model.huggingfacehub_api_token.get_secret_value() == api_token
    assert mock_inference_client.call_args.kwargs["api_key"] == api_token
    assert mock_async_client.call_args.kwargs["api_key"] == api_token
    assert api_token not in repr(model)
    assert api_token not in str(dumpd(model))


@patch("huggingface_hub.AsyncInferenceClient")
@patch("huggingface_hub.InferenceClient")
def test_hf_token_fallback_is_used_without_serialization(
    mock_inference_client: MagicMock,
    mock_async_client: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The HF_TOKEN fallback remains usable without entering model serialization."""
    api_token = "hf_fallback_secret"  # noqa: S105
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", api_token)
    mock_inference_client.return_value = MagicMock()
    mock_async_client.return_value = MagicMock()

    model = HuggingFaceEndpoint(repo_id="test/repo")  # type: ignore[call-arg]

    assert model.huggingfacehub_api_token is None
    assert mock_inference_client.call_args.kwargs["api_key"] == api_token
    assert mock_async_client.call_args.kwargs["api_key"] == api_token
    assert api_token not in repr(model)
    assert api_token not in str(dumpd(model))

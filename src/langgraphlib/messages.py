"""Helpers for building multimodal messages (text + images)."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage

Provider = Literal["openai", "anthropic", "bedrock"]
ImageDetail = Literal["auto", "low", "high"]
ImageInput = str | bytes | Path

_MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

_MAGIC_NUMBERS = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),
)


def _is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _is_data_url(value: str) -> bool:
    return value.startswith("data:")


def _parse_data_url(data_url: str) -> tuple[str, str]:
    """Extract (mime_type, base64_data) from a data URL."""
    if not data_url.startswith("data:"):
        raise ValueError(f"Not a data URL: {data_url[:30]}...")

    header, data = data_url.split(",", 1)
    mime_type = header[5:].split(";")[0] or "image/jpeg"
    return mime_type, data


def _detect_mime_from_bytes(data: bytes) -> str:
    for signature, mime in _MAGIC_NUMBERS:
        if data.startswith(signature):
            return mime
    return "image/jpeg"


def _detect_mime_from_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _MIME_BY_EXT:
        return _MIME_BY_EXT[suffix]

    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "image/jpeg"


def _normalize_image(image: ImageInput) -> tuple[str, str | None, str | None]:
    """
    Normalize an image input into (kind, url, base64_data).

    Returns:
        kind: "url" or "base64"
        url: the URL (if kind == "url"), else None
        base64_data: the base64-encoded data (if kind == "base64"), else None
        mime_type: detected mime type (always present)

    Actually returns a 3-tuple where the third element is the mime_type
    when kind == "base64", and None when kind == "url".
    """
    if isinstance(image, bytes):
        mime = _detect_mime_from_bytes(image)
        encoded = base64.b64encode(image).decode("ascii")
        return "base64", encoded, mime

    if isinstance(image, Path):
        mime = _detect_mime_from_path(image)
        encoded = base64.b64encode(image.read_bytes()).decode("ascii")
        return "base64", encoded, mime

    if isinstance(image, str):
        if _is_data_url(image):
            mime, data = _parse_data_url(image)
            return "base64", data, mime

        if _is_url(image):
            return "url", image, None

        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image}")
        mime = _detect_mime_from_path(path)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return "base64", encoded, mime

    raise TypeError(
        f"Unsupported image input type: {type(image).__name__}. "
        "Expected str (URL, path or data URL), bytes, or Path."
    )


def _build_openai_block(image: ImageInput, detail: ImageDetail) -> dict[str, Any]:
    kind, value, mime = _normalize_image(image)
    url = value if kind == "url" else f"data:{mime};base64,{value}"

    return {
        "type": "image_url",
        "image_url": {"url": url, "detail": detail},
    }


def _build_anthropic_block(image: ImageInput) -> dict[str, Any]:
    kind, value, mime = _normalize_image(image)

    if kind == "url":
        source = {"type": "url", "url": value}
    else:
        source = {"type": "base64", "media_type": mime, "data": value}

    return {"type": "image", "source": source}


def create_image_message(
    text: str,
    images: ImageInput | list[ImageInput],
    *,
    provider: Provider = "openai",
    detail: ImageDetail = "auto",
) -> HumanMessage:
    """
    Build a multimodal HumanMessage with text and one or more images.

    Args:
        text: The text content sent alongside the images.
        images: A single image or list of images. Each image can be:
            - HTTP/HTTPS URL (str)
            - Data URL (str starting with "data:")
            - File path (str or pathlib.Path)
            - Raw bytes
        provider: Output format. Use the provider that matches the model
            consuming this message:
            - "openai": image_url block (works with langchain-openai)
            - "anthropic": image/source block (works with langchain-anthropic)
            - "bedrock": image/source block (works with ChatBedrockConverse)
        detail: Image detail level. Only used by "openai" provider.

    Returns:
        HumanMessage with multimodal content (text + image blocks).

    Raises:
        FileNotFoundError: If a string path does not point to an existing file.
        TypeError: If an image input has an unsupported type.

    Examples:
        from langgraphlib import create_image_message, Agent, MessagesState

        # From URL (OpenAI)
        msg = create_image_message(
            "What is in this picture?",
            "https://example.com/cat.jpg",
        )

        # From file path (Anthropic)
        msg = create_image_message(
            "Describe this chart.",
            "./chart.png",
            provider="anthropic",
        )

        # Multiple images from bytes (Bedrock)
        with open("a.jpg", "rb") as f:
            img1 = f.read()
        msg = create_image_message(
            "Compare these:",
            [img1, "https://example.com/b.png"],
            provider="bedrock",
        )

        result = agent.invoke(MessagesState(messages=[msg]))
    """
    if isinstance(images, str | bytes | Path):
        image_list: list[ImageInput] = [images]
    elif isinstance(images, list):
        image_list = images
    else:
        raise TypeError(
            f"`images` must be str, bytes, Path, or list of those — "
            f"got {type(images).__name__}."
        )

    builders = {
        "openai": lambda img: _build_openai_block(img, detail),
        "anthropic": _build_anthropic_block,
        "bedrock": _build_anthropic_block,
    }

    if provider not in builders:
        raise ValueError(
            f"Unsupported provider: {provider!r}. "
            f"Expected one of: {list(builders.keys())}"
        )

    build_block = builders[provider]
    content: list[str | dict[str, Any]] = [{"type": "text", "text": text}]
    content.extend(build_block(img) for img in image_list)

    return HumanMessage(content=content)

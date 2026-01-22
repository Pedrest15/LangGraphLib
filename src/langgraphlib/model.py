from typing import Any

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

# Load environment variables from .env
load_dotenv()


class ModelError(Exception):
    """Error when creating or configuring model."""

    pass


class ProviderNotInstalledError(ModelError):
    """Provider dependencies are not installed."""

    pass


def _parse_model_string(model_string: str) -> tuple[str, str]:
    """
    Parses string in 'provider/model_name' format.

    Args:
        model_string: String like "openai/gpt-4o" or "anthropic/claude-3-sonnet"

    Returns:
        Tuple (provider, model_name)

    Raises:
        ModelError: If format is invalid
    """
    if "/" not in model_string:
        raise ModelError(
            f"Invalid format: '{model_string}'. "
            "Use 'provider/model' (e.g.: 'openai/gpt-4o')"
        )

    parts = model_string.split("/", 1)
    return parts[0].lower(), parts[1]


def _get_provider_class(provider: str) -> tuple[type, dict[str, Any]]:
    """
    Dynamically imports the provider class.

    Returns:
        Tuple (model_class, extra_kwargs)
    """
    providers: dict[str, tuple[str, str, dict[str, Any]]] = {
        "openai": ("langchain_openai", "ChatOpenAI", {}),
        "anthropic": ("langchain_anthropic", "ChatAnthropic", {}),
        "groq": ("langchain_groq", "ChatGroq", {}),
        "ollama": (
            "langchain_ollama",
            "ChatOllama",
            {"base_url": "http://localhost:11434"},
        ),
        "cerebras": ("langchain_cerebras", "ChatCerebras", {}),
        "google": ("langchain_google_genai", "ChatGoogleGenerativeAI", {}),
        "mistral": ("langchain_mistralai", "ChatMistralAI", {}),
        "cohere": ("langchain_cohere", "ChatCohere", {}),
        "fireworks": ("langchain_fireworks", "ChatFireworks", {}),
        "together": ("langchain_together", "ChatTogether", {}),
    }

    if provider not in providers:
        available = list(providers.keys())
        raise ModelError(
            f"Provider '{provider}' not supported. Available: {available}"
        )

    module_name, class_name, default_kwargs = providers[provider]

    try:
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name), default_kwargs
    except ImportError as e:
        raise ProviderNotInstalledError(
            f"Provider '{provider}' requires installation: pip install {module_name}"
        ) from e


def _get_embedding_class(provider: str) -> tuple[type, dict[str, Any]]:
    """
    Dynamically imports the embedding class for the provider.

    Returns:
        Tuple (embedding_class, extra_kwargs)
    """
    providers: dict[str, tuple[str, str, dict[str, Any]]] = {
        "openai": ("langchain_openai", "OpenAIEmbeddings", {}),
        "ollama": (
            "langchain_ollama",
            "OllamaEmbeddings",
            {"base_url": "http://localhost:11434"},
        ),
        "google": ("langchain_google_genai", "GoogleGenerativeAIEmbeddings", {}),
        "cohere": ("langchain_cohere", "CohereEmbeddings", {}),
        "mistral": ("langchain_mistralai", "MistralAIEmbeddings", {}),
        "huggingface": ("langchain_huggingface", "HuggingFaceEmbeddings", {}),
    }

    if provider not in providers:
        available = list(providers.keys())
        raise ModelError(
            f"Embedding provider '{provider}' not supported. "
            f"Available: {available}"
        )

    module_name, class_name, default_kwargs = providers[provider]

    try:
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name), default_kwargs
    except ImportError as e:
        raise ProviderNotInstalledError(
            f"Provider '{provider}' requires installation: pip install {module_name}"
        ) from e


def get_model(
    model: str,
    *,
    api_key: str | None = None,
    temperature: float = 0,
    max_tokens: int | None = None,
    streaming: bool = False,
    **kwargs: Any,
) -> BaseChatModel:
    """
    Gets a configured chat model.

    Args:
        model: String in "provider/model" format (e.g.: "openai/gpt-4o")
        api_key: Provider API key (uses env var if not provided)
        temperature: Temperature for generation (0-2)
        max_tokens: Token limit in response
        streaming: Enable streaming
        **kwargs: Extra parameters passed to the LangChain model

    Returns:
        Model instance ready for use

    Examples:
        # OpenAI
        model = get_model("openai/gpt-4o", temperature=0.7)

        # Anthropic with explicit API key
        model = get_model("anthropic/claude-3-sonnet", api_key="sk-...")

        # Local Ollama
        model = get_model("ollama/llama3")

        # With streaming
        model = get_model("openai/gpt-4o-mini", streaming=True)
    """
    provider, model_name = _parse_model_string(model)
    model_class, default_kwargs = _get_provider_class(provider)

    # Build model kwargs
    model_kwargs = {
        **default_kwargs,
        "model": model_name,
        "temperature": temperature,
        "streaming": streaming,
        **kwargs,
    }

    # Add api_key if provided
    if api_key:
        # Different providers use different names for api_key
        api_key_names = {
            "openai": "api_key",
            "anthropic": "api_key",
            "groq": "api_key",
            "google": "google_api_key",
            "mistral": "api_key",
            "cohere": "cohere_api_key",
            "fireworks": "api_key",
            "together": "api_key",
            "cerebras": "api_key",
        }
        key_name = api_key_names.get(provider, "api_key")
        model_kwargs[key_name] = api_key

    # Add max_tokens if specified
    if max_tokens:
        model_kwargs["max_tokens"] = max_tokens

    return model_class(**model_kwargs)


def get_embeddings(
    model: str,
    *,
    api_key: str | None = None,
    **kwargs: Any,
) -> Embeddings:
    """
    Gets a configured embeddings model.

    Args:
        model: String in "provider/model" format (e.g.: "openai/text-embedding-3-small")
        api_key: Provider API key (uses env var if not provided)
        **kwargs: Extra parameters passed to the LangChain model

    Returns:
        Embeddings model instance ready for use

    Examples:
        # OpenAI
        embeddings = get_embeddings("openai/text-embedding-3-small")

        # Local Ollama
        embeddings = get_embeddings("ollama/nomic-embed-text")

        # Local HuggingFace
        embeddings = get_embeddings("huggingface/all-MiniLM-L6-v2")
    """
    provider, model_name = _parse_model_string(model)
    embedding_class, default_kwargs = _get_embedding_class(provider)

    # Build kwargs
    model_kwargs = {
        **default_kwargs,
        "model": model_name,
        **kwargs,
    }

    # Add api_key if provided
    if api_key:
        api_key_names = {
            "openai": "api_key",
            "google": "google_api_key",
            "cohere": "cohere_api_key",
            "mistral": "api_key",
        }
        key_name = api_key_names.get(provider, "api_key")
        model_kwargs[key_name] = api_key

    return embedding_class(**model_kwargs)


def list_providers() -> dict[str, list[str]]:
    """
    Lists available providers.

    Returns:
        Dictionary with chat and embedding providers
    """
    return {
        "chat": [
            "openai",
            "anthropic",
            "groq",
            "ollama",
            "cerebras",
            "google",
            "mistral",
            "cohere",
            "fireworks",
            "together",
        ],
        "embedding": [
            "openai",
            "ollama",
            "google",
            "cohere",
            "mistral",
            "huggingface",
        ],
    }

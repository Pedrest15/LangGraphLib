"""Interface simplificada para obter modelos de LLM e embeddings."""

from typing import Any

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

# Carrega variáveis de ambiente do .env
load_dotenv()


class ModelError(Exception):
    """Erro ao criar ou configurar modelo."""

    pass


class ProviderNotInstalledError(ModelError):
    """Dependências do provider não estão instaladas."""

    pass


def _parse_model_string(model_string: str) -> tuple[str, str]:
    """
    Parseia string no formato 'provider/model_name'.

    Args:
        model_string: String como "openai/gpt-4o" ou "anthropic/claude-3-sonnet"

    Returns:
        Tupla (provider, model_name)

    Raises:
        ModelError: Se formato inválido
    """
    if "/" not in model_string:
        raise ModelError(
            f"Formato inválido: '{model_string}'. "
            "Use 'provider/modelo' (ex: 'openai/gpt-4o')"
        )

    parts = model_string.split("/", 1)
    return parts[0].lower(), parts[1]


def _get_provider_class(provider: str) -> tuple[type, dict[str, Any]]:
    """
    Importa dinamicamente a classe do provider.

    Returns:
        Tupla (classe_do_modelo, kwargs_extras)
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
            f"Provider '{provider}' não suportado. Disponíveis: {available}"
        )

    module_name, class_name, default_kwargs = providers[provider]

    try:
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name), default_kwargs
    except ImportError as e:
        raise ProviderNotInstalledError(
            f"Provider '{provider}' requer instalação: pip install {module_name}"
        ) from e


def _get_embedding_class(provider: str) -> tuple[type, dict[str, Any]]:
    """
    Importa dinamicamente a classe de embedding do provider.

    Returns:
        Tupla (classe_do_embedding, kwargs_extras)
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
            f"Provider de embedding '{provider}' não suportado. "
            f"Disponíveis: {available}"
        )

    module_name, class_name, default_kwargs = providers[provider]

    try:
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name), default_kwargs
    except ImportError as e:
        raise ProviderNotInstalledError(
            f"Provider '{provider}' requer instalação: pip install {module_name}"
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
    Obtém um modelo de chat configurado.

    Args:
        model: String no formato "provider/modelo" (ex: "openai/gpt-4o")
        api_key: API key do provider (usa env var se não fornecida)
        temperature: Temperatura para geração (0-2)
        max_tokens: Limite de tokens na resposta
        streaming: Habilitar streaming
        **kwargs: Parâmetros extras passados ao modelo LangChain

    Returns:
        Instância do modelo pronta para uso

    Examples:
        # OpenAI
        model = get_model("openai/gpt-4o", temperature=0.7)

        # Anthropic com API key explícita
        model = get_model("anthropic/claude-3-sonnet", api_key="sk-...")

        # Ollama local
        model = get_model("ollama/llama3")

        # Com streaming
        model = get_model("openai/gpt-4o-mini", streaming=True)
    """
    provider, model_name = _parse_model_string(model)
    model_class, default_kwargs = _get_provider_class(provider)

    # Monta kwargs do modelo
    model_kwargs = {
        **default_kwargs,
        "model": model_name,
        "temperature": temperature,
        "streaming": streaming,
        **kwargs,
    }

    # Adiciona api_key se fornecida
    if api_key:
        # Diferentes providers usam nomes diferentes para api_key
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

    # Adiciona max_tokens se especificado
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
    Obtém um modelo de embeddings configurado.

    Args:
        model: String no formato "provider/modelo" (ex: "openai/text-embedding-3-small")
        api_key: API key do provider (usa env var se não fornecida)
        **kwargs: Parâmetros extras passados ao modelo LangChain

    Returns:
        Instância do modelo de embeddings pronta para uso

    Examples:
        # OpenAI
        embeddings = get_embeddings("openai/text-embedding-3-small")

        # Ollama local
        embeddings = get_embeddings("ollama/nomic-embed-text")

        # HuggingFace local
        embeddings = get_embeddings("huggingface/all-MiniLM-L6-v2")
    """
    provider, model_name = _parse_model_string(model)
    embedding_class, default_kwargs = _get_embedding_class(provider)

    # Monta kwargs
    model_kwargs = {
        **default_kwargs,
        "model": model_name,
        **kwargs,
    }

    # Adiciona api_key se fornecida
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
    Lista providers disponíveis.

    Returns:
        Dicionário com providers de chat e embedding
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

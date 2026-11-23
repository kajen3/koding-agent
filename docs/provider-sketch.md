# Provider sketch

Goal: the loop only sees `LlmCall` (`messages -> str`). A provider owns its
HTTP client, endpoint, request/response translation and error mapping.
Everything lives in `core/providers.py`; loop, TUI and `tests/fakes.py` stay
unchanged.

## 1. Config

One shared config: common fields + an open `options` dict.

```python
@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    base_url: str | None = None   # None -> provider default
    api_key: str | None = None    # None -> provider reads its own env var
    timeout: float = 120
    options: Mapping[str, Any] = field(default_factory=dict)  # provider-specific
```

- API key: the provider knows its env var (`API_KEY_ENV`); secrets stay out of argv.
- `options`: each provider places them where its API expects
  (Ollama: `"options"`, OpenAI-style: top-level fields).

## 2. Protocol

```python
class Provider(Protocol):
    async def complete(self, messages: list[Message]) -> str: ...  # == LlmCall
    async def aclose(self) -> None: ...
```

## 3. Shared HTTP helper

```python
async def post_json(client: AsyncClient, path: str, payload: dict[str, Any]) -> Any:
    # network error, HTTP error, invalid JSON -> LlmError
    # error detail: {"error": "..."} or {"error": {"message": "..."}}
```

## 4. Providers

```python
class OllamaProvider:
    DEFAULT_URL = "http://localhost:11434"

    def __init__(self, model, base_url=DEFAULT_URL, timeout=120, options=None, transport=None):
        self.model, self.options = model, dict(options or {})
        self._client = AsyncClient(base_url=base_url, timeout=timeout, transport=transport)

    @classmethod
    def from_config(cls, cfg: ProviderConfig) -> Self:
        return cls(cfg.model, cfg.base_url or cls.DEFAULT_URL, cfg.timeout, cfg.options)

    async def complete(self, messages):
        data = await post_json(self._client, "/api/chat", {
            "model": self.model, "stream": False, "messages": messages,
            "options": self.options,
        })
        return extract(data, lambda d: d["message"]["content"])  # bad shape -> LlmError

    async def aclose(self):
        await self._client.aclose()


class OpenRouterProvider:
    DEFAULT_URL = "https://openrouter.ai/api/v1"
    API_KEY_ENV = "OPENROUTER_API_KEY"

    @classmethod
    def from_config(cls, cfg: ProviderConfig) -> Self:
        key = cfg.api_key or os.environ.get(cls.API_KEY_ENV)
        if not key:
            raise LlmError(f"Set {cls.API_KEY_ENV} to use OpenRouter")
        ...  # Bearer auth header

    async def complete(self, messages):
        data = await post_json(self._client, "/chat/completions", {
            "model": self.model, "stream": False, "messages": messages,
            **self.options,
        })
        return extract(data, lambda d: d["choices"][0]["message"]["content"])
```

`transport` lets tests inject `httpx2.MockTransport`.

## 5. Registry + lifecycle

Plain dict, edited by hand next to the classes.

```python
PROVIDERS: dict[str, Callable[[ProviderConfig], Provider]] = {
    "ollama": OllamaProvider.from_config,
    "openrouter": OpenRouterProvider.from_config,
}

@asynccontextmanager
async def open_provider(cfg: ProviderConfig) -> AsyncIterator[Provider]:
    try:
        factory = PROVIDERS[cfg.provider]
    except KeyError:
        raise LlmError(f"Unknown provider {cfg.provider!r}; known: {sorted(PROVIDERS)}") from None
    provider = factory(cfg)
    try:
        yield provider
    finally:
        await provider.aclose()
```

## 6. Wiring

- `llm.py`: keep `Message`, `LlmCall`, `LlmError`, `DEFAULT_MODEL`; `call_ollama` moves into `OllamaProvider`.
- `core/__init__.py`: export `ProviderConfig`, `open_provider`; drop `llm_call`.
- `cli.py`:
  - `--provider` (default `ollama`), `--base-url`, repeatable `-o key=value`.
  - Ollama keeps `OLLAMA_HOST` as `base_url` fallback.
  - Replace `ollama_client()` + `partial(llm_call, ...)` with:

    ```python
    async with open_provider(cfg) as provider:
        ... run_agent(..., call=provider.complete)
    ```

## 7. Tests

- `test_llm.py` -> `test_providers.py`, per provider via `MockTransport`:
  path + payload (incl. `options`), reply parsing, each failure -> `LlmError`.
- Missing API key -> `LlmError`.
- Unknown provider -> `LlmError`.
- `open_provider` closes the client (also on exception).

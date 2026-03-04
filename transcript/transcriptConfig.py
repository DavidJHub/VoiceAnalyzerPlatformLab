import os


# Lanza un error claro si la variable no está presente
def _require(var: str) -> str:
    value = os.getenv(var)
    if not value:
        raise RuntimeError(f"Variable de entorno «{var}» no definida")
    return value

DEEPGRAM_API_KEY = _require("DEEPGRAM_API_KEY")
language = _require("language")
model = _require("model")
timeout_attempts = _require("timeout_attempts")

OPTION_TRANSCRIPT_ENGINE = os.getenv("OPTION_TRANSCRIPT_ENGINE", "DEEPGRAM")
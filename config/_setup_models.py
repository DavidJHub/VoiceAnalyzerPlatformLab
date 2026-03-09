"""
VAP - Model Setup Script
Called by vap_setup.bat after pip install.
Handles: pyannote (gated HF), Vosk, NLTK, spaCy, Presidio.
"""

import os
import sys
import shutil
import zipfile
import logging
import argparse
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("vap_setup")

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
VOSK_DIR = MODELS_DIR / "vosk"
MODELS_DIR.mkdir(exist_ok=True)
VOSK_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"), help="HuggingFace token")
parser.add_argument("--vosk-size", choices=["small", "large", "skip"], default="small",
                    help="Vosk Spanish model size to download")
parser.add_argument("--skip-pyannote", action="store_true")
parser.add_argument("--skip-vosk", action="store_true")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# 1. NLTK
# ---------------------------------------------------------------------------
def setup_nltk():
    log.info("── NLTK data")
    import nltk
    packages = [
        "stopwords", "punkt", "punkt_tab",
        "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
        "wordnet",
    ]
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
            log.info(f"   ✓ {pkg}")
        except Exception as e:
            log.warning(f"   ✗ {pkg}: {e}")


# ---------------------------------------------------------------------------
# 2. spaCy models
# ---------------------------------------------------------------------------
def setup_spacy():
    log.info("── spaCy models")
    import subprocess

    models = {
        "es_core_news_sm": "Spanish small (fast, basic NER)",
        "es_core_news_md": "Spanish medium (vectors 20k, better NER)",
        "es_core_news_lg": "Spanish large (vectors 500k, best for Presidio)",
    }
    for model, desc in models.items():
        try:
            import spacy
            spacy.load(model)
            log.info(f"   ✓ {model} already installed")
        except OSError:
            log.info(f"   ↓ Downloading {model} — {desc}")
            result = subprocess.run(
                [sys.executable, "-m", "spacy", "download", model],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                log.warning(f"   ✗ {model} failed: {result.stderr.strip()}")
            else:
                log.info(f"   ✓ {model} installed")


# ---------------------------------------------------------------------------
# 3. Presidio — register spaCy NLP engine for Spanish
# ---------------------------------------------------------------------------
def setup_presidio():
    log.info("── Presidio NLP engine config")
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        config = {
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "es", "model_name": "es_core_news_lg"},
                {"lang_code": "en", "model_name": "en_core_web_sm"},
            ],
        }
        # Validate en_core_web_sm availability (needed for Presidio default)
        import subprocess, spacy
        try:
            spacy.load("en_core_web_sm")
        except OSError:
            log.info("   ↓ Downloading en_core_web_sm (required by Presidio)")
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                capture_output=True
            )

        provider = NlpEngineProvider(nlp_configuration=config)
        provider.create_engine()  # validates config without full init
        log.info("   ✓ Presidio NLP engine config validated")

        # Write config file so the app can load it without rebuilding
        import json
        config_path = ROOT / "presidio_nlp_config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        log.info(f"   ✓ Config written to {config_path}")

    except Exception as e:
        log.warning(f"   ✗ Presidio setup failed: {e}")


# ---------------------------------------------------------------------------
# 4. pyannote models (GATED — requires HF token + accepted terms)
# ---------------------------------------------------------------------------
PYANNOTE_MODELS = [
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
    "pyannote/embedding",
]

def setup_pyannote(hf_token: str):
    log.info("── pyannote.audio models (HuggingFace gated)")

    if not hf_token:
        log.warning("   ✗ No HF_TOKEN provided — skipping pyannote model download.")
        log.warning("     Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1")
        log.warning("     Then re-run: python _setup_models.py --hf-token <YOUR_TOKEN>")
        return

    try:
        from huggingface_hub import snapshot_download, login
        login(token=hf_token, add_to_git_credential=False)
        log.info("   ✓ HuggingFace login OK")
    except Exception as e:
        log.warning(f"   ✗ HF login failed: {e}")
        return

    hf_cache = MODELS_DIR / "huggingface"
    hf_cache.mkdir(exist_ok=True)

    for model_id in PYANNOTE_MODELS:
        local_name = model_id.replace("/", "--")
        dest = hf_cache / local_name
        if dest.exists() and any(dest.iterdir()):
            log.info(f"   ✓ {model_id} already cached")
            continue
        log.info(f"   ↓ Downloading {model_id}")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=model_id,
                token=hf_token,
                local_dir=str(dest),
                local_dir_use_symlinks=False,
                ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
            )
            log.info(f"   ✓ {model_id} saved to {dest}")
        except Exception as e:
            log.error(f"   ✗ {model_id} download failed: {e}")
            log.error("     Make sure you accepted the model terms on HuggingFace.")

    # Write .env hint so the app knows the local cache path
    _write_env_hint("HF_MODELS_DIR", str(hf_cache))
    _write_env_hint("HF_TOKEN", hf_token)


# ---------------------------------------------------------------------------
# 5. Vosk Spanish model
# ---------------------------------------------------------------------------
VOSK_MODELS = {
    "small": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip",
        "folder": "vosk-model-small-es-0.42",
        "size": "39 MB",
    },
    "large": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-es-0.42.zip",
        "folder": "vosk-model-es-0.42",
        "size": "1.4 GB",
    },
}

def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"\r   [{bar}] {pct:5.1f}%  ({downloaded/1e6:.1f}/{total_size/1e6:.1f} MB)", end="", flush=True)

def setup_vosk(size: str):
    log.info(f"── Vosk Spanish model ({size})")
    meta = VOSK_MODELS[size]
    dest_dir = VOSK_DIR / meta["folder"]

    if dest_dir.exists() and any(dest_dir.iterdir()):
        log.info(f"   ✓ Already present at {dest_dir}")
        _write_env_hint("VOSK_MODEL_PATH", str(dest_dir))
        return

    zip_path = VOSK_DIR / f"vosk-es-{size}.zip"
    log.info(f"   ↓ Downloading {meta['size']} from {meta['url']}")

    try:
        urllib.request.urlretrieve(meta["url"], zip_path, reporthook=_progress_hook)
        print()  # newline after progress bar
    except Exception as e:
        log.error(f"   ✗ Download failed: {e}")
        return

    log.info(f"   Extracting to {VOSK_DIR}")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(VOSK_DIR)
        zip_path.unlink()
        log.info(f"   ✓ Vosk model at {dest_dir}")
        _write_env_hint("VOSK_MODEL_PATH", str(dest_dir))
    except Exception as e:
        log.error(f"   ✗ Extraction failed: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_env_hint(key: str, value: str):
    """Append key=value to .env if not already present."""
    env_path = ROOT / ".env"
    existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    if key not in existing:
        with env_path.open("a", encoding="utf-8") as f:
            f.write(f"\n{key}={value}\n")
        log.info(f"   → .env updated: {key}=...")


def check_java():
    log.info("── Java (required by tika)")
    if shutil.which("java"):
        result = os.popen("java -version 2>&1").read().strip()
        log.info(f"   ✓ {result.splitlines()[0] if result else 'java found'}")
    else:
        log.warning("   ✗ Java not found in PATH.")
        log.warning("     tika requires Java JRE 8+. Download: https://adoptium.net/")


def check_build_tools():
    log.info("── Visual C++ Build Tools (required by webrtcvad)")
    cl = shutil.which("cl")
    if cl:
        log.info(f"   ✓ MSVC compiler found: {cl}")
    else:
        log.warning("   ✗ MSVC cl.exe not found.")
        log.warning("     webrtcvad needs Build Tools: https://aka.ms/vs/17/release/vs_BuildTools.exe")
        log.warning("     Install workload: 'Desktop development with C++'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log.info("=" * 55)
    log.info(" VAP Model & Data Setup")
    log.info("=" * 55)

    check_java()
    check_build_tools()
    setup_nltk()
    setup_spacy()
    setup_presidio()

    if not args.skip_pyannote:
        setup_pyannote(args.hf_token)

    if not args.skip_vosk and args.vosk_size != "skip":
        setup_vosk(args.vosk_size)

    log.info("=" * 55)
    log.info(" Model setup complete")
    log.info("=" * 55)
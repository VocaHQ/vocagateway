from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from app.system import SystemInfo

# Integrity pins live beside the catalog rather than inline in the model table
# because they are machine-generated (scripts/harvest-model-pins.py) while the
# table is hand-written. Keeping them apart means regenerating pins produces a
# diff of nothing but digests, which is what makes them reviewable.
PINS_PATH = Path(__file__).parent / "model_pins.json"

ENGINE_WHISPER_CPP = "whisper.cpp"
ENGINE_WHISPERKIT = "whisperkit"
ENGINE_FASTER_WHISPER = "faster-whisper"
ENGINE_MOONSHINE = "moonshine"
ENGINE_SHERPA_ONNX = "sherpa-onnx"
ENGINE_MLX_AUDIO = "mlx-audio"

WHISPER_CPP_REPO = "ggerganov/whisper.cpp"
WHISPERKIT_REPO = "argmaxinc/whisperkit-coreml"

MB = 1_000_000
GB = 1_000_000_000
HIGH_MEMORY_RAM_GB = 12
VERY_HIGH_MEMORY_RAM_GB = 16
DEFAULT_RECOMMENDATION_RAM_GB = 8.0
ENGLISH_LANGUAGE_CODE = "en"
ARABIC_LANGUAGE_CODE = "ar"
BULGARIAN_LANGUAGE_CODE = "bg"
CZECH_LANGUAGE_CODE = "cs"
DANISH_LANGUAGE_CODE = "da"
GERMAN_LANGUAGE_CODE = "de"
GREEK_LANGUAGE_CODE = "el"
SPANISH_LANGUAGE_CODE = "es"
ESTONIAN_LANGUAGE_CODE = "et"
FINNISH_LANGUAGE_CODE = "fi"
FRENCH_LANGUAGE_CODE = "fr"
CROATIAN_LANGUAGE_CODE = "hr"
HUNGARIAN_LANGUAGE_CODE = "hu"
ITALIAN_LANGUAGE_CODE = "it"
JAPANESE_LANGUAGE_CODE = "ja"
KOREAN_LANGUAGE_CODE = "ko"
LITHUANIAN_LANGUAGE_CODE = "lt"
LATVIAN_LANGUAGE_CODE = "lv"
MALTESE_LANGUAGE_CODE = "mt"
DUTCH_LANGUAGE_CODE = "nl"
POLISH_LANGUAGE_CODE = "pl"
PORTUGUESE_LANGUAGE_CODE = "pt"
ROMANIAN_LANGUAGE_CODE = "ro"
RUSSIAN_LANGUAGE_CODE = "ru"
SLOVAK_LANGUAGE_CODE = "sk"
SLOVENIAN_LANGUAGE_CODE = "sl"
SWEDISH_LANGUAGE_CODE = "sv"
UKRAINIAN_LANGUAGE_CODE = "uk"
VIETNAMESE_LANGUAGE_CODE = "vi"
CHINESE_LANGUAGE_CODE = "zh"
MIT_LICENSE = "MIT"
ENGLISH_ONLY = "English only"
SHERPA_MODEL_FILE = "model.int8.onnx"
TOKENS_FILE = "tokens.txt"
ENCODER_INT8_FILE = "encoder.int8.onnx"
DECODER_INT8_FILE = "decoder.int8.onnx"
CC_BY_LICENSE = "CC BY 4.0"
APACHE_LICENSE = "Apache 2.0"
MULTILINGUAL = "Multilingual"
BASE_MODEL_VARIANT = "Base"
MOONSHINE_BASE_SIZE_MB = "141"
FAST_BATCH_QUALITY = "Fast · batch"
TINY_MODEL_SIZE_MB = "75"
FASTEST_QUALITY = "Fastest"
BASE_MODEL_SIZE_MB = "145"
FAST_QUALITY = "Fast"
BALANCED_QUALITY = "Balanced"
MOST_ACCURATE_QUALITY = "Most accurate"


def _megabytes(size_text: str) -> int:
    """Express catalog download sizes as readable decimal megabytes."""
    return int(size_text) * MB


def _gigabytes(size_text: str) -> int:
    """Express catalog download sizes as readable decimal gigabytes."""
    return int(size_text) * GB


@dataclass(frozen=True, slots=True)
class CatalogModel:
    """A downloadable speech-to-text model."""

    id: str
    engine: str
    key: str
    label: str
    size_bytes: int
    languages: str
    quality: str
    minimum_ram_gb: float
    download_url: str | None = None
    huggingface_repo: str | None = None
    huggingface_folder: str | None = None
    family: str = "Whisper"
    description: str = "Local speech recognition model."
    source: str = ENGINE_WHISPER_CPP
    marker_file: str | None = None
    language_code: str | None = None
    model_arch: int | None = None
    supports_streaming: bool = False
    license_name: str = "See model source"
    commercial_use: bool = True
    archive_url: str | None = None
    archive_root: str | None = None
    required_files: tuple[str, ...] = ()
    # Hugging Face commit to download from. Pinning a commit rather than
    # tracking `main` means the bytes cannot change under an existing catalog
    # entry: a re-upload upstream becomes a visible, reviewable catalog change
    # instead of a silent swap. `None` falls back to `main`.
    revision: str | None = None
    # Expected SHA-256 of the single file behind `download_url`, or of the
    # archive behind `archive_url`. Pinned here in git so verification survives
    # a compromised upstream, which TLS alone cannot defend against — the
    # attacker would be the origin and the certificate would be valid.
    sha256: str | None = None
    # (relative path, SHA-256) for models fetched as an explicit file list.
    # Paths are relative to the extracted archive root or the repo folder, and
    # need not cover every file; whatever is listed is enforced.
    file_digests: tuple[tuple[str, str], ...] = ()
    model_type: str | None = None
    language_codes: tuple[str, ...] = ()
    apple_silicon_only: bool = False
    # True when the model decides the language itself and offers no way to pin it.
    # `language_codes` then means "these are transcribed well", not "you may choose
    # one of these" — the app's language setting cannot constrain the result.
    detects_language_automatically: bool = False


def load_pins(path: Path = PINS_PATH) -> dict[str, dict[str, Any]]:
    """Read the generated integrity pins, tolerating their absence.

    A missing or unreadable pin file degrades to "nothing is pinned" rather
    than breaking startup: verification is a safety net over an already-TLS
    -protected download, so losing it must not take the gateway offline.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    models = payload.get("models") if isinstance(payload, dict) else None
    return models if isinstance(models, dict) else {}


def pin_download_url(url: str | None, revision: str | None) -> str | None:
    """Point a Hugging Face `/resolve/main/` URL at a specific commit.

    Single-file entries carry a full URL rather than a repo plus path, so the
    revision has to be substituted into the URL itself; without this the
    digest would be pinned while the bytes it names still tracked `main`.
    """
    if not url or not revision:
        return url
    marker = "/resolve/main/"
    if "huggingface.co/" not in url or marker not in url:
        return url
    return url.replace(marker, f"/resolve/{revision}/", 1)


def apply_pins(
    catalog: tuple[CatalogModel, ...], pins: dict[str, dict[str, Any]] | None = None
) -> tuple[CatalogModel, ...]:
    """Attach revisions and digests from the pin file to catalog entries."""
    records = load_pins() if pins is None else pins
    if not records:
        return catalog
    pinned: list[CatalogModel] = []
    for model in catalog:
        record = records.get(model.id)
        if not isinstance(record, dict):
            pinned.append(model)
            continue
        digests = record.get("file_digests")
        revision = record.get("revision") or model.revision
        pinned.append(
            replace(
                model,
                revision=revision,
                sha256=record.get("sha256") or model.sha256,
                download_url=pin_download_url(model.download_url, revision),
                file_digests=(
                    tuple(
                        sorted((str(key), str(entry_value)) for key, entry_value in digests.items())
                    )
                    if isinstance(digests, dict) and digests
                    else model.file_digests
                ),
            )
        )
    return tuple(pinned)


def _whisper_cpp(
    key: str,
    label: str,
    size_bytes: int,
    languages: str,
    quality: str,
    minimum_ram_gb: float,
    *,
    download_url: str | None = None,
    family: str = "Whisper",
    description: str = "OpenAI Whisper converted for the standalone whisper.cpp engine.",
    source: str = ENGINE_WHISPER_CPP,
    language_codes: tuple[str, ...] = (),
    license_name: str = "See model source",
) -> CatalogModel:
    return CatalogModel(
        id=f"{ENGINE_WHISPER_CPP}:{key}",
        engine=ENGINE_WHISPER_CPP,
        key=key,
        label=label,
        size_bytes=size_bytes,
        languages=languages,
        quality=quality,
        minimum_ram_gb=minimum_ram_gb,
        download_url=download_url
        or f"https://huggingface.co/{WHISPER_CPP_REPO}/resolve/main/{key}",
        family=family,
        description=description,
        source=source,
        language_codes=language_codes or _whisper_language_codes(languages),
        license_name=license_name,
    )


def _whisperkit(
    folder: str,
    label: str,
    size_bytes: int,
    languages: str,
    quality: str,
    minimum_ram_gb: float,
) -> CatalogModel:
    return CatalogModel(
        id=f"{ENGINE_WHISPERKIT}:{folder}",
        engine=ENGINE_WHISPERKIT,
        key=folder,
        label=label,
        size_bytes=size_bytes,
        languages=languages,
        quality=quality,
        minimum_ram_gb=minimum_ram_gb,
        huggingface_repo=WHISPERKIT_REPO,
        huggingface_folder=folder,
        family="Whisper",
        description="Core ML Whisper model optimized for Apple silicon.",
        source="WhisperKit",
        language_codes=_whisper_language_codes(languages),
    )


def _faster_whisper(
    key: str,
    label: str,
    size_bytes: int,
    languages: str,
    quality: str,
    minimum_ram_gb: float,
) -> CatalogModel:
    repository = (
        f"Systran/faster-distil-whisper-{key.removeprefix('distil-')}"
        if key.startswith("distil-")
        else f"Systran/faster-whisper-{key}"
    )
    return CatalogModel(
        id=f"{ENGINE_FASTER_WHISPER}:{key}",
        engine=ENGINE_FASTER_WHISPER,
        key=key,
        label=label,
        size_bytes=size_bytes,
        languages=languages,
        quality=quality,
        minimum_ram_gb=minimum_ram_gb,
        huggingface_repo=repository,
        huggingface_folder="",
        family="Whisper / CTranslate2",
        description=(
            "Persistent CTranslate2 model with CPU INT8 inference; "
            "works well on desktop and server CPUs."
        ),
        source="faster-whisper",
        marker_file="model.bin",
        language_codes=_whisper_language_codes(languages),
    )


def _moonshine(
    key: str,
    language: str,
    architecture: str,
    model_arch: int,
    label: str,
    size_bytes: int,
    quality: str,
    *,
    supports_streaming: bool = False,
    minimum_ram_gb: float = 2,
) -> CatalogModel:
    english = language == ENGLISH_LANGUAGE_CODE
    inference_description = (
        " Uses cached incremental inference while you speak."
        if supports_streaming
        else " Uses the fast batch pipeline after recording."
    )
    return CatalogModel(
        id=f"{ENGINE_MOONSHINE}:{key}",
        engine=ENGINE_MOONSHINE,
        key=key,
        label=label,
        size_bytes=size_bytes,
        languages=f"{_MOONSHINE_LANGUAGE_NAMES[language]} only",
        quality=quality,
        minimum_ram_gb=minimum_ram_gb,
        family="Moonshine",
        description=f"{architecture} model optimized for private local dictation."
        + inference_description,
        source="Moonshine Voice",
        marker_file=".vocagateway-model.json",
        language_code=language,
        # Also as a tuple: the engine reads `language_code`, but the model cards and
        # the language filter read `language_codes`, and an empty tuple there means
        # "covers everything" — which would list every English Moonshine under Hindi.
        language_codes=(language,),
        model_arch=model_arch,
        supports_streaming=supports_streaming,
        license_name=MIT_LICENSE if english else "Moonshine Community License",
        commercial_use=english,
    )


def _sherpa_onnx(
    key: str,
    label: str,
    size_bytes: int,
    languages: str,
    quality: str,
    minimum_ram_gb: float,
    *,
    required_files: tuple[str, ...],
    model_type: str,
    language_codes: tuple[str, ...],
    family: str,
    description: str,
    license_name: str,
    archive_url: str | None = None,
    archive_root: str | None = None,
    huggingface_repo: str | None = None,
    supports_streaming: bool = False,
    detects_language_automatically: bool = False,
) -> CatalogModel:
    """Build a sherpa-onnx catalog entry from either download mechanism.

    Most models ship as a `k2-fsa/sherpa-onnx` GitHub-release `.tar.bz2`
    (`archive_url`/`archive_root`). Some model families (GigaAM, Canary) are
    only published as individual files in a plain Hugging Face model repo with
    no such archive; for those, pass `huggingface_repo` instead and the
    gateway downloads exactly `required_files` from its root.
    """
    if archive_url is not None:
        if archive_root is None:
            raise ValueError(f"{key}: archive_url requires archive_root.")
    elif huggingface_repo is None:
        raise ValueError(f"{key}: provide either archive_url/archive_root or huggingface_repo.")
    return CatalogModel(
        id=f"{ENGINE_SHERPA_ONNX}:{key}",
        engine=ENGINE_SHERPA_ONNX,
        key=key,
        label=label,
        size_bytes=size_bytes,
        languages=languages,
        quality=quality,
        minimum_ram_gb=minimum_ram_gb,
        archive_url=archive_url,
        archive_root=archive_root,
        huggingface_repo=huggingface_repo,
        required_files=required_files,
        family=family,
        description=description,
        source="sherpa-onnx",
        marker_file=".vocagateway-model.json",
        model_type=model_type,
        language_codes=language_codes,
        license_name=license_name,
        supports_streaming=supports_streaming,
        detects_language_automatically=detects_language_automatically,
    )


def _mlx_audio(
    key: str,
    label: str,
    size_bytes: int,
    languages: str,
    quality: str,
    minimum_ram_gb: float,
    *,
    repository: str,
    family: str,
    description: str,
    license_name: str,
    language_codes: tuple[str, ...] = (),
) -> CatalogModel:
    return CatalogModel(
        id=f"{ENGINE_MLX_AUDIO}:{key}",
        engine=ENGINE_MLX_AUDIO,
        key=key,
        label=label,
        size_bytes=size_bytes,
        languages=languages,
        quality=quality,
        minimum_ram_gb=minimum_ram_gb,
        huggingface_repo=repository,
        huggingface_folder="",
        family=family,
        description=description,
        source="MLX Audio",
        marker_file="model.safetensors",
        language_codes=language_codes,
        apple_silicon_only=True,
        license_name=license_name,
    )


# Whisper's own language set, shared by every Whisper-derived entry (whisper.cpp,
# WhisperKit, faster-whisper, MLX Whisper). None of those engines validate against
# it — they pass the language straight to the CLI or library — so this is metadata
# for the model cards and the language filter, not a gate.
WHISPER_LANGUAGES: tuple[str, ...] = tuple(
    str.split(
        "af am ar as az ba be bg bn bo br bs ca cs cy da de el en es et eu fa fi fo fr gl gu ha "
        "haw he hi hr ht hu hy id is it ja jw ka kk km kn ko la lb ln lo lt lv mg mi mk ml mn mr "
        "ms mt my ne nl nn no oc pa pl ps pt ro ru sa sd si sk sl sn so sq sr su sv sw ta te tg th "
        "tk tl tr tt uk ur uz vi yi yo yue zh"
    )
)


def _whisper_language_codes(languages: str) -> tuple[str, ...]:
    """Derive a Whisper entry's codes from its human-readable summary."""
    return (ENGLISH_LANGUAGE_CODE,) if languages == ENGLISH_ONLY else WHISPER_LANGUAGES


# Display names for every code any catalog entry declares, so a model card can list
# "Hindi, Bengali, Tamil" instead of "hi, bn, ta". A missing code falls back to the
# code itself rather than hiding the language.
LANGUAGE_NAMES: dict[str, str] = {
    "af": "Afrikaans",
    "am": "Amharic",
    ARABIC_LANGUAGE_CODE: "Arabic",
    "as": "Assamese",
    "az": "Azerbaijani",
    "ba": "Bashkir",
    "be": "Belarusian",
    BULGARIAN_LANGUAGE_CODE: "Bulgarian",
    "bn": "Bengali",
    "bo": "Tibetan",
    "br": "Breton",
    "bs": "Bosnian",
    "ca": "Catalan",
    CZECH_LANGUAGE_CODE: "Czech",
    "ct": "Yue Chinese",
    "cy": "Welsh",
    DANISH_LANGUAGE_CODE: "Danish",
    GERMAN_LANGUAGE_CODE: "German",
    GREEK_LANGUAGE_CODE: "Greek",
    ENGLISH_LANGUAGE_CODE: "English",
    SPANISH_LANGUAGE_CODE: "Spanish",
    ESTONIAN_LANGUAGE_CODE: "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    FINNISH_LANGUAGE_CODE: "Finnish",
    "fil": "Filipino",
    "fo": "Faroese",
    FRENCH_LANGUAGE_CODE: "French",
    "gl": "Galician",
    "gu": "Gujarati",
    "ha": "Hausa",
    "haw": "Hawaiian",
    "he": "Hebrew",
    "hi": "Hindi",
    CROATIAN_LANGUAGE_CODE: "Croatian",
    "ht": "Haitian Creole",
    HUNGARIAN_LANGUAGE_CODE: "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "is": "Icelandic",
    ITALIAN_LANGUAGE_CODE: "Italian",
    JAPANESE_LANGUAGE_CODE: "Japanese",
    "jv": "Javanese",
    "jw": "Javanese",
    "ka": "Georgian",
    "kab": "Kabyle",
    "kk": "Kazakh",
    "km": "Khmer",
    "kn": "Kannada",
    KOREAN_LANGUAGE_CODE: "Korean",
    "ks": "Kashmiri",
    "ky": "Kyrgyz",
    "la": "Latin",
    "lb": "Luxembourgish",
    "ln": "Lingala",
    "lo": "Lao",
    LITHUANIAN_LANGUAGE_CODE: "Lithuanian",
    LATVIAN_LANGUAGE_CODE: "Latvian",
    "mg": "Malagasy",
    "mi": "Maori",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    MALTESE_LANGUAGE_CODE: "Maltese",
    "my": "Burmese",
    "ne": "Nepali",
    DUTCH_LANGUAGE_CODE: "Dutch",
    "nn": "Norwegian Nynorsk",
    "no": "Norwegian",
    "oc": "Occitan",
    "or": "Odia",
    "pa": "Punjabi",
    POLISH_LANGUAGE_CODE: "Polish",
    "ps": "Pashto",
    PORTUGUESE_LANGUAGE_CODE: "Portuguese",
    ROMANIAN_LANGUAGE_CODE: "Romanian",
    RUSSIAN_LANGUAGE_CODE: "Russian",
    "sa": "Sanskrit",
    "sd": "Sindhi",
    "si": "Sinhala",
    SLOVAK_LANGUAGE_CODE: "Slovak",
    SLOVENIAN_LANGUAGE_CODE: "Slovenian",
    "sn": "Shona",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "su": "Sundanese",
    SWEDISH_LANGUAGE_CODE: "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "th": "Thai",
    "tk": "Turkmen",
    "tl": "Tagalog",
    "tr": "Turkish",
    "tt": "Tatar",
    "ug": "Uyghur",
    UKRAINIAN_LANGUAGE_CODE: "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    VIETNAMESE_LANGUAGE_CODE: "Vietnamese",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "yue": "Cantonese",
    CHINESE_LANGUAGE_CODE: "Mandarin Chinese",
}


def catalog_source_url(model: CatalogModel) -> str | None:
    """Best public page for a catalog entry (Hugging Face repo or project site).

    Prefers a browsable page over a raw .tar/.bin download blob.
    """
    if model.huggingface_repo:
        return f"https://huggingface.co/{model.huggingface_repo}"
    if model.download_url and "huggingface.co/" in model.download_url:
        url = model.download_url
        if "/resolve/" in url:
            head, _, _ = url.partition("/resolve/")
            return head
        return url
    # Release/tag pages are more specific than a project root (e.g. SenseVoice /
    # Parakeet v3 ship only as sherpa-onnx GitHub release assets).
    release = _github_release_page(model.archive_url)
    if release:
        return release
    # Label-specific pages (Handy, Breeze, …) before generic engine fallbacks —
    # otherwise Handy builds incorrectly link to whisper.cpp's GitHub.
    labeled = _SOURCE_LABEL_URLS.get(model.source)
    if labeled:
        return labeled
    project = _ENGINE_SOURCE_URLS.get(model.engine)
    if project:
        return project
    if model.download_url:
        return model.download_url
    if model.archive_url:
        return model.archive_url
    return None


def _github_release_page(archive_url: str | None) -> str | None:
    """Turn a GitHub release asset URL into the browsable release/tag page."""
    if not archive_url or "/releases/download/" not in archive_url:
        return None
    head, _, rest = archive_url.partition("/releases/download/")
    if not head.startswith("https://github.com/") or not rest:
        return None
    tag = rest.split("/", maxsplit=1)[0]
    return f"{head}/releases/tag/{tag}" if tag else None


_ENGINE_SOURCE_URLS = {
    ENGINE_WHISPER_CPP: "https://github.com/ggml-org/whisper.cpp",
    ENGINE_WHISPERKIT: "https://github.com/argmaxinc/WhisperKit",
    ENGINE_FASTER_WHISPER: "https://github.com/SYSTRAN/faster-whisper",
    ENGINE_MOONSHINE: "https://github.com/moonshine-ai/moonshine",
    ENGINE_SHERPA_ONNX: "https://github.com/k2-fsa/sherpa-onnx",
    ENGINE_MLX_AUDIO: "https://github.com/Blaizzy/mlx-audio",
}

_SOURCE_LABEL_URLS = {
    ENGINE_WHISPER_CPP: "https://github.com/ggml-org/whisper.cpp",
    "faster-whisper": "https://github.com/SYSTRAN/faster-whisper",
    "WhisperKit": "https://github.com/argmaxinc/WhisperKit",
    "Moonshine Voice": "https://github.com/moonshine-ai/moonshine",
    "sherpa-onnx": "https://github.com/k2-fsa/sherpa-onnx",
    "MLX Audio": "https://github.com/Blaizzy/mlx-audio",
    # Hosted on Handy's CDN; the product page is the right "source", not whisper.cpp.
    "Handy-compatible": "https://handy.computer",
    "Breeze ASR": "https://huggingface.co/MediaTek-Research/Breeze-ASR-25",
}


def language_names(codes: tuple[str, ...]) -> list[str]:
    """Human-readable names for a model's languages, in the order declared."""
    return [LANGUAGE_NAMES.get(code, code) for code in codes]


# Dolphin's own language codes, from DataoceanAI/Dolphin `languages.md`. Two are not
# ISO 639-1: `ct` is Yue Chinese (`yue` elsewhere in this catalog) and `fil` is Filipino.
_DOLPHIN_LANGUAGE_CODES: tuple[str, ...] = (
    CHINESE_LANGUAGE_CODE,
    JAPANESE_LANGUAGE_CODE,
    "th",
    RUSSIAN_LANGUAGE_CODE,
    KOREAN_LANGUAGE_CODE,
    "id",
    VIETNAMESE_LANGUAGE_CODE,
    "ct",
    "hi",
    "ur",
    "ms",
    "uz",
    ARABIC_LANGUAGE_CODE,
    "fa",
    "bn",
    "ta",
    "te",
    "ug",
    "gu",
    "my",
    "tl",
    "kk",
    "or",
    "ne",
    "mn",
    "km",
    "jv",
    "lo",
    "si",
    "fil",
    "ps",
    "pa",
    "kab",
    "ba",
    "ks",
    "tg",
    "su",
    "mr",
    "ky",
    "az",
)


_MOONSHINE_LANGUAGE_NAMES = {
    ARABIC_LANGUAGE_CODE: "Arabic",
    ENGLISH_LANGUAGE_CODE: "English",
    SPANISH_LANGUAGE_CODE: "Spanish",
    JAPANESE_LANGUAGE_CODE: "Japanese",
    KOREAN_LANGUAGE_CODE: "Korean",
    UKRAINIAN_LANGUAGE_CODE: "Ukrainian",
    VIETNAMESE_LANGUAGE_CODE: "Vietnamese",
    CHINESE_LANGUAGE_CODE: "Mandarin Chinese",
}


_BASE_CATALOG: tuple[CatalogModel, ...] = (
    _sherpa_onnx(
        "sensevoice-small-int8",
        "SenseVoice Small INT8",
        _megabytes("240"),
        "Mandarin, Cantonese, English, Japanese, Korean",
        "Fastest multilingual · punctuation",
        2,
        archive_url=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
            "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2"
        ),
        archive_root="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17",
        required_files=(SHERPA_MODEL_FILE, TOKENS_FILE),
        model_type="sense_voice",
        language_codes=(
            CHINESE_LANGUAGE_CODE,
            "yue",
            ENGLISH_LANGUAGE_CODE,
            JAPANESE_LANGUAGE_CODE,
            KOREAN_LANGUAGE_CODE,
        ),
        family="SenseVoice",
        description=(
            "Compact non-autoregressive INT8 model for fast CPU dictation on Linux and macOS."
        ),
        license_name="FunASR Model License",
        # Loaded with language="auto"; this build exposes no per-stream override.
        detects_language_automatically=True,
    ),
    _sherpa_onnx(
        "parakeet-tdt-0.6b-v3-int8",
        "Parakeet TDT 0.6B v3 INT8",
        _megabytes("672"),
        "25 European languages",
        "Accurate multilingual · punctuation",
        4,
        archive_url=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
            "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"
        ),
        archive_root="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
        required_files=(
            ENCODER_INT8_FILE,
            DECODER_INT8_FILE,
            "joiner.int8.onnx",
            TOKENS_FILE,
        ),
        model_type="nemo_transducer",
        language_codes=(
            BULGARIAN_LANGUAGE_CODE,
            CROATIAN_LANGUAGE_CODE,
            CZECH_LANGUAGE_CODE,
            DANISH_LANGUAGE_CODE,
            DUTCH_LANGUAGE_CODE,
            ENGLISH_LANGUAGE_CODE,
            ESTONIAN_LANGUAGE_CODE,
            FINNISH_LANGUAGE_CODE,
            FRENCH_LANGUAGE_CODE,
            GERMAN_LANGUAGE_CODE,
            GREEK_LANGUAGE_CODE,
            HUNGARIAN_LANGUAGE_CODE,
            ITALIAN_LANGUAGE_CODE,
            LATVIAN_LANGUAGE_CODE,
            LITHUANIAN_LANGUAGE_CODE,
            MALTESE_LANGUAGE_CODE,
            POLISH_LANGUAGE_CODE,
            PORTUGUESE_LANGUAGE_CODE,
            ROMANIAN_LANGUAGE_CODE,
            SLOVAK_LANGUAGE_CODE,
            SLOVENIAN_LANGUAGE_CODE,
            SPANISH_LANGUAGE_CODE,
            SWEDISH_LANGUAGE_CODE,
            RUSSIAN_LANGUAGE_CODE,
            UKRAINIAN_LANGUAGE_CODE,
        ),
        family="Parakeet TDT",
        description=(
            "NVIDIA's multilingual Parakeet converted to INT8 ONNX for fast macOS and Linux CPU "
            "inference."
        ),
        license_name=CC_BY_LICENSE,
    ),
    _sherpa_onnx(
        "parakeet-tdt-0.6b-v2-int8",
        "Parakeet TDT 0.6B v2 INT8",
        _megabytes("661"),
        ENGLISH_ONLY,
        "Most accurate English · punctuation",
        4,
        huggingface_repo="csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
        required_files=(
            ENCODER_INT8_FILE,
            DECODER_INT8_FILE,
            "joiner.int8.onnx",
            TOKENS_FILE,
        ),
        model_type="nemo_transducer",
        language_codes=(ENGLISH_LANGUAGE_CODE,),
        family="Parakeet TDT",
        description=(
            "The English-only Parakeet. v3 trades some English accuracy for 25-language coverage, "
            "so this earlier release still transcribes English more accurately at the same speed."
        ),
        license_name=CC_BY_LICENSE,
    ),
    _sherpa_onnx(
        "gigaam-v3-ctc-russian-int8",
        "GigaAM v3 CTC Russian INT8",
        _megabytes("225"),
        "Russian only",
        "Fastest Russian ASR",
        2,
        huggingface_repo="csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16",
        required_files=(SHERPA_MODEL_FILE, TOKENS_FILE),
        model_type="nemo_ctc",
        language_codes=(RUSSIAN_LANGUAGE_CODE,),
        family="GigaAM",
        description=(
            "Sber's GigaAM CTC converted to INT8 ONNX for fast Russian-only CPU transcription."
        ),
        license_name=MIT_LICENSE,
    ),
    _sherpa_onnx(
        "gigaam-v3-rnnt-russian-int8",
        "GigaAM v3 RNNT Russian",
        _megabytes("230"),
        "Russian only",
        "Most accurate Russian ASR",
        2,
        huggingface_repo="csukuangfj/sherpa-onnx-nemo-transducer-giga-am-v3-russian-2025-12-16",
        required_files=(ENCODER_INT8_FILE, "decoder.onnx", "joiner.onnx", TOKENS_FILE),
        model_type="nemo_transducer",
        language_codes=(RUSSIAN_LANGUAGE_CODE,),
        family="GigaAM",
        description=(
            "Sber's GigaAM RNNT converted to ONNX for the most accurate Russian-only CPU "
            "transcription; only its encoder is INT8-quantized, so it is larger and slower "
            "than the CTC variant."
        ),
        license_name=MIT_LICENSE,
    ),
    _sherpa_onnx(
        "canary-180m-flash-en-int8",
        "Canary 180M Flash English INT8",
        _megabytes("210"),
        "English only in this build",
        "Compact multilingual model, English transcription",
        2,
        huggingface_repo="csukuangfj/sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8",
        required_files=(ENCODER_INT8_FILE, DECODER_INT8_FILE, TOKENS_FILE),
        model_type="nemo_canary",
        language_codes=(ENGLISH_LANGUAGE_CODE,),
        family="Canary",
        description=(
            "NVIDIA's Canary 180M Flash converted to INT8 ONNX. The underlying model also "
            "covers German, French, and Spanish, but its source/target language is fixed when "
            "the recognizer loads rather than per request, so vocaphone loads it English-only "
            "for now."
        ),
        license_name=CC_BY_LICENSE,
    ),
    _sherpa_onnx(
        "streaming-zipformer-en-20m-int8",
        "Streaming Zipformer English 20M INT8",
        _megabytes("44"),
        ENGLISH_ONLY,
        "Fastest live streaming",
        1,
        huggingface_repo="csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17",
        required_files=(
            "encoder-epoch-99-avg-1.int8.onnx",
            "decoder-epoch-99-avg-1.int8.onnx",
            "joiner-epoch-99-avg-1.int8.onnx",
            TOKENS_FILE,
        ),
        model_type="streaming_zipformer",
        language_codes=(ENGLISH_LANGUAGE_CODE,),
        family="Zipformer",
        description=(
            "A small streaming-capable zipformer transducer. Unlike most batch sherpa-onnx "
            "models, this one decodes incrementally with real partial results while you speak."
        ),
        license_name=APACHE_LICENSE,
        supports_streaming=True,
    ),
    _sherpa_onnx(
        "dolphin-small-ctc-int8",
        "Dolphin Small CTC INT8",
        _megabytes("250"),
        "40 Eastern languages",
        "Accurate · South, East and Southeast Asian",
        2,
        huggingface_repo="csukuangfj/sherpa-onnx-dolphin-small-ctc-multi-lang-int8-2025-04-02",
        required_files=(SHERPA_MODEL_FILE, TOKENS_FILE),
        model_type="dolphin_ctc",
        language_codes=_DOLPHIN_LANGUAGE_CODES,
        family="Dolphin",
        description=(
            "DataoceanAI and Tsinghua's model for Eastern languages, converted to INT8 ONNX. The "
            "only entry in this catalog that covers Hindi, Bengali, Tamil, Urdu and the other "
            "South Asian languages, and the most accurate of them on a full sentence. It "
            "detects the language itself and cannot be pinned, and on a short phrase that "
            "detection fails outright — a two-word Hindi clip can come back in Cyrillic. "
            "Dictate whole sentences, or choose a Whisper model for a guaranteed language."
        ),
        license_name=APACHE_LICENSE,
        detects_language_automatically=True,
    ),
    _sherpa_onnx(
        "dolphin-base-ctc-int8",
        "Dolphin Base CTC INT8",
        _megabytes("104"),
        "40 Eastern languages",
        "Fast · South, East and Southeast Asian",
        1,
        huggingface_repo="csukuangfj/sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02",
        required_files=(SHERPA_MODEL_FILE, TOKENS_FILE),
        model_type="dolphin_ctc",
        language_codes=_DOLPHIN_LANGUAGE_CODES,
        family="Dolphin",
        description=(
            "The compact Dolphin build, with the same 40-language coverage as the small variant "
            "at roughly half the accuracy cost of its size. It detects the language itself and "
            "cannot be pinned to one, and confuses related languages more often than the small "
            "variant does."
        ),
        license_name=APACHE_LICENSE,
        detects_language_automatically=True,
    ),
    _sherpa_onnx(
        "qwen3-asr-0.6b-int8",
        "Qwen3-ASR 0.6B INT8",
        _megabytes("987"),
        "11 languages",
        "Accurate multilingual · punctuation",
        6,
        huggingface_repo="csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25",
        # This family reads a Hugging Face tokenizer directory rather than a `tokens.txt`,
        # so the marker check and the recognizer both look under `tokenizer/`.
        required_files=(
            "conv_frontend.onnx",
            ENCODER_INT8_FILE,
            DECODER_INT8_FILE,
            "tokenizer/vocab.json",
            "tokenizer/merges.txt",
            "tokenizer/tokenizer_config.json",
        ),
        model_type="qwen3_asr",
        language_codes=(
            ENGLISH_LANGUAGE_CODE,
            CHINESE_LANGUAGE_CODE,
            JAPANESE_LANGUAGE_CODE,
            KOREAN_LANGUAGE_CODE,
            SPANISH_LANGUAGE_CODE,
            FRENCH_LANGUAGE_CODE,
            GERMAN_LANGUAGE_CODE,
            RUSSIAN_LANGUAGE_CODE,
            ARABIC_LANGUAGE_CODE,
            ITALIAN_LANGUAGE_CODE,
            PORTUGUESE_LANGUAGE_CODE,
        ),
        family="Qwen3-ASR",
        description=(
            "Alibaba's speech-aware Qwen3 converted to INT8 ONNX. An LLM decoder rather than a "
            "CTC or transducer head, so it punctuates well but decodes more slowly. It detects "
            "the language itself and cannot be pinned to one."
        ),
        license_name=APACHE_LICENSE,
        detects_language_automatically=True,
    ),
    _mlx_audio(
        "whisper-large-v3-turbo-4bit",
        "MLX Whisper Large v3 Turbo 4-bit",
        _megabytes("469"),
        MULTILINGUAL,
        "Most accurate · compact",
        8,
        repository="mlx-community/whisper-large-v3-turbo-asr-4bit",
        language_codes=WHISPER_LANGUAGES,
        family="Whisper / MLX",
        description=(
            "Quantized Whisper Large v3 Turbo running natively on Apple silicon through MLX."
        ),
        license_name=MIT_LICENSE,
    ),
    _mlx_audio(
        "parakeet-tdt-0.6b-v3",
        "MLX Parakeet TDT 0.6B v3",
        _megabytes("2510"),
        "25 European languages",
        "Fast and accurate · punctuation",
        8,
        repository="mlx-community/parakeet-tdt-0.6b-v3",
        family="Parakeet TDT / MLX",
        description=(
            "Full-precision Parakeet optimized for the unified memory and GPU of M-series Macs."
        ),
        license_name=CC_BY_LICENSE,
        language_codes=(
            BULGARIAN_LANGUAGE_CODE,
            CROATIAN_LANGUAGE_CODE,
            CZECH_LANGUAGE_CODE,
            DANISH_LANGUAGE_CODE,
            DUTCH_LANGUAGE_CODE,
            ENGLISH_LANGUAGE_CODE,
            ESTONIAN_LANGUAGE_CODE,
            FINNISH_LANGUAGE_CODE,
            FRENCH_LANGUAGE_CODE,
            GERMAN_LANGUAGE_CODE,
            GREEK_LANGUAGE_CODE,
            HUNGARIAN_LANGUAGE_CODE,
            ITALIAN_LANGUAGE_CODE,
            LATVIAN_LANGUAGE_CODE,
            LITHUANIAN_LANGUAGE_CODE,
            MALTESE_LANGUAGE_CODE,
            POLISH_LANGUAGE_CODE,
            PORTUGUESE_LANGUAGE_CODE,
            ROMANIAN_LANGUAGE_CODE,
            SLOVAK_LANGUAGE_CODE,
            SLOVENIAN_LANGUAGE_CODE,
            SPANISH_LANGUAGE_CODE,
            SWEDISH_LANGUAGE_CODE,
            RUSSIAN_LANGUAGE_CODE,
            UKRAINIAN_LANGUAGE_CODE,
        ),
    ),
    _mlx_audio(
        "parakeet-tdt-0.6b-v2",
        "MLX Parakeet TDT 0.6B v2",
        _megabytes("2472"),
        ENGLISH_ONLY,
        "Most accurate English · punctuation",
        8,
        repository="mlx-community/parakeet-tdt-0.6b-v2",
        family="Parakeet TDT / MLX",
        description=(
            "The English-only Parakeet on Apple silicon. More accurate on English than the "
            "multilingual v3 build, which spends capacity on 24 other languages."
        ),
        license_name=CC_BY_LICENSE,
        language_codes=(ENGLISH_LANGUAGE_CODE,),
    ),
    _mlx_audio(
        "qwen3-asr-0.6b-4bit",
        "MLX Qwen3-ASR 0.6B 4-bit",
        _megabytes("713"),
        "11 languages",
        "Accurate multilingual · punctuation",
        8,
        repository="mlx-community/Qwen3-ASR-0.6B-4bit",
        family="Qwen3-ASR / MLX",
        description=(
            "Quantized Qwen3-ASR running natively on Apple silicon through MLX. An LLM decoder, "
            "so it punctuates well but decodes more slowly than Parakeet."
        ),
        license_name=APACHE_LICENSE,
        language_codes=(
            ENGLISH_LANGUAGE_CODE,
            CHINESE_LANGUAGE_CODE,
            JAPANESE_LANGUAGE_CODE,
            KOREAN_LANGUAGE_CODE,
            SPANISH_LANGUAGE_CODE,
            FRENCH_LANGUAGE_CODE,
            GERMAN_LANGUAGE_CODE,
            RUSSIAN_LANGUAGE_CODE,
            ARABIC_LANGUAGE_CODE,
            ITALIAN_LANGUAGE_CODE,
            PORTUGUESE_LANGUAGE_CODE,
        ),
    ),
    _mlx_audio(
        "qwen3-asr-1.7b-4bit",
        "MLX Qwen3-ASR 1.7B 4-bit",
        _megabytes("1608"),
        "11 languages",
        "Most accurate multilingual · punctuation",
        HIGH_MEMORY_RAM_GB,
        repository="mlx-community/Qwen3-ASR-1.7B-4bit",
        family="Qwen3-ASR / MLX",
        description=(
            "The larger Qwen3-ASR for Macs with memory to spare; the same 11 languages as the "
            "0.6B entry, with better accuracy on accented and noisy speech."
        ),
        license_name=APACHE_LICENSE,
        language_codes=(
            ENGLISH_LANGUAGE_CODE,
            CHINESE_LANGUAGE_CODE,
            JAPANESE_LANGUAGE_CODE,
            KOREAN_LANGUAGE_CODE,
            SPANISH_LANGUAGE_CODE,
            FRENCH_LANGUAGE_CODE,
            GERMAN_LANGUAGE_CODE,
            RUSSIAN_LANGUAGE_CODE,
            ARABIC_LANGUAGE_CODE,
            ITALIAN_LANGUAGE_CODE,
            PORTUGUESE_LANGUAGE_CODE,
        ),
    ),
    _mlx_audio(
        "granite-speech-4.1-2b-nar",
        "MLX Granite Speech 4.1 2B",
        _megabytes("2377"),
        ENGLISH_ONLY,
        "Most accurate English",
        HIGH_MEMORY_RAM_GB,
        repository="mlx-community/granite-speech-4.1-2b-nar-mlx-5bit",
        family="Granite Speech / MLX",
        description=(
            "IBM's Granite Speech, quantized for Apple silicon. Its non-autoregressive decoder "
            "keeps it fast for a model of this size, and it sits at the top of the open English "
            "accuracy rankings."
        ),
        license_name=APACHE_LICENSE,
        language_codes=(ENGLISH_LANGUAGE_CODE,),
    ),
    # Keep moonshine:en as the default English ID so existing installations and
    # runtime configuration continue to resolve after adding explicit variants.
    _moonshine(
        ENGLISH_LANGUAGE_CODE,
        ENGLISH_LANGUAGE_CODE,
        "Medium Streaming",
        5,
        "Moonshine English Medium Streaming",
        _megabytes("304"),
        "Most accurate · cached streaming",
        supports_streaming=True,
        minimum_ram_gb=4,
    ),
    _moonshine(
        "en-small-streaming",
        ENGLISH_LANGUAGE_CODE,
        "Small Streaming",
        4,
        "Moonshine English Small Streaming",
        _megabytes("165"),
        "Balanced · cached streaming",
        supports_streaming=True,
    ),
    _moonshine(
        "en-tiny-streaming",
        ENGLISH_LANGUAGE_CODE,
        "Tiny Streaming",
        2,
        "Moonshine English Tiny Streaming",
        _megabytes("52"),
        "Fastest · cached streaming",
        supports_streaming=True,
    ),
    _moonshine(
        "en-base",
        ENGLISH_LANGUAGE_CODE,
        BASE_MODEL_VARIANT,
        1,
        "Moonshine English Base",
        _megabytes(MOONSHINE_BASE_SIZE_MB),
        "Accurate · batch",
    ),
    _moonshine(
        "en-tiny",
        ENGLISH_LANGUAGE_CODE,
        "Tiny",
        0,
        "Moonshine English Tiny",
        _megabytes("44"),
        "Smallest · batch",
    ),
    _moonshine(
        SPANISH_LANGUAGE_CODE,
        SPANISH_LANGUAGE_CODE,
        BASE_MODEL_VARIANT,
        1,
        "Moonshine Spanish",
        _megabytes("65"),
        FAST_BATCH_QUALITY,
    ),
    _moonshine(
        ARABIC_LANGUAGE_CODE,
        ARABIC_LANGUAGE_CODE,
        BASE_MODEL_VARIANT,
        1,
        "Moonshine Arabic",
        _megabytes(MOONSHINE_BASE_SIZE_MB),
        FAST_BATCH_QUALITY,
    ),
    _moonshine(
        JAPANESE_LANGUAGE_CODE,
        JAPANESE_LANGUAGE_CODE,
        BASE_MODEL_VARIANT,
        1,
        "Moonshine Japanese Base",
        _megabytes(MOONSHINE_BASE_SIZE_MB),
        FAST_BATCH_QUALITY,
    ),
    _moonshine(
        "ja-tiny",
        JAPANESE_LANGUAGE_CODE,
        "Tiny",
        0,
        "Moonshine Japanese Tiny",
        _megabytes("72"),
        "Fastest · batch",
    ),
    _moonshine(
        KOREAN_LANGUAGE_CODE,
        KOREAN_LANGUAGE_CODE,
        "Tiny",
        0,
        "Moonshine Korean",
        _megabytes("72"),
        "Fastest · batch",
    ),
    _moonshine(
        CHINESE_LANGUAGE_CODE,
        CHINESE_LANGUAGE_CODE,
        BASE_MODEL_VARIANT,
        1,
        "Moonshine Mandarin",
        _megabytes(MOONSHINE_BASE_SIZE_MB),
        FAST_BATCH_QUALITY,
    ),
    _moonshine(
        UKRAINIAN_LANGUAGE_CODE,
        UKRAINIAN_LANGUAGE_CODE,
        BASE_MODEL_VARIANT,
        1,
        "Moonshine Ukrainian",
        _megabytes(MOONSHINE_BASE_SIZE_MB),
        FAST_BATCH_QUALITY,
    ),
    _moonshine(
        VIETNAMESE_LANGUAGE_CODE,
        VIETNAMESE_LANGUAGE_CODE,
        BASE_MODEL_VARIANT,
        1,
        "Moonshine Vietnamese",
        _megabytes(MOONSHINE_BASE_SIZE_MB),
        FAST_BATCH_QUALITY,
    ),
    _faster_whisper(
        "tiny.en",
        "faster-whisper Tiny EN",
        _megabytes(TINY_MODEL_SIZE_MB),
        ENGLISH_ONLY,
        FASTEST_QUALITY,
        2,
    ),
    _faster_whisper(
        "tiny",
        "faster-whisper Tiny",
        _megabytes(TINY_MODEL_SIZE_MB),
        MULTILINGUAL,
        FASTEST_QUALITY,
        2,
    ),
    _faster_whisper(
        "base.en",
        "faster-whisper Base EN",
        _megabytes(BASE_MODEL_SIZE_MB),
        ENGLISH_ONLY,
        FAST_QUALITY,
        3,
    ),
    _faster_whisper(
        "base", "faster-whisper Base", _megabytes(BASE_MODEL_SIZE_MB), MULTILINGUAL, FAST_QUALITY, 3
    ),
    _faster_whisper(
        "small.en", "faster-whisper Small EN", _megabytes("484"), ENGLISH_ONLY, BALANCED_QUALITY, 6
    ),
    _faster_whisper(
        "small", "faster-whisper Small", _megabytes("484"), MULTILINGUAL, BALANCED_QUALITY, 6
    ),
    _faster_whisper(
        "distil-small.en",
        "Distil-Whisper Small EN",
        _megabytes("332"),
        ENGLISH_ONLY,
        "Fast · distilled",
        5,
    ),
    _faster_whisper(
        "distil-medium.en",
        "Distil-Whisper Medium EN",
        _megabytes("789"),
        ENGLISH_ONLY,
        "Accurate · distilled",
        8,
    ),
    _whisperkit(
        "openai_whisper-tiny", "WhisperKit Tiny", _megabytes("66"), MULTILINGUAL, FASTEST_QUALITY, 4
    ),
    _whisperkit(
        "openai_whisper-tiny.en",
        "WhisperKit Tiny EN",
        _megabytes("66"),
        ENGLISH_ONLY,
        FASTEST_QUALITY,
        4,
    ),
    _whisperkit(
        "openai_whisper-base",
        "WhisperKit Base",
        _megabytes(BASE_MODEL_SIZE_MB),
        MULTILINGUAL,
        FAST_QUALITY,
        4,
    ),
    _whisperkit(
        "openai_whisper-base.en",
        "WhisperKit Base EN",
        _megabytes(BASE_MODEL_SIZE_MB),
        ENGLISH_ONLY,
        FAST_QUALITY,
        4,
    ),
    _whisperkit(
        "openai_whisper-small_216MB",
        "WhisperKit Small (compressed)",
        _megabytes("216"),
        MULTILINGUAL,
        BALANCED_QUALITY,
        8,
    ),
    _whisperkit(
        "openai_whisper-small",
        "WhisperKit Small",
        _megabytes("484"),
        MULTILINGUAL,
        BALANCED_QUALITY,
        8,
    ),
    _whisperkit(
        "openai_whisper-large-v3-v20240930_626MB",
        "WhisperKit Large v3 Turbo (compressed)",
        _megabytes("626"),
        MULTILINGUAL,
        MOST_ACCURATE_QUALITY,
        HIGH_MEMORY_RAM_GB,
    ),
    _whisperkit(
        "openai_whisper-large-v3-v20240930_turbo",
        "WhisperKit Large v3 Turbo",
        _megabytes("1610"),
        MULTILINGUAL,
        MOST_ACCURATE_QUALITY,
        VERY_HIGH_MEMORY_RAM_GB,
    ),
    _whisper_cpp(
        "ggml-tiny.en.bin",
        "whisper.cpp Tiny EN",
        _megabytes(TINY_MODEL_SIZE_MB),
        ENGLISH_ONLY,
        FASTEST_QUALITY,
        4,
    ),
    _whisper_cpp(
        "ggml-tiny.bin",
        "whisper.cpp Tiny",
        _megabytes(TINY_MODEL_SIZE_MB),
        MULTILINGUAL,
        FASTEST_QUALITY,
        4,
    ),
    _whisper_cpp(
        "ggml-base.en.bin", "whisper.cpp Base EN", _megabytes("142"), ENGLISH_ONLY, FAST_QUALITY, 4
    ),
    _whisper_cpp(
        "ggml-base.bin", "whisper.cpp Base", _megabytes("142"), MULTILINGUAL, FAST_QUALITY, 4
    ),
    _whisper_cpp(
        "ggml-small.en.bin",
        "whisper.cpp Small EN",
        _megabytes("466"),
        ENGLISH_ONLY,
        BALANCED_QUALITY,
        8,
    ),
    _whisper_cpp(
        "ggml-small.bin", "whisper.cpp Small", _megabytes("466"), MULTILINGUAL, BALANCED_QUALITY, 8
    ),
    _whisper_cpp(
        "ggml-medium.en.bin",
        "whisper.cpp Medium EN",
        _megabytes("1500"),
        ENGLISH_ONLY,
        "Accurate",
        HIGH_MEMORY_RAM_GB,
    ),
    _whisper_cpp(
        "ggml-medium.bin",
        "whisper.cpp Medium",
        _megabytes("1500"),
        MULTILINGUAL,
        "Accurate",
        HIGH_MEMORY_RAM_GB,
    ),
    _whisper_cpp(
        "whisper-medium-q4_1.bin",
        "Whisper Medium Q4",
        _megabytes("492"),
        MULTILINGUAL,
        "Accurate · compact",
        8,
        download_url="https://blob.handy.computer/whisper-medium-q4_1.bin",
        description=(
            "Handy's compact quantized Whisper Medium (Q4). Same whisper.cpp runtime; weights "
            "are hosted on Handy's CDN and work without the Handy app."
        ),
        source="Handy-compatible",
        # MIT Whisper weights; Handy redistributes the quant.
        # License string left generic so the UI links through to Handy rather than inventing one.
    ),
    _whisper_cpp(
        "ggml-large-v3-turbo.bin",
        "whisper.cpp Large v3 Turbo",
        _megabytes("1620"),
        MULTILINGUAL,
        MOST_ACCURATE_QUALITY,
        VERY_HIGH_MEMORY_RAM_GB,
    ),
    _whisper_cpp(
        "ggml-large-v3-q5_0.bin",
        "Whisper Large v3 Q5",
        _megabytes("1081"),
        MULTILINGUAL,
        "Most accurate · compact",
        VERY_HIGH_MEMORY_RAM_GB,
        download_url="https://blob.handy.computer/ggml-large-v3-q5_0.bin",
        description=(
            "Quantized Whisper Large v3 (Q5) from Handy's model catalog. Larger and more accurate "
            "than Medium Q4; still runs through the local whisper.cpp engine."
        ),
        source="Handy-compatible",
    ),
    _whisper_cpp(
        "breeze-asr-q5_k.bin",
        "Breeze ASR Q5",
        _megabytes("1081"),
        "Taiwanese Mandarin + English",
        "Specialized",
        VERY_HIGH_MEMORY_RAM_GB,
        download_url="https://blob.handy.computer/breeze-asr-q5_k.bin",
        family="Breeze ASR",
        description=(
            "MediaTek Breeze-ASR (Whisper Large v2 fine-tune) quantized to Q5 for whisper.cpp. "
            "Tuned for Taiwanese Mandarin and Mandarin–English code-switching; "
            "weights redistributed via Handy's CDN."
        ),
        source="Breeze ASR",
        language_codes=(CHINESE_LANGUAGE_CODE, ENGLISH_LANGUAGE_CODE),
        license_name=APACHE_LICENSE,
    ),
    _whisper_cpp(
        "ggml-large-v3.bin",
        "whisper.cpp Large v3",
        _gigabytes("3"),
        MULTILINGUAL,
        MOST_ACCURATE_QUALITY,
        24,
    ),
)

DEFAULT_CATALOG: tuple[CatalogModel, ...] = apply_pins(_BASE_CATALOG)


def catalog_by_id(catalog: tuple[CatalogModel, ...] = DEFAULT_CATALOG) -> dict[str, CatalogModel]:
    return {model.id: model for model in catalog}


def recommended_ids(system: SystemInfo) -> set[str]:
    """Pick the models that best fit this machine."""
    preferred_engine = ENGINE_WHISPERKIT if system.is_apple_silicon else ENGINE_FASTER_WHISPER
    ram = system.ram_gb or DEFAULT_RECOMMENDATION_RAM_GB
    if ram >= VERY_HIGH_MEMORY_RAM_GB:
        if preferred_engine == ENGINE_WHISPERKIT:
            return {
                f"{ENGINE_WHISPERKIT}:openai_whisper-large-v3-v20240930_626MB",
                f"{ENGINE_MLX_AUDIO}:whisper-large-v3-turbo-4bit",
                f"{ENGINE_MLX_AUDIO}:parakeet-tdt-0.6b-v3",
                f"{ENGINE_MLX_AUDIO}:parakeet-tdt-0.6b-v2",
            }
        return {
            f"{ENGINE_SHERPA_ONNX}:parakeet-tdt-0.6b-v3-int8",
            f"{ENGINE_SHERPA_ONNX}:parakeet-tdt-0.6b-v2-int8",
            f"{ENGINE_FASTER_WHISPER}:small",
            f"{ENGINE_FASTER_WHISPER}:distil-medium.en",
        }
    if ram >= 8:
        if preferred_engine == ENGINE_WHISPERKIT:
            return {
                f"{ENGINE_WHISPERKIT}:openai_whisper-small_216MB",
                f"{ENGINE_MLX_AUDIO}:whisper-large-v3-turbo-4bit",
            }
        return {
            f"{ENGINE_SHERPA_ONNX}:sensevoice-small-int8",
            f"{ENGINE_FASTER_WHISPER}:base",
            f"{ENGINE_FASTER_WHISPER}:distil-small.en",
        }
    if preferred_engine == ENGINE_WHISPERKIT:
        return {
            f"{ENGINE_WHISPERKIT}:openai_whisper-base",
            f"{ENGINE_SHERPA_ONNX}:sensevoice-small-int8",
        }
    return {
        f"{ENGINE_SHERPA_ONNX}:sensevoice-small-int8",
        f"{ENGINE_FASTER_WHISPER}:tiny",
    }

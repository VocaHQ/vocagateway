from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from app.system import SystemInfo

# Integrity pins live beside the catalog rather than inline in the model table
# because they are machine-generated (scripts/harvest-model-pins.py) while the
# table is hand-written. Keeping them apart means regenerating pins produces a
# diff of nothing but digests, which is what makes them reviewable.
PINS_PATH = Path(__file__).parent / "model_pins.json"

ENGINE_WHISPER_CPP = "whisper.cpp"
ENGINE_TRANSCRIBE_CPP = "transcribe.cpp"
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
HINDI_LANGUAGE_CODE = "hi"
HINGLISH_ROMAN_LANGUAGE_CODE = "hinglish_roman"
TAGALOG_LANGUAGE_CODE = "tl"
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
MOONSHINE_REVISION_V015 = "moonshine-voice-0.1.5"
MOONSHINE_015_REVISION = MOONSHINE_REVISION_V015  # noqa: WPS114
MOONSHINE_STREAMING_SMALL_VARIANT = "Small Streaming"
MOONSHINE_STREAMING_TINY_VARIANT = "Tiny Streaming"
MOONSHINE_STREAMING_BALANCED_QUALITY = "Balanced · cached streaming"
MOONSHINE_STREAMING_FASTEST_QUALITY = "Fastest · cached streaming"
MOONSHINE_RETIREMENT_REASON = (
    "Moonshine deprecated this Community batch model after publishing an MIT streaming replacement."
)
MOONSHINE_RETIREMENT_PLURAL_REASON = (
    "Moonshine deprecated this Community batch model after publishing MIT streaming replacements."
)
NEMOTRON_SIZE_BYTES = 682_215_471
BENGALI_ZIPFORMER_SIZE_BYTES = 94_119_939
MOONSHINE_EN_MEDIUM_STREAMING_SIZE_BYTES = 269_141_623
MOONSHINE_EN_SMALL_STREAMING_SIZE_BYTES = 142_300_974
MOONSHINE_EN_TINY_STREAMING_SIZE_BYTES = 45_233_659
MOONSHINE_EN_BASE_SIZE_BYTES = 141_001_190
MOONSHINE_EN_TINY_SIZE_BYTES = 43_943_830
MOONSHINE_AR_TINY_STREAMING_SIZE_BYTES = 32_349_411
MOONSHINE_DE_SMALL_STREAMING_SIZE_BYTES = 121_800_823
MOONSHINE_DE_TINY_STREAMING_SIZE_BYTES = 32_317_004
MOONSHINE_ES_SMALL_STREAMING_SIZE_BYTES = 121_800_392
MOONSHINE_ES_TINY_STREAMING_SIZE_BYTES = 32_316_573
MOONSHINE_JA_SMALL_STREAMING_SIZE_BYTES = 121_803_780
MOONSHINE_JA_TINY_STREAMING_SIZE_BYTES = 32_319_961
MOONSHINE_ZH_TINY_STREAMING_SIZE_BYTES = 32_290_152
MOONSHINE_TL_TINY_STREAMING_SIZE_BYTES = 32_309_481
MOONSHINE_VI_TINY_STREAMING_SIZE_BYTES = 32_309_008
MOONSHINE_KO_SIZE_BYTES = 71_815_486
MOONSHINE_UK_SIZE_BYTES = 141_001_214
DISTIL_LARGE_V3_SIZE_BYTES = 1_515_408_824
FASTER_WHISPER_LARGE_V3_TURBO_SIZE_BYTES = 1_621_669_956
FASTER_WHISPER_LARGE_V3_SIZE_BYTES = 3_090_839_273
FASTER_WHISPER_MEDIUM_SIZE_BYTES = 1_530_575_217
FASTER_WHISPER_MEDIUM_EN_SIZE_BYTES = 1_530_460_562
ACCURATE_QUALITY = "Accurate"
# Turbo is not the most accurate Whisper — the Open ASR Leaderboard puts it at
# 6.36 average WER against full Large v3's 5.78 — so it must not claim to be in
# a picker that also offers Large v3 and the Q5 build of those same weights.
TURBO_QUALITY = "Large-model accuracy · fast decoder"
WHISPERKIT_COMPRESSED_LARGE_ID = "whisperkit:openai_whisper-large-v3-v20240930_626MB"
DISTIL_LARGE_V35_SIZE_BYTES = 1_516_487_390
WHISPER_TURBO_REPLACEMENT_ID = "whisper.cpp:ggml-large-v3-turbo.bin"
WHISPER_TURBO_RETIREMENT_REASON = (
    "Whisper Large v3 Turbo replaces this tier: the same encoder with four decoder layers "
    "instead of 32, so it is smaller and several times faster. It is not more accurate — the "
    "Open ASR Leaderboard puts Turbo about 0.6 WER points behind full Large v3 on English "
    "and up to 2 behind on German — but Medium trails both, so nothing here is given up."
)
HINGLISH_MODEL_SIZE_BYTES = 574_041_195
# Qwen3-ASR takes its language as a string interpolated into the decoder prompt,
# where the literal "None" is how that prompt spells "do not pin a language".
# It is a decoder token, not Python's None, which would mean "no override".
LANGUAGE_AGNOSTIC_DECODER = "None"

MODEL_SAFETENSORS = "model.safetensors"
JOINER_INT8_FILE = "joiner.int8.onnx"
NEMO_TRANSDUCER_TYPE = "nemo_transducer"
STREAMING_TRANSDUCER_TYPE = "streaming_zipformer"
COHERE_TRANSCRIBE_TYPE = "cohere_transcribe"
ENGLISH_CODES = (ENGLISH_LANGUAGE_CODE,)
SWIFT_SIZE_BYTES = 296_189_187
HINDI_SMALL_SIZE_BYTES = 970_935_688
SROTA_SIZE_BYTES = 1_580_827_103
GRANITE_MULTILINGUAL_SIZE_BYTES = 4_636_316_308
PRIME_SIZE_BYTES = 1_177_039_883
PARAKEET_UNIFIED_SIZE_BYTES = 663_043_117
PARAKEET_UNIFIED_STREAMING_SIZE_BYTES = 663_048_978
CANARY_QWEN_SIZE_BYTES = 1_983_729_024
GRANITE_GGUF_SIZE_BYTES = 1_829_704_544
COHERE_SIZE_BYTES = 2_888_052_036
WHISPER_HF_FILES = (
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    MODEL_SAFETENSORS,
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
SROTA_HF_FILES = (*WHISPER_HF_FILES, "chat_template.json")

# Qwen3-ASR's upstream card lists 30 languages plus Chinese dialects. The
# dialects are represented by the model's Mandarin/`yue` capability rather
# than exposed as separate gateway language selectors.
_QWEN3_LANGUAGE_CODES: tuple[str, ...] = (
    ENGLISH_LANGUAGE_CODE,
    CHINESE_LANGUAGE_CODE,
    "yue",
    JAPANESE_LANGUAGE_CODE,
    KOREAN_LANGUAGE_CODE,
    SPANISH_LANGUAGE_CODE,
    FRENCH_LANGUAGE_CODE,
    GERMAN_LANGUAGE_CODE,
    RUSSIAN_LANGUAGE_CODE,
    ARABIC_LANGUAGE_CODE,
    ITALIAN_LANGUAGE_CODE,
    PORTUGUESE_LANGUAGE_CODE,
    "id",
    "th",
    VIETNAMESE_LANGUAGE_CODE,
    "tr",
    HINDI_LANGUAGE_CODE,
    "ms",
    "nl",
    "sv",
    "da",
    "fi",
    "pl",
    "cs",
    "fil",
    "fa",
    "el",
    "hu",
    "mk",
    "ro",
)


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
    revision: str | None = None
    sha256: str | None = None
    file_digests: tuple[tuple[str, str], ...] = ()
    model_type: str | None = None
    language_codes: tuple[str, ...] = ()
    # Some fine-tunes expose an application-level output contract whose wire
    # value is not a token understood by the decoder (for example, Roman
    # Hinglish is requested as `hinglish_roman` but Whisper expects `hi`).
    # `LANGUAGE_AGNOSTIC_DECODER` pins the decoder to no language at all.
    decoder_language_code: str | None = None
    # Instruction some autoregressive models need in order to transcribe rather
    # than answer or translate. Passed only when the adapter's generate() takes it.
    decoder_prompt: str | None = None
    apple_silicon_only: bool = False
    detects_language_automatically: bool = False
    retired: bool = False
    replacement_id: str | None = None
    retirement_reason: str | None = None


PinsMap = dict[str, dict[str, Any]]


class _PinManager:
    @classmethod
    def load_pins(cls, path: Path = PINS_PATH) -> PinsMap:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        models = payload.get("models") if isinstance(payload, dict) else None
        return models if isinstance(models, dict) else {}

    @classmethod
    def pin_download_url(cls, url: str | None, revision: str | None) -> str | None:
        if not url or not revision:
            return url
        marker = "/resolve/main/"
        if "huggingface.co/" not in url or marker not in url:
            return url
        return url.replace(marker, f"/resolve/{revision}/", 1)

    @classmethod
    def apply_pins(
        cls, catalog: tuple[CatalogModel, ...], pins: PinsMap | None = None
    ) -> tuple[CatalogModel, ...]:
        records = cls.load_pins() if pins is None else pins
        if not records:
            return catalog
        pinned_models = [cls._pin_model(model, records.get(model.id)) for model in catalog]
        return tuple(pinned_models)

    @classmethod
    def _parse_file_digests(
        cls, digests: Any, default: tuple[tuple[str, str], ...]
    ) -> tuple[tuple[str, str], ...]:
        if not isinstance(digests, dict) or not digests:
            return default
        pairs = []
        for key, digest in digests.items():
            pairs.append((str(key), str(digest)))
        return tuple(sorted(pairs))

    @classmethod
    def _pin_model(cls, model: CatalogModel, record: Any) -> CatalogModel:
        if not isinstance(record, dict):
            return model
        rev = record.get("revision") or model.revision
        digests = cls._parse_file_digests(record.get("file_digests"), model.file_digests)
        return replace(
            model,
            revision=rev,
            sha256=record.get("sha256") or model.sha256,
            download_url=cls.pin_download_url(model.download_url, rev),
            file_digests=digests,
        )


class _WhisperModelBuilders:
    @classmethod
    def retirement(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        """The retirement triple, so each Whisper builder spells it once."""
        return {
            "retired": bool(kwargs.get("retired", False)),
            "replacement_id": kwargs.get("replacement_id"),
            "retirement_reason": kwargs.get("retirement_reason"),
        }

    @classmethod
    def whisper_language_codes(cls, languages: str) -> tuple[str, ...]:
        return ENGLISH_CODES if languages == ENGLISH_ONLY else WHISPER_LANGUAGES

    @classmethod
    def whisper_cpp(
        cls,
        key: str,
        label: str,
        *args: Any,
        **kwargs: Any,
    ) -> CatalogModel:
        return CatalogModel(
            id=f"{ENGINE_WHISPER_CPP}:{key}",
            engine=ENGINE_WHISPER_CPP,
            key=key,
            label=label,
            size_bytes=int(args[0]),
            languages=cls._languages(args),
            quality=str(args[2]),
            minimum_ram_gb=float(args[3]),
            download_url=kwargs.get("download_url")
            or (f"https://huggingface.co/{WHISPER_CPP_REPO}/resolve/main/{key}"),
            family=str(kwargs.get("family", "Whisper")),
            description=str(
                kwargs.get(
                    "description",
                    "OpenAI Whisper converted for the standalone whisper.cpp engine.",
                )
            ),
            source=str(kwargs.get("source", ENGINE_WHISPER_CPP)),
            language_codes=tuple(
                kwargs.get("language_codes") or cls.whisper_language_codes(cls._languages(args))
            ),
            decoder_language_code=kwargs.get("decoder_language_code"),
            decoder_prompt=kwargs.get("decoder_prompt"),
            license_name=str(kwargs.get("license_name", "See model source")),  # noqa: WPS226
            **cls.retirement(kwargs),
        )

    @classmethod
    def whisperkit(
        cls,
        folder: str,
        label: str,
        *args: Any,
        **kwargs: Any,
    ) -> CatalogModel:
        return CatalogModel(
            id=f"{ENGINE_WHISPERKIT}:{folder}",
            engine=ENGINE_WHISPERKIT,
            key=folder,
            label=label,
            size_bytes=int(args[0]),
            languages=cls._languages(args),
            quality=str(args[2]),
            minimum_ram_gb=float(args[3]),
            huggingface_repo=WHISPERKIT_REPO,
            huggingface_folder=folder,
            family="Whisper",
            description="Core ML Whisper model optimized for Apple silicon.",
            source="WhisperKit",
            language_codes=cls.whisper_language_codes(cls._languages(args)),
            **cls.retirement(kwargs),
        )

    @classmethod
    def faster_whisper(
        cls,
        key: str,
        label: str,
        *args: Any,
        **kwargs: Any,
    ) -> CatalogModel:
        return CatalogModel(
            id=f"{ENGINE_FASTER_WHISPER}:{key}",
            engine=ENGINE_FASTER_WHISPER,
            key=key,
            label=label,
            size_bytes=int(args[0]),
            languages=cls._languages(args),
            quality=str(args[2]),
            minimum_ram_gb=float(args[3]),
            huggingface_repo=str(kwargs.get("repository") or cls._faster_whisper_repo(key)),
            huggingface_folder="",
            family="Whisper / CTranslate2",
            description=(
                "Persistent CTranslate2 model with CPU INT8 inference; "
                "works well on desktop and server CPUs."
            ),
            source="faster-whisper",
            marker_file="model.bin",
            language_codes=cls.whisper_language_codes(cls._languages(args)),
            license_name=str(kwargs.get("license_name", "See model source")),
            commercial_use=bool(kwargs.get("commercial_use", True)),
            **cls.retirement(kwargs),
        )

    @classmethod
    def mlx_audio(
        cls,
        key: str,
        label: str,
        *args: Any,
        **kwargs: Any,
    ) -> CatalogModel:
        return CatalogModel(
            id=f"{ENGINE_MLX_AUDIO}:{key}",
            engine=ENGINE_MLX_AUDIO,
            key=key,
            label=label,
            size_bytes=int(args[0]),
            languages=cls._languages(args),
            quality=str(args[2]),
            minimum_ram_gb=float(args[3]),
            huggingface_repo=str(kwargs["repository"]),
            huggingface_folder="",
            family=str(kwargs["family"]),
            description=str(kwargs["description"]),
            source="MLX Audio",
            marker_file=str(kwargs.get("marker_file", MODEL_SAFETENSORS)),
            required_files=tuple(kwargs.get("required_files", ())),
            decoder_language_code=kwargs.get("decoder_language_code"),
            decoder_prompt=kwargs.get("decoder_prompt"),
            language_codes=tuple(kwargs.get("language_codes", ())),
            apple_silicon_only=True,
            license_name=str(kwargs["license_name"]),
        )

    @classmethod
    def _languages(cls, args: tuple[Any, ...]) -> str:
        return str(args[1])

    @classmethod
    def _faster_whisper_repo(cls, key: str) -> str:
        """Systran publishes the CTranslate2 conversions this engine loads.

        Entries whose conversion lives elsewhere (Whisper Large v3 Turbo has no
        Systran build) pass `repository=` instead of matching this convention.
        """
        if key.startswith("distil-"):
            return f"Systran/faster-distil-whisper-{key.removeprefix('distil-')}"
        return f"Systran/faster-whisper-{key}"


class _SpecializedModelBuilders:
    @classmethod
    def megabytes(cls, size_text: str) -> int:
        return int(size_text) * MB

    @classmethod
    def gigabytes(cls, size_text: str) -> int:
        return int(size_text) * GB

    @classmethod
    def moonshine(
        cls,
        key: str,
        language: str,
        *args: Any,
        **kwargs: Any,
    ) -> CatalogModel:
        return CatalogModel(
            id=f"{ENGINE_MOONSHINE}:{key}",
            engine=ENGINE_MOONSHINE,
            key=key,
            label=str(args[2]),
            size_bytes=int(args[3]),
            languages=f"{_MOONSHINE_LANGUAGE_NAMES[language]} only",
            quality=str(args[4]),
            minimum_ram_gb=float(kwargs.get("minimum_ram_gb", 2)),
            family="Moonshine",
            description=cls._moonshine_description(
                str(args[0]), bool(kwargs.get("supports_streaming", False))
            ),
            source="Moonshine Voice",
            marker_file=".vocagateway-model.json",
            language_code=language,
            language_codes=(language,),
            model_arch=int(args[1]),
            supports_streaming=bool(kwargs.get("supports_streaming", False)),
            license_name=str(
                kwargs.get(
                    "license_name",
                    MIT_LICENSE
                    if language == ENGLISH_LANGUAGE_CODE
                    else "Moonshine Community License",
                )
            ),
            commercial_use=bool(kwargs.get("commercial_use", language == ENGLISH_LANGUAGE_CODE)),
            required_files=tuple(kwargs.get("required_files", ())),
            revision=kwargs.get("revision"),
            retired=bool(kwargs.get("retired", False)),
            replacement_id=kwargs.get("replacement_id"),
            retirement_reason=kwargs.get("retirement_reason"),
        )

    @classmethod
    def sherpa_onnx(
        cls,
        key: str,
        label: str,
        *args: Any,
        **kwargs: Any,
    ) -> CatalogModel:
        cls._validate_sherpa_source(
            key,
            kwargs.get("archive_url"),
            kwargs.get("archive_root"),
            kwargs.get("huggingface_repo"),
        )
        return CatalogModel(
            id=f"{ENGINE_SHERPA_ONNX}:{key}",
            engine=ENGINE_SHERPA_ONNX,
            key=key,
            label=label,
            size_bytes=int(args[0]),
            languages=cls._languages(args),
            quality=str(args[2]),
            minimum_ram_gb=float(args[3]),
            archive_url=kwargs.get("archive_url"),
            archive_root=kwargs.get("archive_root"),
            huggingface_repo=kwargs.get("huggingface_repo"),
            required_files=tuple(kwargs["required_files"]),
            family=str(kwargs["family"]),
            description=str(kwargs["description"]),
            source="sherpa-onnx",
            marker_file=".vocagateway-model.json",
            model_type=str(kwargs["model_type"]),
            language_codes=tuple(kwargs["language_codes"]),
            license_name=str(kwargs["license_name"]),
            supports_streaming=bool(kwargs.get("supports_streaming", False)),
            detects_language_automatically=bool(
                kwargs.get("detects_language_automatically", False)
            ),
        )

    @classmethod
    def _languages(cls, args: tuple[Any, ...]) -> str:
        return str(args[1])

    @classmethod
    def _moonshine_description(cls, arch: str, streaming: bool) -> str:
        inf = (
            " Uses cached incremental inference while you speak."
            if streaming
            else " Uses the fast batch pipeline after recording."
        )
        return f"{arch} model optimized for private local dictation.{inf}"

    @classmethod
    def _validate_sherpa_source(
        cls, key: str, archive_url: Any, archive_root: Any, huggingface_repo: Any
    ) -> None:
        if archive_url is not None:
            if archive_root is None:
                raise ValueError(f"{key}: archive_url requires archive_root.")
        elif huggingface_repo is None:
            raise ValueError(f"{key}: provide either archive_url/archive_root or huggingface_repo.")


_whisper_cpp = _WhisperModelBuilders.whisper_cpp
_whisperkit = _WhisperModelBuilders.whisperkit
_faster_whisper = _WhisperModelBuilders.faster_whisper
_mlx_audio = _WhisperModelBuilders.mlx_audio
_whisper_language_codes = _WhisperModelBuilders.whisper_language_codes
_megabytes = _SpecializedModelBuilders.megabytes
_gigabytes = _SpecializedModelBuilders.gigabytes
_moonshine = _SpecializedModelBuilders.moonshine
_sherpa_onnx = _SpecializedModelBuilders.sherpa_onnx

WHISPER_LANGUAGES: tuple[str, ...] = tuple(
    str.split(
        "af am ar as az ba be bg bn bo br bs ca cs cy da de el en es et eu fa fi fo fr gl gu ha "
        "haw he hi hr ht hu hy id is it ja jw ka kk km kn ko la lb ln lo lt lv mg mi mk ml mn mr "
        "ms mt my ne nl nn no oc pa pl ps pt ro ru sa sd si sk sl sn so sq sr su sv sw ta te tg th "
        "tk tl tr tt uk ur uz vi yi yo yue zh"
    )
)


# Display names for every code any catalog entry declares, so a model card can list
# "Hindi, Bengali, Tamil" instead of "hi, bn, ta". A missing code falls back to the
# code itself rather than hiding the language.
LANGUAGE_NAMES: MappingProxyType[str, str] = MappingProxyType(
    {
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
        HINDI_LANGUAGE_CODE: "Hindi",
        HINGLISH_ROMAN_LANGUAGE_CODE: "Hinglish — Roman",
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
        TAGALOG_LANGUAGE_CODE: "Tagalog",
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
)

_ENGINE_SOURCE_URLS = MappingProxyType(
    {
        ENGINE_WHISPER_CPP: "https://github.com/ggml-org/whisper.cpp",
        ENGINE_WHISPERKIT: "https://github.com/argmaxinc/WhisperKit",
        ENGINE_FASTER_WHISPER: "https://github.com/SYSTRAN/faster-whisper",
        ENGINE_MOONSHINE: "https://github.com/moonshine-ai/moonshine",
        ENGINE_SHERPA_ONNX: "https://github.com/k2-fsa/sherpa-onnx",
        ENGINE_MLX_AUDIO: "https://github.com/Blaizzy/mlx-audio",
    }
)

_SOURCE_LABEL_URLS = MappingProxyType(
    {
        ENGINE_WHISPER_CPP: "https://github.com/ggml-org/whisper.cpp",
        "faster-whisper": "https://github.com/SYSTRAN/faster-whisper",
        "WhisperKit": "https://github.com/argmaxinc/WhisperKit",
        "Moonshine Voice": "https://github.com/moonshine-ai/moonshine",
        "sherpa-onnx": "https://github.com/k2-fsa/sherpa-onnx",
        "MLX Audio": "https://github.com/Blaizzy/mlx-audio",
        "Handy-compatible": "https://handy.computer",
        "Breeze ASR": "https://huggingface.co/MediaTek-Research/Breeze-ASR-25",
    }
)


class _CatalogUrls:
    @classmethod
    def source_url(cls, model: CatalogModel) -> str | None:
        hf_url = cls._huggingface_url(model)
        if hf_url:
            return hf_url
        release = cls._github_release_page(model.archive_url)
        if release:
            return release
        named_source = _SOURCE_LABEL_URLS.get(model.source) or _ENGINE_SOURCE_URLS.get(model.engine)
        if named_source:
            return named_source
        return model.download_url or model.archive_url

    @classmethod
    def language_names(cls, codes: tuple[str, ...]) -> list[str]:
        return [LANGUAGE_NAMES.get(code, code) for code in codes]

    @classmethod
    def _huggingface_url(cls, model: CatalogModel) -> str | None:
        if model.huggingface_repo:
            return f"https://huggingface.co/{model.huggingface_repo}"
        if model.download_url and "huggingface.co/" in model.download_url:
            head, has_resolve, _ = model.download_url.partition("/resolve/")
            return head if has_resolve else model.download_url
        return None

    @classmethod
    def _github_release_page(cls, archive_url: str | None) -> str | None:
        if not archive_url or "/releases/download/" not in archive_url:
            return None
        head, _, rest = archive_url.partition("/releases/download/")
        if not head.startswith("https://github.com/") or not rest:
            return None
        tag = rest.split("/", maxsplit=1)[0]
        return f"{head}/releases/tag/{tag}" if tag else None


load_pins = _PinManager.load_pins
pin_download_url = _PinManager.pin_download_url
apply_pins = _PinManager.apply_pins
catalog_source_url = _CatalogUrls.source_url
language_names = _CatalogUrls.language_names


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
    HINDI_LANGUAGE_CODE,
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
    TAGALOG_LANGUAGE_CODE,
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


_MOONSHINE_LANGUAGE_NAMES = MappingProxyType(
    {
        ARABIC_LANGUAGE_CODE: "Arabic",
        GERMAN_LANGUAGE_CODE: "German",
        ENGLISH_LANGUAGE_CODE: "English",
        SPANISH_LANGUAGE_CODE: "Spanish",
        JAPANESE_LANGUAGE_CODE: "Japanese",
        KOREAN_LANGUAGE_CODE: "Korean",
        TAGALOG_LANGUAGE_CODE: "Tagalog",
        UKRAINIAN_LANGUAGE_CODE: "Ukrainian",
        VIETNAMESE_LANGUAGE_CODE: "Vietnamese",
        CHINESE_LANGUAGE_CODE: "Mandarin Chinese",
    }
)

_MOONSHINE_STREAMING_FILES: tuple[str, ...] = (
    "adapter.ort",
    "cross_kv.ort",
    "decoder_kv.ort",
    "encoder.ort",
    "frontend.model.ort",
    "frontend.weights.ort",
    "streaming_config.json",
    "tokenizer.bin",
)

_MOONSHINE_BATCH_FILES: tuple[str, ...] = (
    "decoder_model_merged.ort",
    "encoder_model.ort",
    "tokenizer.bin",
)


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
            JOINER_INT8_FILE,
            TOKENS_FILE,
        ),
        model_type=NEMO_TRANSDUCER_TYPE,
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
            JOINER_INT8_FILE,
            TOKENS_FILE,
        ),
        model_type=NEMO_TRANSDUCER_TYPE,
        language_codes=ENGLISH_CODES,
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
        model_type=NEMO_TRANSDUCER_TYPE,
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
        language_codes=ENGLISH_CODES,
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
        model_type=STREAMING_TRANSDUCER_TYPE,
        language_codes=ENGLISH_CODES,
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
        "30 languages + Chinese dialects",
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
        language_codes=_QWEN3_LANGUAGE_CODES,
        family="Qwen3-ASR",
        description=(
            "Alibaba's speech-aware Qwen3 converted to INT8 ONNX. An LLM decoder rather than a "
            "CTC or transducer head, so it punctuates well but decodes more slowly. It detects "
            "the language itself and cannot be pinned to one. The converted artifact is advertised "
            "for the 30 languages in the Qwen3-ASR model card; Chinese dialect support is not "
            "selectable as separate gateway language codes."
        ),
        license_name=APACHE_LICENSE,
        detects_language_automatically=True,
    ),
    _sherpa_onnx(
        "nemotron-3.5-asr-streaming-0.6b-320ms-int8",
        "Nemotron 3.5 ASR Streaming 0.6B INT8",
        NEMOTRON_SIZE_BYTES,
        "32 ready or broad-coverage locales",
        "Multilingual streaming · punctuation",
        8,
        huggingface_repo=(
            "csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-320ms-int8-2026-06-11"
        ),
        required_files=(
            ENCODER_INT8_FILE,
            DECODER_INT8_FILE,
            JOINER_INT8_FILE,
            TOKENS_FILE,
        ),
        model_type=STREAMING_TRANSDUCER_TYPE,
        language_codes=(
            ENGLISH_LANGUAGE_CODE,
            SPANISH_LANGUAGE_CODE,
            FRENCH_LANGUAGE_CODE,
            ITALIAN_LANGUAGE_CODE,
            PORTUGUESE_LANGUAGE_CODE,
            DUTCH_LANGUAGE_CODE,
            GERMAN_LANGUAGE_CODE,
            "tr",
            RUSSIAN_LANGUAGE_CODE,
            ARABIC_LANGUAGE_CODE,
            HINDI_LANGUAGE_CODE,
            JAPANESE_LANGUAGE_CODE,
            KOREAN_LANGUAGE_CODE,
            VIETNAMESE_LANGUAGE_CODE,
            UKRAINIAN_LANGUAGE_CODE,
            "pl",
            SWEDISH_LANGUAGE_CODE,
            CZECH_LANGUAGE_CODE,
            "no",
            DANISH_LANGUAGE_CODE,
            BULGARIAN_LANGUAGE_CODE,
            FINNISH_LANGUAGE_CODE,
            CROATIAN_LANGUAGE_CODE,
            SLOVAK_LANGUAGE_CODE,
            CHINESE_LANGUAGE_CODE,
            HUNGARIAN_LANGUAGE_CODE,
            ROMANIAN_LANGUAGE_CODE,
            ESTONIAN_LANGUAGE_CODE,
        ),
        family="Nemotron 3.5 ASR",
        description=(
            "NVIDIA's cache-aware multilingual RNNT converted to INT8 ONNX for sherpa-onnx. "
            "The 320 ms export supports explicit language prompts and Automatic mode, with "
            "punctuation and capitalization. Greek, Lithuanian, Latvian, Maltese, Slovenian, "
            "Hebrew, Thai, and Norwegian Nynorsk are adaptation-ready only and are not advertised."
        ),
        license_name="OpenMDW 1.1",
        supports_streaming=True,
        detects_language_automatically=True,
    ),
    _sherpa_onnx(
        "streaming-zipformer-bn-vosk-2026-02-09",
        "Bengali Streaming Zipformer",
        BENGALI_ZIPFORMER_SIZE_BYTES,
        "Bengali only",
        "Compact Bengali streaming",
        2,
        huggingface_repo="csukuangfj2/sherpa-onnx-streaming-zipformer-bn-vosk-2026-02-09",
        required_files=("encoder.onnx", "decoder.onnx", "joiner.onnx", TOKENS_FILE),
        model_type=STREAMING_TRANSDUCER_TYPE,
        language_codes=("bn",),
        family="Zipformer",
        description=(
            "Alpha Cephei's Bengali streaming Zipformer converted to sherpa-onnx. A small CPU "
            "model for live Bengali dictation; punctuation and robustness are validated by the "
            "gateway benchmark rather than inferred from the upstream card."
        ),
        license_name=APACHE_LICENSE,
        supports_streaming=True,
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
        language_codes=ENGLISH_CODES,
    ),
    _mlx_audio(
        "qwen3-asr-0.6b-4bit",
        "MLX Qwen3-ASR 0.6B 4-bit",
        _megabytes("713"),
        "30 languages + Chinese dialects",
        "Accurate multilingual · punctuation",
        8,
        repository="mlx-community/Qwen3-ASR-0.6B-4bit",
        family="Qwen3-ASR / MLX",
        description=(
            "Quantized Qwen3-ASR running natively on Apple silicon through MLX. The upstream "
            "card covers 30 languages plus Chinese dialects; an LLM decoder punctuates well but "
            "decodes more slowly than Parakeet."
        ),
        license_name=APACHE_LICENSE,
        language_codes=_QWEN3_LANGUAGE_CODES,
    ),
    _mlx_audio(
        "qwen3-asr-1.7b-4bit",
        "MLX Qwen3-ASR 1.7B 4-bit",
        _megabytes("1608"),
        "30 languages + Chinese dialects",
        "Most accurate multilingual · punctuation",
        HIGH_MEMORY_RAM_GB,
        repository="mlx-community/Qwen3-ASR-1.7B-4bit",
        family="Qwen3-ASR / MLX",
        description=(
            "The larger Qwen3-ASR for Macs with memory to spare; the same 30-language coverage "
            "and Chinese dialect support as the 0.6B entry, with better accuracy on accented "
            "and noisy speech."
        ),
        license_name=APACHE_LICENSE,
        language_codes=_QWEN3_LANGUAGE_CODES,
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
        language_codes=ENGLISH_CODES,
    ),
    _mlx_audio(
        "hinglish-swift",
        "MLX Hindi to Roman — Swift (Experimental)",
        SWIFT_SIZE_BYTES,
        "Hindi in Roman letters",
        "Fast · Roman output",
        4,
        repository="Oriserve/Whisper-Hindi2Hinglish-Swift",
        family="Whisper / Roman Hindi",
        description=(
            "Compact Whisper Base fine-tune that writes spoken Hindi in Roman letters. Faster and "
            "smaller than Apex, with higher published error rates. No English translation. "
        ),
        license_name=APACHE_LICENSE,
        language_codes=("hinglish_roman",),
        decoder_language_code=HINDI_LANGUAGE_CODE,
        marker_file=MODEL_SAFETENSORS,
        required_files=WHISPER_HF_FILES,
    ),
    _mlx_audio(
        "whisper-small-hindi",
        "MLX Whisper Small Hindi (Experimental)",
        HINDI_SMALL_SIZE_BYTES,
        "Hindi",
        "Specialized · read Hindi",
        4,
        repository="zindagi-technologies/whisper-small-hindi",
        family="Whisper / Hindi",
        description=(
            "Whisper Small fine-tuned on Kathbath Hindi. Writes Devanagari; conversational speech "
            "and Hindi-English mixing are not validated by the publisher. "
        ),
        license_name=APACHE_LICENSE,
        language_codes=(HINDI_LANGUAGE_CODE,),
        decoder_language_code=HINDI_LANGUAGE_CODE,
        marker_file=MODEL_SAFETENSORS,
        required_files=(
            "config.json",
            "generation_config.json",
            MODEL_SAFETENSORS,
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ),
    ),
    _mlx_audio(
        "srota-hinglish",
        "MLX Srota — Hindi + English (Experimental)",
        SROTA_SIZE_BYTES,
        "Hindi + English, mixed script",
        "Specialized · mixed script",
        8,
        repository="moorlee/qwen3-asr-0.6b-hinglish",
        family="Qwen3-ASR / Srota",
        description=(
            "Qwen3-ASR fine-tune for conversational and tutorial Hindi-English speech. Hindi words "
            "use Devanagari, English words use Latin. This is not the Roman Hindi output mode. "
        ),
        license_name=APACHE_LICENSE,
        language_codes=(HINDI_LANGUAGE_CODE, ENGLISH_LANGUAGE_CODE),
        decoder_language_code=LANGUAGE_AGNOSTIC_DECODER,
        marker_file=MODEL_SAFETENSORS,
        required_files=SROTA_HF_FILES,
    ),
    _mlx_audio(
        "srota-conversational",
        "MLX Srota Conversational (Experimental)",
        SROTA_SIZE_BYTES,
        "Hindi + English, mixed script",
        "Specialized · conversation",
        8,
        repository="moorlee/qwen3-asr-0.6b-hinglish-hiacc-v1",
        family="Qwen3-ASR / Srota",
        description=(
            "Conversation-only Srota fine-tune. Mixed Devanagari and Latin output, not Roman "
            "Hindi. Its published conversational evaluation shares speakers with training. "
        ),
        license_name=APACHE_LICENSE,
        language_codes=(HINDI_LANGUAGE_CODE, ENGLISH_LANGUAGE_CODE),
        decoder_language_code=LANGUAGE_AGNOSTIC_DECODER,
        marker_file=MODEL_SAFETENSORS,
        required_files=SROTA_HF_FILES,
    ),
    _mlx_audio(
        "granite-speech-4.1-2b",
        "MLX Granite Speech 4.1 2B Multilingual",
        GRANITE_MULTILINGUAL_SIZE_BYTES,
        "Six languages",
        "Accurate · multilingual",
        VERY_HIGH_MEMORY_RAM_GB,
        repository="ibm-granite/granite-speech-4.1-2b",
        family="Granite Speech / MLX",
        description=(
            "Autoregressive Granite Speech for English, French, German, Spanish, Portuguese and "
            "Japanese. Larger download than the English NAR quantization; upstream BF16 weights. "
        ),
        license_name=APACHE_LICENSE,
        language_codes=(
            ENGLISH_LANGUAGE_CODE,
            FRENCH_LANGUAGE_CODE,
            GERMAN_LANGUAGE_CODE,
            SPANISH_LANGUAGE_CODE,
            PORTUGUESE_LANGUAGE_CODE,
            JAPANESE_LANGUAGE_CODE,
        ),
        decoder_language_code=None,
        # Without it MLX Granite reads an explicit language hint as a request
        # to translate into that language rather than transcribe.
        decoder_prompt="transcribe the speech with proper punctuation and capitalization.",
        marker_file="model.safetensors.index.json",
        required_files=(
            "added_tokens.json",
            "config.json",
            "merges.txt",
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
            "model.safetensors.index.json",
            "preprocessor_config.json",
            "processor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ),
    ),
    _whisper_cpp(
        "ggml-hindi2hinglish-prime.bin",
        "Hindi to Roman — Prime Q5 (Experimental)",
        PRIME_SIZE_BYTES,
        "Hindi in Roman letters",
        "Specialized · Roman output",
        8,
        download_url="https://huggingface.co/curiophile/whisper-hindi2hinglish-ggml/resolve/main/ggml-hindi2hinglish-prime.bin",
        family="Whisper / Roman Hindi",
        source="Oriserve / community GGML",
        description=(
            "Whisper Large v3 fine-tune for direct Roman Hindi output. Larger and slower than "
            "Apex; published accuracy varies by dataset. Community Q5_1 conversion. "
        ),
        language_codes=(HINGLISH_ROMAN_LANGUAGE_CODE,),
        decoder_language_code=HINDI_LANGUAGE_CODE,
        license_name=APACHE_LICENSE,
    ),
    _sherpa_onnx(
        "parakeet-unified-en-0.6b-int8",
        "Parakeet Unified English INT8",
        PARAKEET_UNIFIED_SIZE_BYTES,
        ENGLISH_ONLY,
        "Accurate · batch",
        4,
        huggingface_repo="csukuangfj2/sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming",
        required_files=(ENCODER_INT8_FILE, DECODER_INT8_FILE, JOINER_INT8_FILE, TOKENS_FILE),
        family="Parakeet Unified",
        model_type=NEMO_TRANSDUCER_TYPE,
        description=(
            "Unified FastConformer RNNT for English. INT8 CPU export. The streaming variant uses "
            "560 ms model context; end-to-end latency depends on the host. "
        ),
        language_codes=ENGLISH_CODES,
        license_name="NVIDIA Open Model License",
        supports_streaming=False,
    ),
    _sherpa_onnx(
        "parakeet-unified-en-0.6b-560ms-int8",
        "Parakeet Unified English INT8 Streaming 560 ms",
        PARAKEET_UNIFIED_STREAMING_SIZE_BYTES,
        ENGLISH_ONLY,
        "Fast · streaming",
        4,
        huggingface_repo="csukuangfj2/sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-streaming-560ms",
        required_files=(ENCODER_INT8_FILE, DECODER_INT8_FILE, JOINER_INT8_FILE, TOKENS_FILE),
        family="Parakeet Unified",
        model_type=STREAMING_TRANSDUCER_TYPE,
        description=(
            "Unified FastConformer RNNT for English. INT8 CPU export. The streaming variant uses "
            "560 ms model context; end-to-end latency depends on the host. "
        ),
        language_codes=ENGLISH_CODES,
        license_name="NVIDIA Open Model License",
        supports_streaming=True,
    ),
    _sherpa_onnx(
        "cohere-transcribe-14-lang-int8",
        "Cohere Transcribe INT8",
        COHERE_SIZE_BYTES,
        "14 languages",
        "Accurate · choose a language",
        8,
        huggingface_repo="csukuangfj2/sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01",
        required_files=(
            ENCODER_INT8_FILE,
            "encoder.int8.onnx.data",
            DECODER_INT8_FILE,
            TOKENS_FILE,
        ),
        family="Cohere Transcribe",
        model_type=COHERE_TRANSCRIBE_TYPE,
        description=(
            "Cohere's multilingual speech recognizer through a community INT8 CPU export. Select "
            "the spoken language explicitly; automatic language detection and Hindi are not "
            "supported. "
        ),
        language_codes=(
            ENGLISH_LANGUAGE_CODE,
            FRENCH_LANGUAGE_CODE,
            GERMAN_LANGUAGE_CODE,
            "it",
            SPANISH_LANGUAGE_CODE,
            PORTUGUESE_LANGUAGE_CODE,
            "el",
            "nl",
            POLISH_LANGUAGE_CODE,
            "zh",
            JAPANESE_LANGUAGE_CODE,
            "ko",
            "vi",
            "ar",
        ),
        license_name=APACHE_LICENSE,
    ),
    CatalogModel(
        id="transcribe.cpp:canary-qwen-2.5b-Q5_K_M.gguf",
        engine=ENGINE_TRANSCRIBE_CPP,
        key="canary-qwen-2.5b-Q5_K_M.gguf",
        label="Canary-Qwen 2.5B Q5",
        size_bytes=CANARY_QWEN_SIZE_BYTES,
        languages="English only",
        quality="Accurate · native Q5",
        minimum_ram_gb=8,
        download_url="https://huggingface.co/handy-computer/canary-qwen-2.5b-gguf/resolve/main/canary-qwen-2.5b-Q5_K_M.gguf",
        family="Canary-Qwen",
        source="Handy / transcribe.cpp",
        language_codes=ENGLISH_CODES,
        license_name=CC_BY_LICENSE,
        description=(
            "Community Q5 GGUF conversion. Requires a separately installed transcribe-cli "
            "with a CPU, Metal, CUDA or Vulkan backend. Each recording reloads the model; "
            "this entry does not provide live streaming."
        ),
    ),
    CatalogModel(
        id="transcribe.cpp:granite-speech-4.1-2b-Q5_K_M.gguf",
        engine=ENGINE_TRANSCRIBE_CPP,
        key="granite-speech-4.1-2b-Q5_K_M.gguf",
        label="Granite Speech 4.1 Multilingual Q5",
        size_bytes=GRANITE_GGUF_SIZE_BYTES,
        languages="Six languages",
        quality="Accurate · native Q5",
        minimum_ram_gb=8,
        download_url="https://huggingface.co/handy-computer/granite-speech-4.1-2b-gguf/resolve/main/granite-speech-4.1-2b-Q5_K_M.gguf",
        family="Granite Speech",
        source="Handy / transcribe.cpp",
        language_codes=(
            ENGLISH_LANGUAGE_CODE,
            FRENCH_LANGUAGE_CODE,
            GERMAN_LANGUAGE_CODE,
            SPANISH_LANGUAGE_CODE,
            PORTUGUESE_LANGUAGE_CODE,
            JAPANESE_LANGUAGE_CODE,
        ),
        license_name=APACHE_LICENSE,
        description=(
            "Community Q5 GGUF conversion. Requires a separately installed transcribe-cli "
            "with a CPU, Metal, CUDA or Vulkan backend. Each recording reloads the model; "
            "this entry does not provide live streaming."
        ),
    ),
    # Keep moonshine:en as the default English ID so existing installations and
    # runtime configuration continue to resolve after adding explicit variants.
    _moonshine(
        ENGLISH_LANGUAGE_CODE,
        ENGLISH_LANGUAGE_CODE,
        "Medium Streaming",
        5,
        "Moonshine English Medium Streaming",
        MOONSHINE_EN_MEDIUM_STREAMING_SIZE_BYTES,
        MOONSHINE_STREAMING_FASTEST_QUALITY,
        supports_streaming=True,
        minimum_ram_gb=4,
        required_files=_MOONSHINE_STREAMING_FILES,
        revision=MOONSHINE_REVISION_V015,
    ),
    _moonshine(
        "en-small-streaming",
        ENGLISH_LANGUAGE_CODE,
        MOONSHINE_STREAMING_SMALL_VARIANT,
        4,
        "Moonshine English Small Streaming",
        MOONSHINE_EN_SMALL_STREAMING_SIZE_BYTES,
        MOONSHINE_STREAMING_BALANCED_QUALITY,
        supports_streaming=True,
        required_files=_MOONSHINE_STREAMING_FILES,
        revision=MOONSHINE_015_REVISION,
    ),
    _moonshine(
        "en-tiny-streaming",
        ENGLISH_LANGUAGE_CODE,
        MOONSHINE_STREAMING_TINY_VARIANT,
        2,
        "Moonshine English Tiny Streaming",
        MOONSHINE_EN_TINY_STREAMING_SIZE_BYTES,
        MOONSHINE_STREAMING_FASTEST_QUALITY,
        supports_streaming=True,
        required_files=_MOONSHINE_STREAMING_FILES,
        revision=MOONSHINE_015_REVISION,
    ),
    _moonshine(
        "en-base",
        ENGLISH_LANGUAGE_CODE,
        BASE_MODEL_VARIANT,
        1,
        "Moonshine English Base",
        MOONSHINE_EN_BASE_SIZE_BYTES,
        "Accurate · batch",
        required_files=_MOONSHINE_BATCH_FILES,
        revision=MOONSHINE_015_REVISION,
    ),
    _moonshine(
        "en-tiny",
        ENGLISH_LANGUAGE_CODE,
        "Tiny",
        0,
        "Moonshine English Tiny",
        MOONSHINE_EN_TINY_SIZE_BYTES,
        "Smallest · batch",
        required_files=_MOONSHINE_BATCH_FILES,
        revision=MOONSHINE_015_REVISION,
    ),
    _moonshine(
        "ar-tiny-streaming",
        ARABIC_LANGUAGE_CODE,
        MOONSHINE_STREAMING_TINY_VARIANT,
        2,
        "Moonshine Arabic Tiny Streaming",
        MOONSHINE_AR_TINY_STREAMING_SIZE_BYTES,
        MOONSHINE_STREAMING_FASTEST_QUALITY,
        supports_streaming=True,
        required_files=_MOONSHINE_STREAMING_FILES,
        license_name=MIT_LICENSE,
        commercial_use=True,
        revision=MOONSHINE_015_REVISION,
    ),
    _moonshine(
        "de-small-streaming",
        GERMAN_LANGUAGE_CODE,
        MOONSHINE_STREAMING_SMALL_VARIANT,
        4,
        "Moonshine German Small Streaming",
        MOONSHINE_DE_SMALL_STREAMING_SIZE_BYTES,
        MOONSHINE_STREAMING_BALANCED_QUALITY,
        supports_streaming=True,
        required_files=_MOONSHINE_STREAMING_FILES,
        license_name=MIT_LICENSE,
        commercial_use=True,
        revision=MOONSHINE_015_REVISION,
    ),
    _moonshine(
        "de-tiny-streaming",
        GERMAN_LANGUAGE_CODE,
        MOONSHINE_STREAMING_TINY_VARIANT,
        2,
        "Moonshine German Tiny Streaming",
        MOONSHINE_DE_TINY_STREAMING_SIZE_BYTES,
        MOONSHINE_STREAMING_FASTEST_QUALITY,
        supports_streaming=True,
        required_files=_MOONSHINE_STREAMING_FILES,
        license_name=MIT_LICENSE,
        commercial_use=True,
        revision=MOONSHINE_015_REVISION,
    ),
    _moonshine(
        "es-small-streaming",
        SPANISH_LANGUAGE_CODE,
        MOONSHINE_STREAMING_SMALL_VARIANT,
        4,
        "Moonshine Spanish Small Streaming",
        MOONSHINE_ES_SMALL_STREAMING_SIZE_BYTES,
        MOONSHINE_STREAMING_BALANCED_QUALITY,
        supports_streaming=True,
        required_files=_MOONSHINE_STREAMING_FILES,
        license_name=MIT_LICENSE,
        commercial_use=True,
        revision=MOONSHINE_015_REVISION,
    ),
    _moonshine(
        "es-tiny-streaming",
        SPANISH_LANGUAGE_CODE,
        MOONSHINE_STREAMING_TINY_VARIANT,
        2,
        "Moonshine Spanish Tiny Streaming",
        MOONSHINE_ES_TINY_STREAMING_SIZE_BYTES,
        MOONSHINE_STREAMING_FASTEST_QUALITY,
        supports_streaming=True,
        required_files=_MOONSHINE_STREAMING_FILES,
        license_name=MIT_LICENSE,
        commercial_use=True,
        revision=MOONSHINE_015_REVISION,
    ),
    _moonshine(
        "ja-small-streaming",
        JAPANESE_LANGUAGE_CODE,
        MOONSHINE_STREAMING_SMALL_VARIANT,
        4,
        "Moonshine Japanese Small Streaming",
        MOONSHINE_JA_SMALL_STREAMING_SIZE_BYTES,
        MOONSHINE_STREAMING_BALANCED_QUALITY,
        supports_streaming=True,
        required_files=_MOONSHINE_STREAMING_FILES,
        license_name=MIT_LICENSE,
        commercial_use=True,
        revision=MOONSHINE_015_REVISION,
    ),
    _moonshine(
        "ja-tiny-streaming",
        JAPANESE_LANGUAGE_CODE,
        MOONSHINE_STREAMING_TINY_VARIANT,
        2,
        "Moonshine Japanese Tiny Streaming",
        MOONSHINE_JA_TINY_STREAMING_SIZE_BYTES,
        MOONSHINE_STREAMING_FASTEST_QUALITY,
        supports_streaming=True,
        required_files=_MOONSHINE_STREAMING_FILES,
        license_name=MIT_LICENSE,
        commercial_use=True,
        revision=MOONSHINE_015_REVISION,
    ),
    _moonshine(
        "zh-tiny-streaming",
        CHINESE_LANGUAGE_CODE,
        MOONSHINE_STREAMING_TINY_VARIANT,
        2,
        "Moonshine Mandarin Tiny Streaming",
        MOONSHINE_ZH_TINY_STREAMING_SIZE_BYTES,
        MOONSHINE_STREAMING_FASTEST_QUALITY,
        supports_streaming=True,
        required_files=_MOONSHINE_STREAMING_FILES,
        license_name=MIT_LICENSE,
        commercial_use=True,
        revision=MOONSHINE_015_REVISION,
    ),
    _moonshine(
        "tl-tiny-streaming",
        TAGALOG_LANGUAGE_CODE,
        MOONSHINE_STREAMING_TINY_VARIANT,
        2,
        "Moonshine Tagalog Tiny Streaming",
        MOONSHINE_TL_TINY_STREAMING_SIZE_BYTES,
        MOONSHINE_STREAMING_FASTEST_QUALITY,
        supports_streaming=True,
        required_files=_MOONSHINE_STREAMING_FILES,
        license_name=MIT_LICENSE,
        commercial_use=True,
        revision=MOONSHINE_015_REVISION,
    ),
    _moonshine(
        "vi-tiny-streaming",
        VIETNAMESE_LANGUAGE_CODE,
        MOONSHINE_STREAMING_TINY_VARIANT,
        2,
        "Moonshine Vietnamese Tiny Streaming",
        MOONSHINE_VI_TINY_STREAMING_SIZE_BYTES,
        MOONSHINE_STREAMING_FASTEST_QUALITY,
        supports_streaming=True,
        required_files=_MOONSHINE_STREAMING_FILES,
        license_name=MIT_LICENSE,
        commercial_use=True,
        revision=MOONSHINE_015_REVISION,
    ),
    _moonshine(
        SPANISH_LANGUAGE_CODE,
        SPANISH_LANGUAGE_CODE,
        BASE_MODEL_VARIANT,
        1,
        "Moonshine Spanish",
        _megabytes("65"),
        FAST_BATCH_QUALITY,
        retired=True,
        replacement_id="moonshine:es-small-streaming",
        retirement_reason=MOONSHINE_RETIREMENT_PLURAL_REASON,
    ),
    _moonshine(
        ARABIC_LANGUAGE_CODE,
        ARABIC_LANGUAGE_CODE,
        BASE_MODEL_VARIANT,
        1,
        "Moonshine Arabic",
        _megabytes(MOONSHINE_BASE_SIZE_MB),
        FAST_BATCH_QUALITY,
        retired=True,
        replacement_id="moonshine:ar-tiny-streaming",
        retirement_reason=MOONSHINE_RETIREMENT_REASON,
    ),
    _moonshine(
        JAPANESE_LANGUAGE_CODE,
        JAPANESE_LANGUAGE_CODE,
        BASE_MODEL_VARIANT,
        1,
        "Moonshine Japanese Base",
        _megabytes(MOONSHINE_BASE_SIZE_MB),
        FAST_BATCH_QUALITY,
        retired=True,
        replacement_id="moonshine:ja-small-streaming",
        retirement_reason=MOONSHINE_RETIREMENT_PLURAL_REASON,
    ),
    _moonshine(
        "ja-tiny",
        JAPANESE_LANGUAGE_CODE,
        "Tiny",
        0,
        "Moonshine Japanese Tiny",
        _megabytes("72"),
        "Fastest · batch",
        retired=True,
        replacement_id="moonshine:ja-tiny-streaming",
        retirement_reason=MOONSHINE_RETIREMENT_REASON,
    ),
    _moonshine(
        KOREAN_LANGUAGE_CODE,
        KOREAN_LANGUAGE_CODE,
        "Tiny",
        0,
        "Moonshine Korean",
        MOONSHINE_KO_SIZE_BYTES,
        "Fastest · batch",
        required_files=_MOONSHINE_BATCH_FILES,
        revision=MOONSHINE_015_REVISION,
    ),
    _moonshine(
        CHINESE_LANGUAGE_CODE,
        CHINESE_LANGUAGE_CODE,
        BASE_MODEL_VARIANT,
        1,
        "Moonshine Mandarin",
        _megabytes(MOONSHINE_BASE_SIZE_MB),
        FAST_BATCH_QUALITY,
        retired=True,
        replacement_id="moonshine:zh-tiny-streaming",
        retirement_reason=MOONSHINE_RETIREMENT_REASON,
    ),
    _moonshine(
        UKRAINIAN_LANGUAGE_CODE,
        UKRAINIAN_LANGUAGE_CODE,
        BASE_MODEL_VARIANT,
        1,
        "Moonshine Ukrainian",
        MOONSHINE_UK_SIZE_BYTES,
        FAST_BATCH_QUALITY,
        required_files=_MOONSHINE_BATCH_FILES,
        revision=MOONSHINE_015_REVISION,
    ),
    _moonshine(
        VIETNAMESE_LANGUAGE_CODE,
        VIETNAMESE_LANGUAGE_CODE,
        BASE_MODEL_VARIANT,
        1,
        "Moonshine Vietnamese",
        _megabytes(MOONSHINE_BASE_SIZE_MB),
        FAST_BATCH_QUALITY,
        retired=True,
        replacement_id="moonshine:vi-tiny-streaming",
        retirement_reason=MOONSHINE_RETIREMENT_REASON,
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
        "large-v3-turbo",
        "faster-whisper Large v3 Turbo",
        FASTER_WHISPER_LARGE_V3_TURBO_SIZE_BYTES,
        MULTILINGUAL,
        TURBO_QUALITY,
        8,
        repository="deepdml/faster-whisper-large-v3-turbo-ct2",
        license_name=MIT_LICENSE,
    ),
    _faster_whisper(
        "medium.en",
        "faster-whisper Medium EN",
        FASTER_WHISPER_MEDIUM_EN_SIZE_BYTES,
        ENGLISH_ONLY,
        ACCURATE_QUALITY,
        8,
        license_name=MIT_LICENSE,
    ),
    _faster_whisper(
        "medium",
        "faster-whisper Medium",
        FASTER_WHISPER_MEDIUM_SIZE_BYTES,
        MULTILINGUAL,
        ACCURATE_QUALITY,
        8,
        license_name=MIT_LICENSE,
    ),
    _faster_whisper(
        "large-v3",
        "faster-whisper Large v3",
        FASTER_WHISPER_LARGE_V3_SIZE_BYTES,
        MULTILINGUAL,
        MOST_ACCURATE_QUALITY,
        VERY_HIGH_MEMORY_RAM_GB,
        license_name=MIT_LICENSE,
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
    _faster_whisper(
        "distil-large-v3.5",
        "Distil-Whisper Large v3.5",
        DISTIL_LARGE_V35_SIZE_BYTES,
        ENGLISH_ONLY,
        "Most accurate English · distilled",
        8,
        repository="distil-whisper/distil-large-v3.5-ct2",
        license_name=MIT_LICENSE,
    ),
    _faster_whisper(
        "distil-large-v3",
        "Distil-Whisper Large v3",
        DISTIL_LARGE_V3_SIZE_BYTES,
        ENGLISH_ONLY,
        "Most accurate · distilled",
        VERY_HIGH_MEMORY_RAM_GB,
        license_name=MIT_LICENSE,
        retired=True,
        replacement_id="faster-whisper:distil-large-v3.5",
        retirement_reason=(
            "v3.5 is the same architecture and within 1 MB of the same download, and it is the "
            "one the Open ASR Leaderboard measures: 5.40 average WER, ahead of full Whisper "
            "Large v3 at 5.78 and Turbo at 6.36."
        ),
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
        retired=True,
        replacement_id=WHISPERKIT_COMPRESSED_LARGE_ID,
        retirement_reason=(
            "On the same LibriSpeech run this scores 3.95% WER against 2.49% for the compressed "
            "Large v3, which is 142 MB larger and needs no more memory. The compressed Small "
            "build stays as the genuinely small option."
        ),
    ),
    _whisperkit(
        "openai_whisper-large-v3-v20240930_626MB",
        "WhisperKit Large v3 Turbo (compressed)",
        _megabytes("626"),
        MULTILINGUAL,
        MOST_ACCURATE_QUALITY,
        # Now the only full-quality WhisperKit tier, so it has to be offered to
        # the 8 GB Macs that used to pick Small. A 626 MB Core ML model is well
        # within that budget; the old 12 GB floor was inherited from the 1.6 GB
        # build this entry replaces.
        8,
    ),
    _whisperkit(
        "openai_whisper-large-v3-v20240930_turbo",
        "WhisperKit Large v3 Turbo",
        _megabytes("1610"),
        MULTILINGUAL,
        MOST_ACCURATE_QUALITY,
        VERY_HIGH_MEMORY_RAM_GB,
        retired=True,
        replacement_id=WHISPERKIT_COMPRESSED_LARGE_ID,
        retirement_reason=(
            "Argmax's own evaluation runs both builds over the 2,620 LibriSpeech utterances: "
            "2.40% WER here against 2.49% for the compressed build. Eight hundredths of a WER "
            "point is not worth 984 MB of download and 16 GB of required RAM."
        ),
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
        ACCURATE_QUALITY,
        HIGH_MEMORY_RAM_GB,
        retired=True,
        replacement_id=WHISPER_TURBO_REPLACEMENT_ID,
        retirement_reason=WHISPER_TURBO_RETIREMENT_REASON,
    ),
    _whisper_cpp(
        "ggml-medium.bin",
        "whisper.cpp Medium",
        _megabytes("1500"),
        MULTILINGUAL,
        ACCURATE_QUALITY,
        HIGH_MEMORY_RAM_GB,
        retired=True,
        replacement_id=WHISPER_TURBO_REPLACEMENT_ID,
        retirement_reason=WHISPER_TURBO_RETIREMENT_REASON,
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
        TURBO_QUALITY,
        # Now the top whisper.cpp tier, so it has to be offered to the machines
        # the retired Medium entries used to serve. A 1.6 GB f16 model needs
        # about 2.5 GB resident, which a 12 GB host has to spare.
        HIGH_MEMORY_RAM_GB,
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
        "ggml-apex-hinglish-q5_0.bin",
        "Hinglish — Roman (Experimental)",
        HINGLISH_MODEL_SIZE_BYTES,
        "Hindi + English, Roman script",
        "Experimental · Roman output",
        4,
        download_url=(
            "https://huggingface.co/Marquestra/Whisper-Hindi2Hinglish-Apex-GGML/"
            "resolve/main/ggml-apex-hinglish-q5_0.bin"
        ),
        family="Whisper / Hinglish",
        description=(
            "Experimental Whisper Large v3 Turbo fine-tune for mixed Hindi and English. "
            "It returns the words as spoken in one Latin script rather than Devanagari "
            "or an English translation."
        ),
        source="Whisper-Hindi2Hinglish-Apex",
        language_codes=(HINGLISH_ROMAN_LANGUAGE_CODE,),
        decoder_language_code=HINDI_LANGUAGE_CODE,
        license_name=APACHE_LICENSE,
    ),
    _whisper_cpp(
        "ggml-large-v3.bin",
        "whisper.cpp Large v3",
        _gigabytes("3"),
        MULTILINGUAL,
        MOST_ACCURATE_QUALITY,
        24,
        retired=True,
        replacement_id="faster-whisper:large-v3",
        retirement_reason=(
            "These are the most accurate Whisper weights there are — 5.78 average WER on the "
            "Open ASR Leaderboard against Turbo's 6.36 — but not at 3 GB and 24 GB of RAM "
            "through a CLI that reloads them on every request. The faster-whisper entry runs "
            "the same weights as a resident INT8 model instead."
        ),
    ),
)

_PINNED_CATALOG: tuple[CatalogModel, ...] = apply_pins(_BASE_CATALOG)
DEFAULT_CATALOG: tuple[CatalogModel, ...] = tuple(
    model for model in _PINNED_CATALOG if not model.retired
)
RETIRED_CATALOG: tuple[CatalogModel, ...] = tuple(
    model for model in _PINNED_CATALOG if model.retired
)


def catalog_by_id(catalog: tuple[CatalogModel, ...] = DEFAULT_CATALOG) -> dict[str, CatalogModel]:
    return {model.id: model for model in catalog}


class _CatalogRecommender:
    @classmethod
    def recommended_ids(cls, system: SystemInfo) -> set[str]:
        """Pick the models that best fit this machine."""
        ram = system.ram_gb or DEFAULT_RECOMMENDATION_RAM_GB
        if ram >= VERY_HIGH_MEMORY_RAM_GB:
            return cls._high_ram_ids(system.is_apple_silicon)
        return cls._standard_ram_ids(system.is_apple_silicon, ram)

    @classmethod
    def _high_ram_ids(cls, is_apple: bool) -> set[str]:
        if is_apple:
            return {
                f"{ENGINE_WHISPERKIT}:openai_whisper-large-v3-v20240930_626MB",
                f"{ENGINE_MLX_AUDIO}:whisper-large-v3-turbo-4bit",
                f"{ENGINE_MLX_AUDIO}:parakeet-tdt-0.6b-v3",
                f"{ENGINE_MLX_AUDIO}:parakeet-tdt-0.6b-v2",
            }
        return {
            f"{ENGINE_SHERPA_ONNX}:parakeet-tdt-0.6b-v3-int8",
            f"{ENGINE_SHERPA_ONNX}:parakeet-tdt-0.6b-v2-int8",
            # Turbo is the one Whisper tier a CPU-only host can run at a usable
            # speed: same encoder as Large v3, four decoder layers instead of 32.
            f"{ENGINE_FASTER_WHISPER}:large-v3-turbo",
            f"{ENGINE_FASTER_WHISPER}:distil-medium.en",
        }

    @classmethod
    def _standard_ram_ids(cls, is_apple: bool, ram: float) -> set[str]:
        if ram >= 8:
            if is_apple:
                return {
                    f"{ENGINE_WHISPERKIT}:openai_whisper-small_216MB",
                    f"{ENGINE_MLX_AUDIO}:whisper-large-v3-turbo-4bit",
                }
            return {
                f"{ENGINE_SHERPA_ONNX}:sensevoice-small-int8",
                f"{ENGINE_FASTER_WHISPER}:base",
                f"{ENGINE_FASTER_WHISPER}:distil-small.en",
            }
        if is_apple:
            return {
                f"{ENGINE_WHISPERKIT}:openai_whisper-base",
                f"{ENGINE_SHERPA_ONNX}:sensevoice-small-int8",
            }
        return {
            f"{ENGINE_SHERPA_ONNX}:sensevoice-small-int8",
            f"{ENGINE_FASTER_WHISPER}:tiny",
        }


recommended_ids = _CatalogRecommender.recommended_ids

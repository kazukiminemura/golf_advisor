"""Lightweight modular chatbot with clear OpenVINO and llama.cpp backends.

Highlights:
- Supports both OpenVINO (via openvino-genai) and llama.cpp (via llama-cpp-python).
- Uses the same GGUF model repo/id across backends (e.g.,
  "bartowski/Qwen2.5-1.5B-Instruct-GGUF").
- Auto-downloads GGUF and converts tokenizer for OpenVINO when needed.
- Defaults to the OpenVINO backend (alias: "openvion").
- Maintains backward compatibility via SimpleChatBot.ask/ask_stream.
"""


import argparse
import os
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Tuple, Callable
from pathlib import Path
import sys
import subprocess
import shutil
from pathlib import Path as _PathForDlls
import math


def _prepare_windows_openvino_dlls() -> None:
    """On Windows, ensure dependent DLL directories are on the search path.

    Addresses errors like Windows loader error 126 when loading
    openvino_tokenizers.dll by adding relevant site-packages folders to the
    DLL search path for the current process.
    """
    if os.name != "nt":
        return
    try:
        # Python 3.8+ requires explicit addition of DLL directories
        add_dir = getattr(os, "add_dll_directory", None)
        paths_added = []
        if add_dir is None:  # Older Python on Windows
            return
        # Try to locate openvino and openvino-tokenizers package lib folders
        try:
            import openvino  # type: ignore
            ov_libs = _PathForDlls(openvino.__file__).parent / "libs"
            if ov_libs.exists():
                add_dir(str(ov_libs))
                os.environ["PATH"] = str(ov_libs) + os.pathsep + os.environ.get("PATH", "")
                paths_added.append(str(ov_libs))
        except Exception:
            pass
        try:
            import openvino_tokenizers  # type: ignore
            ovt_lib = _PathForDlls(openvino_tokenizers.__file__).parent / "lib"
            if ovt_lib.exists():
                add_dir(str(ovt_lib))
                os.environ["PATH"] = str(ovt_lib) + os.pathsep + os.environ.get("PATH", "")
                paths_added.append(str(ovt_lib))
        except Exception:
            pass
        # Optionally add openvino_genai binary dir if present
        try:
            import openvino_genai  # type: ignore
            genai_dir = _PathForDlls(openvino_genai.__file__).parent
            # Some wheel layouts may place binaries alongside the package
            for candidate in {genai_dir, genai_dir / "bin", genai_dir / "libs"}:
                if candidate.exists():
                    add_dir(str(candidate))
                    os.environ["PATH"] = str(candidate) + os.pathsep + os.environ.get("PATH", "")
                    paths_added.append(str(candidate))
                    break
        except Exception:
            pass
        if os.environ.get("OV_DLL_DEBUG", "").lower() in {"1", "true", "yes"} and paths_added:
            print(f"[OV][DLL] Added search paths: {paths_added}")
    except Exception:
        # Best-effort only; fall through to normal import behavior
        pass


class ModelInterface(ABC):
    """Abstract interface for different model backends.

    - SRP: Concerned only with text generation.
    - ISP: Optional chat hooks are defined as separate protocols below.
    """

    @abstractmethod
    def generate_response(self, message: str) -> str:
        pass


# ISP: Optional hook protocols (kept separate from core interface)
try:  # pragma: no cover - safe fallback for environments without Protocol
    from typing import Protocol, runtime_checkable  # type: ignore
except Exception:  # pragma: no cover
    Protocol = object  # type: ignore
    def runtime_checkable(x):  # type: ignore
        return x


@runtime_checkable
class SupportsSystemPrompt(Protocol):
    def set_system_prompt(self, prompt: str) -> None: ...


@runtime_checkable
class SupportsResetHistory(Protocol):
    def reset_history(self) -> None: ...


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip())
    except Exception:
        return default


class ChatHistory:
    """Minimal, backend-agnostic chat history and prompt builder.

    Produces a simple bracketed prompt template compatible with plain text
    generation backends. Example:

        [SYSTEM]\n...\n[USER]\nHi\n[ASSISTANT]\nHello!\n[USER]\n...
    """

    def __init__(self) -> None:
        self.system_prompt: Optional[str] = None
        self.turns: List[Tuple[str, str]] = []  # list of (role, content)

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt

    def reset(self) -> None:
        self.turns.clear()

    def add_user(self, content: str) -> None:
        self.turns.append(("USER", content))

    def add_assistant(self, content: str) -> None:
        self.turns.append(("ASSISTANT", content))

    def build_prompt(self, next_user: Optional[str] = None) -> str:
        parts: List[str] = []
        if self.system_prompt:
            parts.append(f"[SYSTEM]\n{self.system_prompt}\n")
        for role, content in self.turns:
            parts.append(f"[{role}]\n{content}\n")
        if next_user is not None:
            parts.append(f"[USER]\n{next_user}\n[ASSISTANT]\n")
        return "".join(parts)


def _maybe_auto_install(spec: str, reason: str, env_flag: str = "AUTO_INSTALL_TOKENIZERS") -> bool:
    """Optionally install a pip package at runtime when explicitly allowed.

    Controlled by environment variable `AUTO_INSTALL_TOKENIZERS` or an
    equivalent CLI flag that sets it to '1'. Returns True if installation
    appears successful.
    """
    if os.environ.get(env_flag, "").strip().lower() not in {"1", "true", "yes"}:
        return False
    try:
        # Support multiple specs separated by spaces
        specs = spec.split()
        print(f"[auto-install] Installing {spec} to {sys.executable} ({reason}) ...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", *specs],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(result.stdout)
        return result.returncode == 0
    except Exception as e:
        print(f"[auto-install] Failed to install {spec}: {e}")
        return False


def _convert_tokenizer_with_options(tokenizer_id: str, out_dir: Path) -> bool:
    """Convert tokenizer to OpenVINO IR with detokenizer when possible.

    Tries `with_detokenizer=True` first, then falls back if not supported.
    Returns True on apparent success.
    """
    try:
        from openvino_tokenizers import convert_tokenizer  # type: ignore
    except Exception as e:
        print("[tokenizer] openvino-tokenizers not available:", e)
        return False
    try:
        convert_tokenizer(tokenizer_id, str(out_dir), with_detokenizer=True)  # type: ignore[arg-type]
        return True
    except TypeError:
        # Older versions may not accept with_detokenizer
        try:
            convert_tokenizer(tokenizer_id, str(out_dir))
            return True
        except Exception as e2:
            print(f"[tokenizer] Conversion failed without detokenizer: {e2}")
            return False
    except Exception as e:
        print(f"[tokenizer] Conversion failed: {e}")
        return False

def _convert_tokenizer_via_cli(tokenizer_id: str, out_dir: Path, with_detokenizer: bool = True) -> bool:
    """Fallback to the `convert_tokenizer` CLI if Python API isn't available."""
    cmd = ["convert_tokenizer", tokenizer_id, "-o", str(out_dir)]
    if with_detokenizer:
        cmd.insert(2, "--with-detokenizer")
    try:
        print(f"[tokenizer] Running CLI: {' '.join(cmd)}")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(res.stdout)
        return res.returncode == 0
    except FileNotFoundError:
        print("[tokenizer] 'convert_tokenizer' CLI not found in PATH")
        return False
    except Exception as e:
        print(f"[tokenizer] CLI conversion failed: {e}")
        return False

def ensure_tokenizer_converted(tokenizer_id: str, out_dir: Path, with_detokenizer: bool = True) -> bool:
    """Ensure tokenizer is converted to OpenVINO IR at `out_dir`.

    Tries Python API; falls back to CLI; optionally installs dependencies
    when `AUTO_INSTALL_TOKENIZERS=1`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_xml = out_dir / "openvino_tokenizer.xml"
    if tok_xml.exists():
        print(f"[tokenizer] Already present: {tok_xml}")
        return True

    # Try Python API first
    ok = _convert_tokenizer_with_options(tokenizer_id, out_dir)
    if ok:
        return True

    # Try CLI fallback
    if _convert_tokenizer_via_cli(tokenizer_id, out_dir, with_detokenizer=with_detokenizer):
        return True

    # Try installing and retry
    if _maybe_auto_install("openvino-tokenizers[transformers]", "tokenizer conversion support"):
        if _convert_tokenizer_with_options(tokenizer_id, out_dir):
            return True
    if _maybe_auto_install("transformers[sentencepiece] tiktoken", "HF tokenizer conversion dependencies"):
        if _convert_tokenizer_with_options(tokenizer_id, out_dir):
            return True
        if _convert_tokenizer_via_cli(tokenizer_id, out_dir, with_detokenizer=with_detokenizer):
            return True
    return False


def _download_gguf_via_cli(repo_id: str, filename: str, out_dir: Path) -> Optional[Path]:
    """Attempt GGUF download using `huggingface-cli` and return local path.

    Searches `out_dir` recursively for `filename` after download.

    If the file already exists locally, the download is skipped.
    """
    # If file already present, skip invoking the CLI to avoid re-downloads
    existing = next(out_dir.rglob(filename), None)
    if existing is not None:
        print(f"[hf-cli] Found existing {existing}, skipping download")
        return existing

    cmd = ["huggingface-cli", "download", repo_id, filename, "--local-dir", str(out_dir)]
    if shutil.which(cmd[0]) is None:
        print("[hf-cli] 'huggingface-cli' not found in PATH")
        return None
    try:
        print(f"[hf-cli] Running: {' '.join(cmd)}")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(res.stdout)
        if res.returncode != 0:
            return None
        # Try to locate the file under out_dir
        for p in out_dir.rglob(filename):
            return p
        return None
    except Exception as e:
        print(f"[hf-cli] Download failed: {e}")
        return None


class TransformersModel(ModelInterface):
    """Deprecated placeholder (removed)."""
    def __init__(self, *_, **__):
        raise RuntimeError("Transformers backend has been removed in this refactor.")


class EchoModel(ModelInterface):
    """Simple echo model for testing."""

    def __init__(self, prefix: str = "Echo: "):
        self.prefix = prefix

    def generate_response(self, message: str) -> str:
        return f"{self.prefix}{message}"


class LlamaCppModel(ModelInterface):
    """llama.cpp backend with optional NVIDIA GPU offload.

    Uses `llama-cpp-python`. If the wheel is built with CUDA support and an
    NVIDIA GPU is available, it offloads layers automatically.
    """

    def __init__(self, model_path: str, gguf_filename: Optional[str] = None, max_new_tokens: int | None = None):
        self.model_path = model_path
        self._gguf_filename = gguf_filename
        self.max_new_tokens = max_new_tokens or _env_int("LLM_MAX_TOKENS", 256)
        self._history = ChatHistory()
        self._llm = None
        self._load_model()

    @staticmethod
    def _has_nvidia_gpu() -> bool:
        try:
            out = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.DEVNULL)
            return bool(out.strip())
        except Exception:
            return False

    @staticmethod
    def _is_repo_id(path_or_id: str) -> bool:
        return "/" in path_or_id and not Path(path_or_id).drive

    def _resolve_gguf(self) -> Optional[Path]:
        raw = self.model_path
        p = Path(raw)
        if p.exists() and p.is_file() and p.suffix.lower() == ".gguf":
            return p
        if p.exists() and p.is_dir():
            # Try common GGUF names inside directory
            for cand in p.rglob("*.gguf"):
                return cand
            return None
        if self._is_repo_id(raw):
            # Resolve GGUF filename candidates
            env_gguf = os.environ.get("GGUF_FILENAME")
            if self._gguf_filename or env_gguf:
                candidates = [self._gguf_filename or env_gguf]  # type: ignore[list-item]
            else:
                candidates = [
                    "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
                    "qwen2.5-1.5b-instruct-q4_k_m.gguf",
                ]
            for filename in candidates:
                out_dir = Path(os.environ.get("HF_LOCAL_DIR", "models"))
                out_dir.mkdir(parents=True, exist_ok=True)
                cli_path = _download_gguf_via_cli(raw, filename, out_dir)
                if cli_path is not None:
                    return cli_path
        return None

    def _load_model(self) -> None:
        try:
            gguf = self._resolve_gguf()
            if gguf is None:
                print("[llama.cpp] GGUF not found; falling back to echo mode")
                self._llm = None
                return
            try:
                from llama_cpp import Llama  # type: ignore
            except Exception as e:
                print(f"[llama.cpp] 'llama-cpp-python' not available: {e}")
                self._llm = None
                return

            # Determine GPU offload layers
            n_gpu_layers = _env_int("LLAMA_N_GPU_LAYERS", -1 if self._has_nvidia_gpu() else 0)
            if n_gpu_layers < 0:
                # Try to offload all layers when possible; if it fails, llama-cpp will adjust
                n_gpu_layers = 99999

            n_ctx = _env_int("LLAMA_CTX", 4096)
            print(f"[llama.cpp] Loading {gguf} with n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx}")
            self._llm = Llama(
                model_path=str(gguf),
                n_ctx=n_ctx,
                seed=0,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
        except Exception as e:
            print(f"[llama.cpp] Failed to initialize: {e}")
            self._llm = None

    def set_system_prompt(self, prompt: str) -> None:
        self._history.set_system_prompt(prompt)

    def reset_history(self) -> None:
        self._history.reset()

    def _build_prompt(self, message: str) -> str:
        return self._history.build_prompt(next_user=message)

    def generate_response(self, message: str) -> str:
        if self._llm is None:
            return f"Echo: {message}"
        prompt = self._build_prompt(message)
        try:
            # Prefer completion API for broad compatibility
            out = self._llm.create_completion(prompt=prompt, temperature=0.7, max_tokens=self.max_new_tokens)
            text = (out.get("choices") or [{}])[0].get("text", "").strip()
        except Exception as e:
            return f"[llama.cpp generation error: {e}]"

        reply = text[len(prompt):].strip() if text.startswith(prompt) else text
        self._history.add_user(message)
        self._history.add_assistant(reply)
        return reply


class OpenVINOModel(ModelInterface):
    """OpenVINO GenAI backend for local optimized LLMs.

    Usage:
        Pass model identifier as "openvino:<path>" where <path> points to
        either:
          - a GGUF file, which will be loaded via openvino_genai.LLM, or
          - an OpenVINO-optimized model directory, loaded via TextGeneration.

        Examples:
            --model bartowski/Qwen2.5-1.5B-Instruct-GGUF

    Notes:
        - Requires the Python package "openvino-genai". If it is not installed,
          this class gracefully falls back to echo behavior and prints a hint.
        - Device can be selected via env var OPENVINO_DEVICE (CPU, GPU.0, etc.).
    """

    def __init__(self, model_path: str, device: Optional[str] = None, gguf_filename: Optional[str] = None, tokenizer_id: Optional[str] = None):
        self.model_path = model_path
        self.device = device or os.environ.get("OPENVINO_DEVICE", "CPU")
        self.max_new_tokens = _env_int("LLM_MAX_TOKENS", 256)
        self._gguf_filename = gguf_filename
        self._tokenizer_id = tokenizer_id

        self._pipe = None           # TextGeneration or LLM fallback
        self._cfg = None            # GenerationConfig for TextGeneration
        self._ov_pipe = None        # LLMPipeline for GGUF + Tokenizer path
        self._ov_cfg = None         # GenerationConfig for LLMPipeline
        self._ov_chat_started = False
        self._mode = ""  # "llm" for GGUF, "textgen" for OV IR
        self._history = ChatHistory()
        self._load_model()

    def _is_repo_id(self, path_or_id: str) -> bool:
        # Heuristic: repo ids look like "org/name" without drive letters
        return "/" in path_or_id and not Path(path_or_id).drive

    def _ensure_resources(self) -> Tuple[Optional[Path], Optional[Path]]:
        """Ensure local model resources for OpenVINO GGUF path and tokenizer.

        Returns (gguf_path, tokenizer_dir) if GGUF pipeline should be used.
        Returns (None, model_dir) if TextGeneration should be used for OV IR.
        """
        raw = self.model_path
        p = Path(raw)
        # Case 1: direct OV IR directory (contains openvino_model.xml or similar)
        if p.exists() and p.is_dir():
            # Let TextGeneration load this directory
            return (None, p)

        # Case 2: local GGUF path
        if p.exists() and p.is_file() and p.suffix.lower() == ".gguf":
            # Prefer explicit out dir via env, else alongside the GGUF file
            out_dir = Path(os.environ.get("TOKENIZER_OUT_DIR", p.parent))
            tok_xml = out_dir / "openvino_tokenizer.xml"
            if not tok_xml.exists():
                tok_id = (
                    self._tokenizer_id
                    or os.environ.get("TOKENIZER_ID")
                    or "Qwen/Qwen2.5-1.5B-Instruct"
                )
                print(f"[tokenizer] Converting '{tok_id}' into {out_dir} (local GGUF)...")
                ensure_tokenizer_converted(tok_id, out_dir, with_detokenizer=True)
            return (p, out_dir)

        # Case 3: treat as Hugging Face repo id and download GGUF
        if self._is_repo_id(raw):
            # Normalize common typos like '-GGU' -> '-GGUF'
            repo_id = raw
            if repo_id.endswith("-GGU"):
                repo_id += "F"

            # Resolve GGUF filename: prefer explicit arg/env, else try common variants
            env_gguf = os.environ.get("GGUF_FILENAME")
            if self._gguf_filename or env_gguf:
                candidates = [self._gguf_filename or env_gguf]  # type: ignore[list-item]
            else:
                # Try common names across repos (case sensitive on HF)
                candidates = [
                    # bartowski naming
                    "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
                    "Qwen2.5-1.5B-Instruct-Q5_K_M.gguf",
                    "Qwen2.5-1.5B-Instruct-Q4_K_S.gguf",
                    # lowercase variant used elsewhere
                    "qwen2.5-1.5b-instruct-q4_k_m.gguf",
                ]
            gguf_path = None
            for filename in candidates:
                out_dir = Path(os.environ.get("HF_LOCAL_DIR", "models"))
                out_dir.mkdir(parents=True, exist_ok=True)
                cli_path = _download_gguf_via_cli(repo_id, filename, out_dir)
                if cli_path is not None:
                    gguf_path = cli_path
                    break
            if gguf_path is None:
                print("[download] Failed to fetch any GGUF file. Tried:")
                for fn in candidates:
                    print(f" - {repo_id}/{fn}")
                print("Hint: ensure 'huggingface-cli' is installed (pip install -U huggingface-hub).")
                print("Hint: pass --gguf-filename <file.gguf> or set GGUF_FILENAME if your repo uses a different name.")
                return (None, None)

            # Tokenizer id: drop -GGUF suffix if present, or use provided env/arg
            default_tok_id = None
            if repo_id.endswith("-GGUF"):
                base_id = repo_id[:-6]
                # Map common community repos back to original model org for tokenizer
                if base_id.startswith("bartowski/"):
                    default_tok_id = "Qwen/Qwen2.5-1.5B-Instruct"
                else:
                    default_tok_id = base_id
            else:
                default_tok_id = repo_id
            tok_id = self._tokenizer_id or os.environ.get("TOKENIZER_ID") or default_tok_id
            # Choose output dir: env override, else project 'models' dir if present, else cache dir
            tok_dir = Path(os.environ.get("TOKENIZER_OUT_DIR") or (Path("models") if Path("models").exists() else gguf_path.parent))
            tok_xml = tok_dir / "openvino_tokenizer.xml"
            if not tok_xml.exists():
                print(f"[tokenizer] Converting tokenizer from '{tok_id}' into {tok_dir} (repo GGUF) ...")
                ensure_tokenizer_converted(tok_id, tok_dir, with_detokenizer=True)
            return (gguf_path, tok_dir)

        # Fallback: string path that doesn't exist; let TextGeneration try
        return (None, p if p.exists() else None)

    def _load_model(self) -> None:
        try:
            # Ensure Windows can locate dependent DLLs for OpenVINO/tokenizers
            _prepare_windows_openvino_dlls()
            gguf_path, aux_path = self._ensure_resources()
            if gguf_path is not None:
                # --- GGUF path: Tokenizer + LLMPipeline ---
                from openvino_genai import Tokenizer, LLMPipeline, GenerationConfig  # type: ignore
                print(f"[OV] LLMPipeline(GGUF) loading: {gguf_path} on {self.device}")
                tokenizer = Tokenizer(str(aux_path))  # aux_path is tokenizer dir
                pipe = LLMPipeline(str(gguf_path), tokenizer, self.device)
                cfg = GenerationConfig()
                cfg.max_new_tokens = self.max_new_tokens
                self._ov_pipe, self._ov_cfg, self._mode = pipe, cfg, "llm"
            else:
                # --- OV IR path or unresolved: try TextGeneration ---
                from openvino_genai import TextGeneration, GenerationConfig  # type: ignore
                model_dir = str(aux_path) if aux_path is not None else self.model_path
                print(f"[OV] TextGeneration(IR) loading: {model_dir} on {self.device}")
                self._pipe = TextGeneration(model_dir, device=self.device)
                try:
                    self._cfg = GenerationConfig(max_new_tokens=self.max_new_tokens, temperature=0.7)
                except Exception:
                    self._cfg = None
                self._mode = "textgen"
            print("OpenVINO model loaded!")
        except Exception as e:
            print("Failed to init OpenVINO GenAI. Tips:\n"
                " - `pip install -U openvino openvino-genai openvino-tokenizers`\n"
                " - GGUFは `convert_tokenizer` 必須\n"
                f"Error: {e}")
            self._pipe = self._cfg = self._ov_pipe = self._ov_cfg = None


    def is_ready(self) -> bool:
        return (self._ov_pipe is not None) or (self._pipe is not None)


    # --- chat-style conditioning ---
    def set_system_prompt(self, prompt: str) -> None:
        self._history.set_system_prompt(prompt)

    def reset_history(self) -> None:
        self._history.reset()

    def _build_prompt(self, message: str) -> str:
        return self._history.build_prompt(next_user=message)

    def generate_response(self, message: str) -> str:
        if self._pipe is None and self._ov_pipe is None:
            return f"Echo: {message}"

        prompt = self._build_prompt(message)
        try:
            if self._ov_pipe is not None:
                if not self._ov_chat_started:
                    self._ov_pipe.start_chat()
                    self._ov_chat_started = True
                cfg = self._ov_cfg
                out = self._ov_pipe.generate(prompt, cfg) if cfg is not None else self._ov_pipe.generate(prompt)
                reply_text = str(out).strip()
            elif self._mode == "llm":
                out = self._pipe.generate(prompt)
                reply_text = str(out).strip()
            else:
                # Some versions accept a config object; others accept kwargs
                if self._cfg is not None:
                    out = self._pipe.generate(prompt, self._cfg)
                else:
                    out = self._pipe.generate(prompt)
                reply_text = str(out).strip()
        except Exception as e:
            return f"[OpenVINO generation error: {e}]"

        # If the pipeline returns full text, strip the prompt prefix to get the reply.
        reply = reply_text[len(prompt):].strip() if reply_text.startswith(prompt) else reply_text
        self._history.add_user(message)
        self._history.add_assistant(reply)
        return reply


class ChatInterface:
    """Handles user interaction and chat flow."""
    
    def __init__(self, model: ModelInterface):
        self.model = model
        self.running = True
    
    def start(self):
        """Start the chat session."""
        self._print_welcome()
        
        while self.running:
            try:
                user_input = self._get_user_input()
                if self._should_exit(user_input):
                    break
                    
                response = self.model.generate_response(user_input)
                self._print_response(response)
                
            except KeyboardInterrupt:
                self._handle_interrupt()
                break
    
    def _print_welcome(self):
        """Print welcome message."""
        print("Chatbot ready! Type 'quit' to exit.\n")
    
    def _get_user_input(self) -> str:
        """Get input from user."""
        return input("You: ").strip()
    
    def _should_exit(self, user_input: str) -> bool:
        """Check if user wants to exit."""
        return user_input.lower() in ["quit", "exit"]
    
    def _print_response(self, response: str):
        """Print bot response."""
        print(f"Bot: {response}\n")
    
    def _handle_interrupt(self):
        """Handle Ctrl+C gracefully."""
        print("\nGoodbye!")


class ChatBotFactory:
    """Factory for creating different types of chatbots.

    - OCP: New backends plug in via `register_backend`.
    - DIP: Callers depend on `ModelInterface` and this registry API.
    """

    _BackendFactory = Callable[[str, Optional[str]], ModelInterface]
    _registry: Dict[str, _BackendFactory] = {}

    @staticmethod
    def _env_flag(name: str, default: str = "") -> bool:
        """Return True if the environment variable ``name`` looks truthy."""
        return os.environ.get(name, default).strip().lower() in {"1", "true", "yes"}

    @classmethod
    def register_backend(cls, name: str, factory: _BackendFactory) -> None:
        cls._registry[name.lower()] = factory

    @classmethod
    def create_model(cls, model_name: str, gguf_filename: Optional[str] = None, backend: Optional[str] = None) -> ModelInterface:
        """Select and build a model using the registered backend (OpenVINO by default)."""
        name = (model_name or "").strip()

        # Debug/echo short-circuit
        if cls._env_flag("CHATBOT_DEBUG") or name.lower() == "echo":
            return EchoModel(prefix="(debug) " if name.lower() != "echo" else "Echo: ")

        # Backend selection with sensible default
        choice = (backend or os.environ.get("LLM_BACKEND", "openvion")).strip().lower()
        if choice in {"", "auto"}:
            # Prefer llama.cpp when NVIDIA GPU is available (offload)
            try:
                has_nv = False
                try:
                    _ = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.DEVNULL)
                    has_nv = True
                except Exception:
                    has_nv = False
                choice = "llama" if has_nv else "openvion"
            except Exception:
                choice = "openvion"

        factory = cls._registry.get(choice)
        if factory is None:
            raise ValueError(f"Unsupported backend '{choice}'. Registered: {sorted(cls._registry.keys())}")

        # Map sentinel names to a default repo
        norm_name = name
        if norm_name.lower() in {"openvion", "openvino", "", "default"}:
            norm_name = "bartowski/Qwen2.5-1.5B-Instruct-GGUF"

        # Normalize explicit openvino:<path>
        ov_path = norm_name.split(":", 1)[1].strip() if norm_name.lower().startswith("openvino:") else norm_name
        return factory(ov_path, gguf_filename)


# Register default OpenVINO backend(s)
def _openvino_backend_factory(model_name: str, gguf_filename: Optional[str]) -> ModelInterface:
    return OpenVINOModel(model_path=model_name, gguf_filename=gguf_filename)


ChatBotFactory.register_backend("openvion", _openvino_backend_factory)
ChatBotFactory.register_backend("openvino", _openvino_backend_factory)


def _llama_backend_factory(model_name: str, gguf_filename: Optional[str]) -> ModelInterface:
    return LlamaCppModel(model_path=model_name, gguf_filename=gguf_filename)

ChatBotFactory.register_backend("llama", _llama_backend_factory)


# Shared model cache for quick reuse across instances
_SHARED_MODEL: Optional[ModelInterface] = None


def preload_model(model_name: str = "openvion", gguf_filename: Optional[str] = None, backend: Optional[str] = None) -> None:
    """Load the LLM backend asynchronously-safe for reuse.

    Subsequent ``SimpleChatBot`` instances reuse this shared model to avoid
    reload and large allocations during request handling.
    """
    global _SHARED_MODEL
    if _SHARED_MODEL is None:
        _SHARED_MODEL = ChatBotFactory.create_model(model_name, gguf_filename=gguf_filename, backend=backend)


class SimpleChatBot:
    """Backward compatible wrapper providing a minimal ``ask`` interface.

    Older parts of the project expect a ``SimpleChatBot`` class with an
    ``ask`` method.  The original implementation was refactored into modular
    components (``ModelInterface``, ``ChatBotFactory`` and friends).  This
    wrapper restores the previous API by delegating to a lightweight model
    created via :class:`ChatBotFactory`.

    Parameters
    ----------
    model_name:
        Name of the model backend to use. Defaults to
        ``"bartowski/Qwen2.5-1.5B-Instruct-GGUF"``.
    """

    def __init__(self, model_name: str = "bartowski/Qwen2.5-1.5B-Instruct-GGUF", gguf_filename: Optional[str] = None, backend: Optional[str] = None):
        global _SHARED_MODEL
        # Reuse a preloaded model if present; otherwise build a fresh one.
        if _SHARED_MODEL is None:
            self._model = ChatBotFactory.create_model(model_name, gguf_filename=gguf_filename, backend=backend)
        else:
            self._model = _SHARED_MODEL

    def ask(self, message: str) -> str:
        """Generate a reply for ``message`` using the underlying model."""
        return self._model.generate_response(message)

    def ask_stream(self, message: str):
        """Yield a reply incrementally.

        This is a simple wrapper around :meth:`ask` that yields the final
        response one character at a time.  If the underlying model provides a
        ``generate_response_stream`` method, it will be used instead to allow
        token-level streaming.
        """
        gen = getattr(self._model, "generate_response_stream", None)
        if callable(gen):
            for chunk in gen(message):
                yield chunk
        else:
            reply = self.ask(message)
            for ch in reply:
                yield ch

    # Expose optional hooks when available
    def set_system_prompt(self, prompt: str) -> None:
        model = self._model
        try:
            if isinstance(model, SupportsSystemPrompt):  # type: ignore[arg-type]
                model.set_system_prompt(prompt)
                return
        except Exception:
            pass
        if hasattr(model, "set_system_prompt"):
            getattr(model, "set_system_prompt")(prompt)

    def reset_history(self) -> None:
        model = self._model
        try:
            if isinstance(model, SupportsResetHistory):  # type: ignore[arg-type]
                model.reset_history()
                return
        except Exception:
            pass
        if hasattr(model, "reset_history"):
            getattr(model, "reset_history")()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simple modular chatbot")
    parser.add_argument(
        "--model",
        default="openvion",
        help=(
            "Model to use. Default: 'openvion' sentinel which maps to "
            "bartowski/Qwen2.5-1.5B-Instruct-GGUF with GGUF 'Qwen2.5-1.5B-Instruct-Q4_K_M.gguf'. "
            "Also accepts repo id (e.g. 'bartowski/Qwen2.5-1.5B-Instruct-GGUF'), 'echo', or "
            "'openvino:<path_to_ov_or_gguf>'."
        )
    )
    parser.add_argument(
        "--gguf-filename",
        default="Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        help="GGUF filename within the repo (used for OpenVINO GGUF downloads)"
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "openvion", "openvino", "llama"],
        default=os.environ.get("LLM_BACKEND", "openvion"),
        help="Backend: auto, openvion/openvino, or llama (default: openvion)."
    )
    parser.add_argument(
        "--auto-install-tokenizers",
        action="store_true",
        help="Allow runtime pip install for tokenizer conversion dependencies."
    )
    # Tokenizer conversion utility
    parser.add_argument(
        "--convert-tokenizer",
        metavar="TOKENIZER_ID",
        help="Convert a HF tokenizer (e.g., 'Qwen/Qwen2.5-1.5B-Instruct') to OpenVINO IR and exit."
    )
    parser.add_argument(
        "-o", "--tokenizer-out",
        default="models",
        help="Output directory for OpenVINO tokenizer files (default: models)."
    )
    parser.add_argument(
        "--with-detokenizer",
        action="store_true",
        help="Also convert and save OpenVINO detokenizer when supported."
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    if args.auto_install_tokenizers:
        # Signal optional runtime install to the OpenVINO path
        os.environ["AUTO_INSTALL_TOKENIZERS"] = "1"

    # Standalone tokenizer conversion mode
    if args.convert_tokenizer:
        out_dir = Path(args.tokenizer_out)
        ok = ensure_tokenizer_converted(args.convert_tokenizer, out_dir, with_detokenizer=args.with_detokenizer)
        if ok:
            print(f"[tokenizer] Saved to: {out_dir}")
            return
        else:
            print("[tokenizer] Conversion failed.")
            sys.exit(1)
    
    model = ChatBotFactory.create_model(args.model, gguf_filename=args.gguf_filename, backend=args.backend)
    chat = ChatInterface(model)
    chat.start()


if __name__ == "__main__":
    main()




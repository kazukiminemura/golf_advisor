"""Lightweight modular chatbot with unified prompt handling.

This refactor simplifies the original implementation by:
- Unifying chat prompt construction across backends
- Avoiding tensor-based history for transformers (reduces memory/device issues)
- Streamlining backend selection in a small factory
- Keeping backward compatibility via SimpleChatBot.ask
"""


import argparse
import os
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import time
import torch


class ModelInterface(ABC):
    """Abstract interface for different model backends."""
    
    @abstractmethod
    def generate_response(self, message: str) -> str:
        pass

    # Optional chat-style hooks (no-op by default)
    def set_system_prompt(self, prompt: str) -> None:  # pragma: no cover - optional
        return None

    def reset_history(self) -> None:  # pragma: no cover - optional
        return None


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


class TransformersModel(ModelInterface):
    """Hugging Face transformers model implementation."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.history = ChatHistory()
        self._load_model()
        # Shared generation limit across implementations
        self.max_new_tokens = _env_int("LLM_MAX_TOKENS", 256)
    
    def _load_model(self):
        """Load the transformer model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            print(f"Loading {self.model_name}...")
            # Many modern chat models (e.g. Qwen) require trust_remote_code
            # to load custom architectures/tokenizers.
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            # Try to place the model appropriately if CUDA is available.
            kwargs = {"trust_remote_code": True}
            if torch.cuda.is_available():
                kwargs.update({"device_map": "auto"})
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **kwargs,
            )
            print("Model loaded!")

        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None
            self.tokenizer = None
    
    def generate_response(self, message: str) -> str:
        """Generate a response using simple text prompting and decode delta."""
        if not self.model or not self.tokenizer:
            return f"Echo: {message}"

        # Build prompt including prior turns; append new user turn placeholder
        prompt = self.history.build_prompt(next_user=message)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens (answer portion)
        gen_ids = output_ids[0, input_ids.shape[-1]:]
        reply = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        # Update history
        self.history.add_user(message)
        self.history.add_assistant(reply)
        return reply
    
    # --- chat-style conditioning ---
    def set_system_prompt(self, prompt: str) -> None:
        self.history.set_system_prompt(prompt)

    def reset_history(self) -> None:
        self.history.reset()
    
    # Remove tensor-history specific helpers in favor of ChatHistory


class EchoModel(ModelInterface):
    """Simple echo model for testing."""

    def __init__(self, prefix: str = "Echo: "):
        self.prefix = prefix

    def generate_response(self, message: str) -> str:
        return f"{self.prefix}{message}"


class LlamaCppModel(ModelInterface):
    """llama.cpp backend for GGUF models (e.g., Qwen GGUF).

    This uses llama-cpp-python to load a model from Hugging Face Hub via
    `from_pretrained`, enabling the `chat_completion` interface.
    """

    def __init__(self, repo_id: str, filename: Optional[str] = None, n_ctx: int = 4096):
        self.repo_id = repo_id
        self.filename = filename or "qwen2.5-3b-instruct-q4_k_m.gguf"
        self.n_ctx = n_ctx
        self.llm = None
        self.history: List[Dict[str, str]] = []
        # Generation/token limits
        self.max_new_tokens = _env_int("LLM_MAX_TOKENS", 256)
        self._load_model()

    def _load_model(self):
        try:
            from llama_cpp import Llama
            threads = max(1, os.cpu_count() or 1)
            n_batch = _env_int("LLAMA_N_BATCH", 1024)
            n_gpu_layers = _env_int("LLAMA_N_GPU_LAYERS", 0)
            print(f"Loading GGUF from {self.repo_id}/{self.filename} ...")
            # Build args; include n_gpu_layers only when > 0 to avoid None issues
            kwargs = dict(
                repo_id=self.repo_id,
                filename=self.filename,
                n_ctx=self.n_ctx,
                n_threads=threads,
                n_batch=n_batch,
                use_mmap=True,
                chat_format="qwen",
            )
            if n_gpu_layers > 0:
                kwargs["n_gpu_layers"] = n_gpu_layers
            # Use built-in HF downloader; chat_format 'qwen' per llama.cpp support
            self.llm = Llama.from_pretrained(**kwargs)
            print("GGUF model loaded!")
        except Exception as e:
            print(f"Failed to load GGUF model: {e}")
            self.llm = None

    def generate_response(self, message: str) -> str:
        if not self.llm:
            return f"Echo: {message}"
        # Maintain a minimal chat history
        self.history.append({"role": "user", "content": message})
        try:
            result = self.llm.create_chat_completion(
                messages=self.history,
                temperature=0.7,
                max_tokens=self.max_new_tokens,
            )
            reply = result["choices"][0]["message"]["content"].strip()
            self.history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            return f"[Error generating response: {e}]"

    # --- chat-style conditioning ---
    def set_system_prompt(self, prompt: str) -> None:
        # Insert or replace a leading system message to steer behavior
        if self.history and self.history[0].get("role") == "system":
            self.history[0]["content"] = prompt
        else:
            self.history.insert(0, {"role": "system", "content": prompt})

    def reset_history(self) -> None:
        self.history = []


class OpenVINOModel(ModelInterface):
    """OpenVINO GenAI backend for local optimized LLMs.

    Usage:
        Pass model identifier as "openvino:<path>" where <path> points to
        either:
          - a GGUF file, which will be loaded via openvino_genai.LLM, or
          - an OpenVINO-optimized model directory, loaded via TextGeneration.

        Examples:
            --model openvino:C:\\models\\Qwen2.5-Instruct-3B.Q4_K_M.gguf
            --model openvino:C:\\models\\qwen2.5-3b-instruct-int4-ov

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
            tok_dir = p.parent
            return (p, tok_dir)

        # Case 3: treat as Hugging Face repo id and download GGUF
        if self._is_repo_id(raw):
            try:
                from huggingface_hub import hf_hub_download  # type: ignore
            except Exception as e:
                print(f"[hint] Install huggingface-hub for auto-download: pip install huggingface-hub\nError: {e}")
                return (None, None)

            filename = self._gguf_filename or os.environ.get("GGUF_FILENAME", "qwen2.5-3b-instruct-q4_k_m.gguf")
            try:
                local_file = hf_hub_download(repo_id=raw, filename=filename)
                gguf_path = Path(local_file)
            except Exception as e:
                print(f"[download] Failed to fetch GGUF {raw}/{filename}: {e}")
                return (None, None)

            # Tokenizer id: drop -GGUF suffix if present, or use provided env/arg
            tok_id = (
                self._tokenizer_id
                or os.environ.get("TOKENIZER_ID")
                or (raw[:-6] if raw.endswith("-GGUF") else raw)
            )
            tok_dir = gguf_path.parent
            tok_xml = tok_dir / "openvino_tokenizer.xml"
            if not tok_xml.exists():
                try:
                    from openvino_tokenizers import convert_tokenizer  # type: ignore
                except Exception as e:
                    print("[hint] Install openvino-tokenizers to auto-convert tokenizer: pip install openvino-tokenizers")
                    print(f"[tokenizer] Missing tokenizer IR at {tok_dir}; unable to convert automatically. Error: {e}")
                else:
                    try:
                        print(f"[tokenizer] Converting tokenizer from '{tok_id}' into {tok_dir} ...")
                        convert_tokenizer(tok_id, tok_dir)
                    except Exception as e:
                        print(f"[tokenizer] Conversion failed: {e}")
            return (gguf_path, tok_dir)

        # Fallback: string path that doesn't exist; let TextGeneration try
        return (None, p if p.exists() else None)

    def _load_model(self) -> None:
        try:
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
    """Factory for creating different types of chatbots."""

    @staticmethod
    def _env_flag(name: str, default: str = "") -> bool:
        """Return True if the environment variable ``name`` looks truthy."""
        return os.environ.get(name, default).strip().lower() in {"1", "true", "yes"}

    @classmethod
    def create_model(cls, model_name: str, gguf_filename: Optional[str] = None, backend: Optional[str] = None) -> ModelInterface:
        """Select and build a model with straightforward rules.

        Priority:
        1) CHATBOT_DEBUG -> Echo
        2) Explicit backend arg/env overrides detection
        3) URL-scheme-style or filename-based hints
        4) Default to transformers
        """
        name = (model_name or "").strip()
        if cls._env_flag("CHATBOT_DEBUG") or name.lower() == "echo":
            return EchoModel(prefix="(debug) " if name.lower() != "echo" else "Echo: ")

        # Default backend changed to OpenVINO unless explicitly overridden
        choice = (backend or os.environ.get("LLM_BACKEND", "openvino")).strip().lower()
        if choice == "openvino":
            ov_path = name.split(":", 1)[1].strip() if name.lower().startswith("openvino:") else name
            return OpenVINOModel(model_path=ov_path, gguf_filename=gguf_filename)
        if choice == "llama":
            if "/" in name and not name.lower().endswith(".gguf"):
                return LlamaCppModel(repo_id=name, filename=gguf_filename)
            return LlamaCppModel(repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF", filename=gguf_filename or name)
        if choice == "transformers":
            return TransformersModel(name)

        # Auto-detect
        lname = name.lower()
        if lname.startswith("openvino:"):
            return OpenVINOModel(model_path=name.split(":", 1)[1].strip())
        if lname.endswith(".gguf") or "gguf" in lname:
            if "/" in name and not lname.endswith(".gguf"):
                return LlamaCppModel(repo_id=name, filename=gguf_filename)
            return LlamaCppModel(repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF", filename=gguf_filename or name)
        return TransformersModel(name)


# Shared model cache for quick reuse across instances
_SHARED_MODEL: Optional[ModelInterface] = None


def preload_model(model_name: str = "Qwen/Qwen2.5-3B-Instruct-GGUF", gguf_filename: Optional[str] = None, backend: Optional[str] = None) -> None:
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
        ``"Qwen/Qwen2.5-3B-Instruct-GGUF"``.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct-GGUF", gguf_filename: Optional[str] = None, backend: Optional[str] = None):
        global _SHARED_MODEL
        # Reuse a preloaded model if present; otherwise build a fresh one.
        if _SHARED_MODEL is None:
            self._model = ChatBotFactory.create_model(model_name, gguf_filename=gguf_filename, backend=backend)
        else:
            self._model = _SHARED_MODEL

    def ask(self, message: str) -> str:
        """Generate a reply for ``message`` using the underlying model."""
        return self._model.generate_response(message)

    # Expose optional hooks when available
    def set_system_prompt(self, prompt: str) -> None:
        if hasattr(self._model, "set_system_prompt"):
            self._model.set_system_prompt(prompt)  # type: ignore[attr-defined]

    def reset_history(self) -> None:
        if hasattr(self._model, "reset_history"):
            self._model.reset_history()  # type: ignore[attr-defined]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simple modular chatbot")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct-GGUF",
        help=(
            "Model to use: repo 'Qwen/Qwen2.5-3B-Instruct-GGUF', 'echo', or "
            "'openvino:<path_to_ov_or_gguf>'"
        )
    )
    parser.add_argument(
        "--gguf-filename",
        default="qwen2.5-3b-instruct-q4_k_m.gguf",
        help="GGUF filename within the repo (for llama.cpp backend)"
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "openvino", "llama", "transformers"],
        default=os.environ.get("LLM_BACKEND", "openvino"),
        help="Backend: auto, openvino, llama, transformers (default: openvino)."
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    model = ChatBotFactory.create_model(args.model, gguf_filename=args.gguf_filename, backend=args.backend)
    chat = ChatInterface(model)
    chat.start()


if __name__ == "__main__":
    main()




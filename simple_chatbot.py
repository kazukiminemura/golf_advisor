"""Refactored modular chatbot with separation of concerns."""


import argparse
import os
from abc import ABC, abstractmethod
from typing import Optional, List, Dict
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


class TransformersModel(ModelInterface):
    """Hugging Face transformers model implementation."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.history = None
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
        """Generate response using the loaded model."""
        if not self.model:
            return f"Echo: {message}"
        
        inputs = self._encode_input(message)
        outputs = self._generate_output(inputs)
        return self._decode_response(outputs, inputs)
    
    # --- chat-style conditioning ---
    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt
        self._primed = False

    def reset_history(self) -> None:
        self.history = None
        self._primed = False
    
    def _encode_input(self, message: str) -> torch.Tensor:
        """Encode user input with history."""
        # Apply system prompt once to prime persona/context
        text = message
        if getattr(self, "system_prompt", None) and not getattr(self, "_primed", False):
            text = f"[SYSTEM]\n{self.system_prompt}\n[USER]\n{message}"
            self._primed = True
        inputs = self.tokenizer.encode(text + self.tokenizer.eos_token, return_tensors="pt")
        
        if self.history is not None:
            inputs = torch.cat([self.history, inputs], dim=-1)
            
        return inputs
    
    def _generate_output(self, inputs: torch.Tensor) -> torch.Tensor:
        """Generate model output."""
        with torch.no_grad():
            return self.model.generate(
                inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
    
    def _decode_response(self, outputs: torch.Tensor, inputs: torch.Tensor) -> str:
        """Decode and clean the response."""
        self.history = outputs
        response = self.tokenizer.decode(
            outputs[:, inputs.shape[-1]:][0], 
            skip_special_tokens=True
        )
        return response.strip()


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

    def __init__(self, model_path: str, device: Optional[str] = None):
        self.model_path = model_path
        self.device = device or os.environ.get("OPENVINO_DEVICE", "CPU")
        self.max_new_tokens = _env_int("LLM_MAX_TOKENS", 256)

        self._pipe = None
        self._cfg = None
        self._mode = ""  # "llm" for GGUF, "textgen" for OV IR
        self._context = ""
        self._primed = False
        self.system_prompt: Optional[str] = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            # Lazy imports based on mode to avoid ImportError for unavailable APIs
            if self.model_path.lower().endswith(".gguf"):
                from openvino_genai import LLM  # type: ignore
                print(f"Loading OpenVINO GenAI LLM (GGUF) from {self.model_path} ...")
                self._pipe = LLM(self.model_path)
                self._mode = "llm"
            else:
                # TextGeneration path (OV IR); GenerationConfig may vary by version
                try:
                    from openvino_genai import TextGeneration, GenerationConfig  # type: ignore
                except ImportError as ie:
                    # Provide a clearer hint for older installs lacking TextGeneration
                    raise RuntimeError(
                        "openvino-genai TextGeneration API not found. Install/upgrade: pip install -U openvino-genai"
                    ) from ie
                print(f"Loading OpenVINO TextGeneration from {self.model_path} on {self.device} ...")
                self._pipe = TextGeneration(self.model_path, device=self.device)
                try:
                    self._cfg = GenerationConfig(
                        max_new_tokens=self.max_new_tokens,
                        temperature=0.7,
                    )
                except Exception:
                    # If config class not available, proceed without it
                    self._cfg = None
                self._mode = "textgen"
            print("OpenVINO model loaded!")
        except Exception as e:
            print(
                "Failed to initialize OpenVINO GenAI. Install with 'pip install openvino-genai' "
                f"or verify the model path. Error: {e}"
            )
            self._pipe = None
            self._cfg = None

    # --- chat-style conditioning ---
    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt
        self._primed = False

    def reset_history(self) -> None:
        self._context = ""
        self._primed = False

    def _build_prompt(self, message: str) -> str:
        # Mimic a simple chat-style format. If the model was exported with a
        # chat template, GenAI may apply it; otherwise this basic prompting is used.
        if self.system_prompt and not self._primed:
            self._context += f"[SYSTEM]\n{self.system_prompt}\n"
            self._primed = True
        # Append new user turn
        prompt = f"{self._context}[USER]\n{message}\n[ASSISTANT]\n"
        return prompt

    def _update_context(self, user_message: str, assistant_reply: str) -> None:
        self._context += f"[USER]\n{user_message}\n[ASSISTANT]\n{assistant_reply}\n"

    def generate_response(self, message: str) -> str:
        if self._pipe is None or self._cfg is None:
            return f"Echo: {message}"

        prompt = self._build_prompt(message)
        try:
            if self._mode == "llm":
                out = self._pipe.generate(prompt)
            else:
                # Some versions accept a config object; others accept kwargs
                if self._cfg is not None:
                    out = self._pipe.generate(prompt, self._cfg)
                else:
                    out = self._pipe.generate(prompt)
        except Exception as e:
            return f"[OpenVINO generation error: {e}]"

        # If the pipeline returns full text, strip the prompt prefix to get the reply.
        reply = out[len(prompt):].strip() if isinstance(out, str) and out.startswith(prompt) else str(out).strip()
        self._update_context(message, reply)
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
        """Create appropriate model based on name or debug env flag."""
        if cls._env_flag("CHATBOT_DEBUG"):
            return EchoModel(prefix="(debug) ")
        if model_name.lower() == "echo":
            return EchoModel()
        # Optional explicit backend via arg or env var LLM_BACKEND
        backend_choice = (backend or os.environ.get("LLM_BACKEND", "auto")).strip().lower()
        if backend_choice in {"openvino", "llama", "transformers"}:
            if backend_choice == "openvino":
                ov_path = (
                    model_name.split(":", 1)[1].strip()
                    if model_name.lower().startswith("openvino:")
                    else model_name
                )
                return OpenVINOModel(model_path=ov_path)
            if backend_choice == "llama":
                if "/" in model_name and not model_name.lower().endswith(".gguf"):
                    return LlamaCppModel(repo_id=model_name, filename=gguf_filename)
                return LlamaCppModel(repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF", filename=gguf_filename or model_name)
            # transformers
            return TransformersModel(model_name)
        # OpenVINO backend via openvino-genai using scheme: openvino:<path>
        if model_name.lower().startswith("openvino:"):
            ov_path = model_name.split(":", 1)[1].strip()
            return OpenVINOModel(model_path=ov_path)
        # GGUF backend via llama.cpp if the identifier suggests GGUF
        if model_name.lower().endswith(".gguf") or "gguf" in model_name.lower():
            if "/" in model_name and not model_name.lower().endswith(".gguf"):
                # Treat as repo_id for GGUF collection
                return LlamaCppModel(repo_id=model_name, filename=gguf_filename)
            # Otherwise assume it's a direct local/remote filename (not implemented here)
            return LlamaCppModel(repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF", filename=gguf_filename or model_name)
        # Fallback to transformers for non-GGUF identifiers
        return TransformersModel(model_name)


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
        default=os.environ.get("LLM_BACKEND", "auto"),
        help="Backend: auto, openvino, llama, transformers (overrides auto-detect)."
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




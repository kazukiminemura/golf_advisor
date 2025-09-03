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


class TransformersModel(ModelInterface):
    """Hugging Face transformers model implementation."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.history = None
        self._load_model()
    
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
    
    def _encode_input(self, message: str) -> torch.Tensor:
        """Encode user input with history."""
        inputs = self.tokenizer.encode(
            message + self.tokenizer.eos_token, 
            return_tensors="pt"
        )
        
        if self.history is not None:
            inputs = torch.cat([self.history, inputs], dim=-1)
            
        return inputs
    
    def _generate_output(self, inputs: torch.Tensor) -> torch.Tensor:
        """Generate model output."""
        with torch.no_grad():
            return self.model.generate(
                inputs,
                max_length=inputs.shape[-1] + 50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
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
        self._load_model()

    def _load_model(self):
        try:
            from llama_cpp import Llama
            threads = max(1, os.cpu_count() or 1)
            print(f"Loading GGUF from {self.repo_id}/{self.filename} ...")
            # Use built-in HF downloader; chat_format 'qwen' per llama.cpp support
            self.llm = Llama.from_pretrained(
                repo_id=self.repo_id,
                filename=self.filename,
                n_ctx=self.n_ctx,
                n_threads=threads,
                chat_format="qwen",
            )
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
                max_tokens=512,
            )
            reply = result["choices"][0]["message"]["content"].strip()
            self.history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            return f"[Error generating response: {e}]"


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
    def create_model(cls, model_name: str, gguf_filename: Optional[str] = None) -> ModelInterface:
        """Create appropriate model based on name or debug env flag."""
        if cls._env_flag("CHATBOT_DEBUG"):
            return EchoModel(prefix="(デバッグ) ")
        if model_name.lower() == "echo":
            return EchoModel()
        # GGUF backend via llama.cpp if the identifier suggests GGUF
        if model_name.lower().endswith(".gguf") or "gguf" in model_name.lower():
            if "/" in model_name and not model_name.lower().endswith(".gguf"):
                # Treat as repo_id for GGUF collection
                return LlamaCppModel(repo_id=model_name, filename=gguf_filename)
            # Otherwise assume it's a direct local/remote filename (not implemented here)
            return LlamaCppModel(repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF", filename=gguf_filename or model_name)
        # Fallback to transformers for non-GGUF identifiers
        return TransformersModel(model_name)


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

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct-GGUF", gguf_filename: Optional[str] = None):
        self._model = ChatBotFactory.create_model(model_name, gguf_filename=gguf_filename)

    def ask(self, message: str) -> str:
        """Generate a reply for ``message`` using the underlying model."""
        return self._model.generate_response(message)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simple modular chatbot")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct-GGUF",
        help="Model to use (e.g., repo 'Qwen/Qwen2.5-3B-Instruct-GGUF' or 'echo')"
    )
    parser.add_argument(
        "--gguf-filename",
        default="qwen2.5-3b-instruct-q4_k_m.gguf",
        help="GGUF filename within the repo (for llama.cpp backend)"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    model = ChatBotFactory.create_model(args.model, gguf_filename=args.gguf_filename)
    chat = ChatInterface(model)
    chat.start()


if __name__ == "__main__":
    main()

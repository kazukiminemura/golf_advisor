"""Refactored modular chatbot with separation of concerns."""

import argparse
from abc import ABC, abstractmethod
from typing import Optional
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
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
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
    
    def generate_response(self, message: str) -> str:
        return f"Echo: {message}"


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
    def create_model(model_name: str) -> ModelInterface:
        """Create appropriate model based on name."""
        if model_name.lower() == "echo":
            return EchoModel()
        else:
            return TransformersModel(model_name)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simple modular chatbot")
    parser.add_argument(
        "--model", 
        default="microsoft/DialoGPT-medium", 
        help="Model to use (or 'echo' for testing)"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    model = ChatBotFactory.create_model(args.model)
    chat = ChatInterface(model)
    chat.start()


if __name__ == "__main__":
    main()
"""Terminal based conversational chatbot."""

import argparse
import torch


class SimpleChatBot:
    """Basic conversational chatbot using a small language model."""

    def __init__(self, model_name: str = "Qwen/Qwen3-8B"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:  # BitsAndBytes is optional and may not be installed
            from transformers import BitsAndBytesConfig
        except Exception:  # pragma: no cover - import fallback
            BitsAndBytesConfig = None

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            model_kwargs = {"trust_remote_code": True, "device_map": "auto"}
            if BitsAndBytesConfig is not None:
                try:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True
                    )
                except Exception:  # pragma: no cover - quantization optional
                    pass
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, **model_kwargs
            )
        except Exception:  # pragma: no cover - network-related
            # Fallback to a trivial echo mode if the model can't be downloaded.
            self.tokenizer = None
            self.model = None
        self.chat_history = None

    def ask(self, message: str) -> str:
        """Return a reply to *message* using the conversation history."""
        if not self.tokenizer or not self.model:
            return f"[no-model] You said: {message}"

        new_user_input_ids = self.tokenizer.encode(
            message + self.tokenizer.eos_token, return_tensors="pt"
        )
        bot_input_ids = (
            torch.cat([self.chat_history, new_user_input_ids], dim=-1)
            if self.chat_history is not None
            else new_user_input_ids
        )
        output_ids = self.model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        self.chat_history = output_ids
        response = self.tokenizer.decode(
            output_ids[:, bot_input_ids.shape[-1] :][0], skip_special_tokens=True
        )
        return response.strip()


def main() -> None:
    """Launch a simple command line chat session."""
    parser = argparse.ArgumentParser(description="Chat with a small language model")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
        help="HuggingFace model name to use",
    )
    args = parser.parse_args()

    bot = SimpleChatBot(model_name=args.model)
    print("SimpleChatBot ready. Type 'exit' to quit.")

    while True:
        try:
            message = input("You: ").strip()
        except EOFError:  # pragma: no cover - handles pipe input ending
            print()
            break

        if message.lower() in {"exit", "quit"}:
            print("Bot: Goodbye!")
            break

        response = bot.ask(message)
        print(f"Bot: {response}")


if __name__ == "__main__":
    main()

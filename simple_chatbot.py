import torch


class SimpleChatBot:
    """Basic conversational chatbot using a small language model."""

    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.chat_history = None

    def ask(self, message: str) -> str:
        """Return a reply to *message* using the conversation history."""
        new_user_input_ids = self.tokenizer.encode(message + self.tokenizer.eos_token, return_tensors="pt")
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
            output_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True
        )
        return response.strip()

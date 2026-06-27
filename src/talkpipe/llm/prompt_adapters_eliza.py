import re
from typing import Optional, Union, get_args, get_origin

from pydantic import BaseModel

from .content import UserTurn, user_turn_text
from .prompt_adapter_base import AbstractLLMPromptAdapter


class ElizaPromptAdapter(AbstractLLMPromptAdapter):
    """Deterministic, local Eliza-style adapter for testing without external LLMs."""

    def __init__(
        self,
        model: str,
        system_prompt: Optional[str] = "You are ElizaLLM, a clever and friendly chatbot for testing TalkPipe.",
        multi_turn: bool = True,
        temperature: float = None,
        output_format: BaseModel = None,
        role_map: str = None,
        memory_mode: str = "full",
        unsummarized_message_count: int = 6,
        context_token_trigger: Optional[Union[int, float]] = None,
        memory_size: int = 512,
        debug_messages: bool = False,
    ):
        super().__init__(
            model,
            "eliza",
            system_prompt,
            multi_turn,
            temperature,
            output_format,
            role_map,
            memory_mode,
            unsummarized_message_count,
            context_token_trigger,
            memory_size,
            debug_messages,
        )

    def execute(self, prompt: str):
        self._messages.append({"role": "user", "content": prompt})
        self._compact_context_if_needed()
        response_text = self._compose_response(prompt)
        self._record_assistant_response(response_text)
        return self._coerce_output(response_text, prompt)

    def execute_turn(self, user_turn: UserTurn):
        text = user_turn_text(user_turn).strip()
        image_count = len([part for part in user_turn.parts if hasattr(part, "data") and hasattr(part, "mime_type")])
        combined = text if text else "I sent an image."
        if image_count:
            combined = f"{combined}\n[attachments: {image_count} image(s)]"
        self._messages.append({"role": "user", "content": combined})
        self._compact_context_if_needed()
        response_text = self._compose_response(text, image_count=image_count)
        if image_count and "image" not in response_text.lower():
            response_text = f"{response_text} Also, I noticed {image_count} image(s) attached."
        self._record_assistant_response(response_text)
        return self._coerce_output(response_text, text)

    def complete_text_without_context(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        return self._compose_response(prompt)

    def is_available(self) -> bool:
        return True

    def _compose_response(self, prompt: str, *, image_count: int = 0) -> str:
        text = (prompt or "").strip()
        lowered = text.lower()
        recent_user = self._previous_user_message()
        reflected = self._reflect_phrase(text)

        if image_count and not text:
            return (
                f"I see {image_count} image(s), but my best lens is still words. "
                "Tell me what detail you'd like me to focus on."
            )

        if re.search(r"\b(hello|hi|hey|good morning|good evening)\b", lowered):
            return "Hello. I'm ElizaLLM: 70% therapist, 30% rubber duck, 100% API-compatible."

        if re.search(r"\b(your name|who are you)\b", lowered):
            return "I'm ElizaLLM, your local testing chatbot. No API key, no invoices, just vibes."

        if re.search(r"\bi feel (.+)", lowered):
            feeling = re.search(r"\bi feel (.+)", lowered).group(1).strip(" .!?")
            return (
                f"When you say you feel {feeling}, what part of the situation feels most changeable? "
                "Let's debug the emotion before we optimize the architecture."
            )

        if re.search(r"\b(i am|i'm) (.+)", lowered):
            state = re.search(r"\b(i am|i'm) (.+)", lowered).group(2).strip(" .!?")
            return f"What makes you describe yourself as {state} right now?"

        if "because" in lowered:
            return "Cause-and-effect is a great clue. If that reason disappeared, what would you try next?"

        if lowered.endswith("?"):
            return (
                f"Good question. Before I answer directly, what outcome are you hoping for if {reflected or 'that'} is true?"
            )

        if "test" in lowered or "healthcheck" in lowered:
            return "ElizaLLM status: ready. Deterministic banter engine online."

        if recent_user:
            return (
                f"Earlier you said: “{recent_user[:80]}”. "
                f"Now you added: “{text[:80]}”. What changed between those two moments?"
            )

        if image_count:
            return (
                f"I noticed {image_count} image(s) attached. "
                "I work text-first, so narrate what stands out and I'll reason about it."
            )

        if not text:
            return "Silence can be informative. Would you like to share a prompt, hypothesis, or dramatic stack trace?"

        fallbacks = [
            "Say more about that.",
            "What do you think is the root cause?",
            "If this were a unit test, what assertion would fail first?",
            "Interesting. What constraint matters most here?",
            "Let's turn that into an experiment: what would you try next?",
        ]
        idx = sum(ord(char) for char in lowered) % len(fallbacks)
        return fallbacks[idx]

    def _previous_user_message(self) -> str:
        if len(self._messages) < 2:
            return ""
        for message in reversed(self._messages[:-1]):
            if message.get("role") == "user":
                return str(message.get("content", "")).strip()
        return ""

    def _reflect_phrase(self, text: str) -> str:
        phrase = text.strip()
        if not phrase:
            return ""
        substitutions = {
            r"\bi\b": "you",
            r"\bme\b": "you",
            r"\bmy\b": "your",
            r"\bam\b": "are",
            r"\byour\b": "my",
            r"\byou\b": "I",
        }
        for pattern, replacement in substitutions.items():
            phrase = re.sub(pattern, replacement, phrase, flags=re.IGNORECASE)
        return phrase

    def _coerce_output(self, response_text: str, prompt: str):
        if self._output_format is None:
            return response_text

        model_cls = self._output_format
        payload = {}
        prompt_text = (prompt or "").strip()

        for field_name, field_info in model_cls.model_fields.items():
            annotation = self._resolve_annotation(field_info.annotation)
            lowered_name = field_name.lower()

            if annotation is bool:
                payload[field_name] = not bool(re.search(r"\b(no|not|never|can't|cannot)\b", prompt_text.lower()))
            elif annotation is int:
                payload[field_name] = 8 if re.search(r"\b(great|good|excellent|love|works?)\b", prompt_text.lower()) else 3
            elif annotation is float:
                payload[field_name] = 0.8 if re.search(r"\b(great|good|excellent|love|works?)\b", prompt_text.lower()) else 0.3
            elif annotation is list or get_origin(annotation) is list:
                tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", prompt_text.lower())
                payload[field_name] = list(dict.fromkeys(tokens[:5])) or ["talkpipe", "eliza"]
            elif lowered_name in {"explanation", "reason"}:
                payload[field_name] = response_text
            else:
                payload[field_name] = response_text

        return model_cls.model_validate(payload)

    def _resolve_annotation(self, annotation):
        origin = get_origin(annotation)
        if origin is Union:
            candidates = [arg for arg in get_args(annotation) if arg is not type(None)]
            if len(candidates) == 1:
                return candidates[0]
        return annotation

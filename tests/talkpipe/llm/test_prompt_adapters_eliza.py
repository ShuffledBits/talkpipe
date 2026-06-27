from pydantic import BaseModel

from talkpipe.llm.content import UserTurn, TextPart, user_turn_from_fields
from talkpipe.llm.prompt_adapters import ElizaPromptAdapter


def test_eliza_execute_tracks_history_and_returns_string():
    adapter = ElizaPromptAdapter("eliza-v1", multi_turn=True)

    first = adapter.execute("Hello there")
    second = adapter.execute("I feel stuck with my test pipeline")

    assert isinstance(first, str) and first.strip()
    assert isinstance(second, str) and second.strip()
    assert adapter._messages[-1]["role"] == "assistant"
    assert len(adapter._messages) == 4


def test_eliza_single_turn_clears_messages_and_summary():
    adapter = ElizaPromptAdapter("eliza-v1", multi_turn=False)
    adapter._summary_message = {"role": "system", "content": "old summary"}

    result = adapter.execute("healthcheck")

    assert "ready" in result.lower()
    assert adapter._messages == []
    assert adapter._summary_message is None


def test_eliza_execute_turn_handles_multimodal_prompt():
    adapter = ElizaPromptAdapter("eliza-v1")
    user_turn = user_turn_from_fields(prompt="What do you notice?", images=[b"fake-image-bytes"])

    result = adapter.execute_turn(user_turn)

    assert isinstance(result, str)
    assert "text-first" in result.lower() or "image" in result.lower()
    assert adapter._messages[0]["role"] == "user"
    assert "attachments: 1 image(s)" in adapter._messages[0]["content"]


def test_eliza_complete_text_without_context_does_not_mutate_history():
    adapter = ElizaPromptAdapter("eliza-v1")
    adapter.execute("Hello")

    before = list(adapter._messages)
    response = adapter.complete_text_without_context("Respond to test", max_tokens=8)

    assert isinstance(response, str)
    assert response.strip()
    assert adapter._messages == before


def test_eliza_is_always_available():
    adapter = ElizaPromptAdapter("eliza-v1")
    assert adapter.is_available() is True


class ScoreAnswer(BaseModel):
    explanation: str
    score: int


class BinaryAnswer(BaseModel):
    explanation: str
    answer: bool


class ExtractedTerms(BaseModel):
    terms: list[str]


class UnionValue(BaseModel):
    value: int | str


def test_eliza_output_format_returns_pydantic_models():
    score_adapter = ElizaPromptAdapter("eliza-v1", output_format=ScoreAnswer)
    binary_adapter = ElizaPromptAdapter("eliza-v1", output_format=BinaryAnswer)
    terms_adapter = ElizaPromptAdapter("eliza-v1", output_format=ExtractedTerms)

    score_result = score_adapter.execute("This works great")
    binary_false_result = binary_adapter.execute("This is not working")
    binary_true_result = binary_adapter.execute("This is working well")
    terms_result = terms_adapter.execute("Extract terms from talkpipe eliza adapter tests")

    assert isinstance(score_result, ScoreAnswer)
    assert score_result.explanation
    assert isinstance(score_result.score, int)
    assert isinstance(binary_false_result, BinaryAnswer)
    assert binary_false_result.answer is False
    assert isinstance(binary_true_result, BinaryAnswer)
    assert binary_true_result.answer is True
    assert isinstance(terms_result, ExtractedTerms)
    assert terms_result.terms


def test_eliza_output_format_handles_multi_type_union_fields():
    adapter = ElizaPromptAdapter("eliza-v1", output_format=UnionValue)
    result = adapter.execute("This works great")
    assert isinstance(result, UnionValue)
    assert isinstance(result.value, int)


def test_eliza_execute_turn_with_text_only_user_turn():
    adapter = ElizaPromptAdapter("eliza-v1")
    turn = UserTurn(parts=[TextPart(text="I am curious")])

    response = adapter.execute_turn(turn)

    assert isinstance(response, str)
    assert "curious" in response.lower() or "describe yourself" in response.lower()

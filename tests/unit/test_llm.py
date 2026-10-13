from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from idengine.config import LLMSettings
from idengine.errors import LLMError
from idengine.llm import (
    ExtractiveLLM,
    LLMRequest,
    OllamaLLM,
    OpenAICompatibleLLM,
    Passage,
    Task,
    create_llm,
)
from idengine.llm.base import parse_citations
from idengine.llm.extractive import expected_answer_types
from idengine.llm.prompts import NOT_FOUND_ANSWER, render_messages

PASSAGES = [
    Passage(1, "The total amount due is $8,940.00. Payment is due within 45 days.", "Invoice"),
    Passage(2, "The lease term is five years beginning July 1, 2023. Rent is $14,500.", "Lease"),
    Passage(3, "The budget was approved by the Chief Financial Officer, Maria Gonzalez.", "Memo"),
    Passage(4, "The autoscaler keeps between 3 and 12 replicas under load.", "Runbook"),
]


def ask(question: str, passages: list[Passage] = PASSAGES) -> LLMRequest:
    return LLMRequest(Task.ANSWER, question, passages, max_sentences=2)


class TestExtractive:
    @pytest.mark.parametrize(
        ("question", "short", "citation"),
        [
            ("What is the total amount due?", "$8,940.00", 1),
            ("When does the lease term begin?", "July 1, 2023", 2),
            ("Who approved the budget?", "Maria Gonzalez", 3),
            ("How many replicas does the autoscaler keep?", "between 3 and 12", 4),
            ("How long do customers have to make the payment?", "45 days", 1),
        ],
    )
    def test_typed_short_answers_with_citations(
        self, question: str, short: str, citation: int
    ) -> None:
        response = ExtractiveLLM().generate(ask(question))
        assert response.answerable
        assert response.short_answer == short
        assert response.citations[0] == citation
        assert f"[{citation}]" in response.text

    def test_declines_when_nothing_matches(self) -> None:
        response = ExtractiveLLM().generate(ask("What is the boiling point of mercury?"))
        assert not response.answerable
        assert response.text == NOT_FOUND_ANSWER and response.citations == []
        assert not ExtractiveLLM().generate(ask("anything", passages=[])).answerable

    def test_answer_types(self) -> None:
        assert expected_answer_types("Who signed?") == ("PERSON", "ORG")
        assert expected_answer_types("How many weeks of paid leave?") == ("NUMBER",)
        assert expected_answer_types("What is the e-mail address?") == ("EMAIL",)
        assert expected_answer_types("Which law governs the contract?") == ("LOCATION",)
        assert expected_answer_types("Tell me about it") == ()

    def test_summarize_cites_sources(self) -> None:
        request = LLMRequest(Task.SUMMARIZE, "", PASSAGES, max_sentences=2)
        response = ExtractiveLLM().generate(request)
        assert response.text and set(response.citations) <= {1, 2, 3, 4}
        empty = ExtractiveLLM().generate(LLMRequest(Task.SUMMARIZE, "", []))
        assert empty.text == "" and not empty.answerable


def test_parse_citations_validates_numbers() -> None:
    assert parse_citations("A [2]. B [1, 3]. C [9]. D [2].", {1, 2, 3}) == [2, 1, 3]


def test_prompt_rendering() -> None:
    messages = render_messages(ask("Q?"))
    assert messages[0]["role"] == "system" and "[1]" in messages[1]["content"]
    assert "(Invoice)" in messages[1]["content"]
    summary = render_messages(LLMRequest(Task.SUMMARIZE, "", PASSAGES[:1], max_sentences=4))
    assert "at most 4 sentences" in summary[1]["content"]


Handler = Callable[[httpx.Request], httpx.Response]


def transport(
    *responses: httpx.Response | Exception,
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    seen: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return httpx.MockTransport(handler), seen


def no_sleep(_: float) -> None:
    return None


class TestOllama:
    def test_success(self) -> None:
        mock, seen = transport(
            httpx.Response(
                200,
                json={
                    "message": {"content": "Due in 45 days [1]."},
                    "prompt_eval_count": 50,
                    "eval_count": 7,
                },
            )
        )
        llm = OllamaLLM(base_url="http://ollama:11434/", model="llama3", transport=mock)
        response = llm.generate(ask("When is payment due?"))
        assert response.text == "Due in 45 days [1]." and response.citations == [1]
        assert response.usage == {"prompt_tokens": 50, "completion_tokens": 7}
        body = json.loads(seen[0].content)
        assert seen[0].url.path == "/api/chat"
        assert body["model"] == "llama3" and body["stream"] is False
        assert body["options"]["temperature"] == 0.0

    def test_retries_transient_failures(self) -> None:
        mock, seen = transport(
            httpx.ConnectError("refused"),
            httpx.Response(503, text="busy"),
            httpx.Response(200, json={"message": {"content": NOT_FOUND_ANSWER}}),
        )
        llm = OllamaLLM(
            base_url="http://o", model="m", transport=mock, max_retries=2, sleep=no_sleep
        )
        response = llm.generate(ask("q?"))
        assert len(seen) == 3
        assert not response.answerable

    def test_gives_up_after_retries(self) -> None:
        mock, seen = transport(httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow"))
        llm = OllamaLLM(
            base_url="http://o", model="m", transport=mock, max_retries=1, sleep=no_sleep
        )
        with pytest.raises(LLMError, match="timed out"):
            llm.generate(ask("q?"))
        assert len(seen) == 2

    @pytest.mark.parametrize(
        "response",
        [
            httpx.Response(400, text="bad request"),
            httpx.Response(200, text="not json"),
            httpx.Response(200, json=["list"]),
            httpx.Response(200, json={"message": {}}),
        ],
    )
    def test_non_retryable_and_malformed(self, response: httpx.Response) -> None:
        mock, seen = transport(response)
        llm = OllamaLLM(base_url="http://o", model="m", transport=mock, sleep=no_sleep)
        with pytest.raises(LLMError):
            llm.generate(ask("q?"))
        assert len(seen) == 1

    def test_health(self) -> None:
        mock, _ = transport(httpx.Response(200, json={"models": []}))
        assert OllamaLLM(base_url="http://o", model="m", transport=mock).health()["ok"]
        failing, _ = transport(httpx.ConnectError("down"))
        assert not OllamaLLM(base_url="http://o", model="m", transport=failing).health()["ok"]


class TestOpenAICompatible:
    def test_success_with_auth_and_path(self) -> None:
        mock, seen = transport(
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "It is $8,940.00 [1]."}}],
                    "usage": {"prompt_tokens": 80, "completion_tokens": 9},
                },
            )
        )
        llm = OpenAICompatibleLLM(
            base_url="http://local/v1", model="gpt", api_key="sk-x", transport=mock
        )
        response = llm.generate(ask("What is due?"))
        assert response.citations == [1] and response.usage["completion_tokens"] == 9
        assert seen[0].url.path == "/v1/chat/completions"
        assert seen[0].headers["authorization"] == "Bearer sk-x"
        assert json.loads(seen[0].content)["max_tokens"] == 512

    def test_path_without_v1_and_missing_content(self) -> None:
        mock, seen = transport(httpx.Response(200, json={"choices": []}))
        llm = OpenAICompatibleLLM(base_url="http://local", model="m", transport=mock)
        with pytest.raises(LLMError, match="no message content"):
            llm.generate(ask("q?"))
        assert seen[0].url.path == "/v1/chat/completions"
        assert "authorization" not in seen[0].headers


def test_factory_defaults_to_offline_extractive() -> None:
    assert isinstance(create_llm(LLMSettings()), ExtractiveLLM)
    mock, _ = transport()
    ollama = create_llm(LLMSettings(provider="ollama", base_url="http://o"), transport=mock)
    assert isinstance(ollama, OllamaLLM)
    openai = create_llm(
        LLMSettings(provider="openai", base_url="http://o", api_key="k"),  # type: ignore[arg-type]
        transport=mock,
    )
    assert isinstance(openai, OpenAICompatibleLLM)
    openai.close()

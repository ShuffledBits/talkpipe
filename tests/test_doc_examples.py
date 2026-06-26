"""Pytest tests for documentation examples extracted from markdown."""

import os
import shutil

import pytest
from pathlib import Path
from huggingface_hub.errors import LocalEntryNotFoundError

from talkpipe.app.doc_examples import (
    classify_unavailable_example_exception,
    detect_example_requirements,
    extract_all_examples,
    run_example,
    unavailable_example_reason,
)

# Project root: tests/ -> parent
_project_root = Path(__file__).resolve().parent.parent

# Artifacts created by doc examples (e.g. README RAG example creates my_knowledge_base)
_DOC_EXAMPLE_ARTIFACTS = ["my_knowledge_base", "my_kb"]


def _is_safe_to_delete(root: Path, name: str) -> bool:
    """Return True only if name is a safe child of root (no path traversal)."""
    if not name or "/" in name or "\\" in name or ".." in name:
        return False
    try:
        child = (root / name).resolve()
        root_resolved = root.resolve()
        return child != root_resolved and child.is_relative_to(root_resolved)
    except (OSError, RuntimeError, ValueError):
        return False


def test_is_safe_to_delete():
    """Ensure cleanup only deletes paths strictly under project root."""
    root = _project_root
    assert _is_safe_to_delete(root, "my_knowledge_base") is True
    assert _is_safe_to_delete(root, "foo") is True
    assert _is_safe_to_delete(root, "") is False
    assert _is_safe_to_delete(root, "..") is False
    assert _is_safe_to_delete(root, "a/b") is False
    assert _is_safe_to_delete(root, "a\\b") is False
    assert _is_safe_to_delete(root, "../etc") is False


def test_detect_example_requirements():
    """Classify common provider-dependent doc examples so offline examples can still run."""
    assert detect_example_requirements('| llmPrompt[source="ollama"]') == {"ollama"}
    assert detect_example_requirements('| llmPrompt[source="openai"]') == {"openai"}
    assert detect_example_requirements('| llmEmbed[source="model2vec"]') == {"model2vec"}
    assert detect_example_requirements("| mongoSearch") == {"mongodb"}
    assert detect_example_requirements('print("hello")') == set()


def test_classify_unavailable_example_exception():
    """Recognize dependency-availability failures that should be skipped, not failed."""
    assert (
        classify_unavailable_example_exception(
            ConnectionError(
                "Failed to connect to Ollama. Please check that Ollama is downloaded, running and accessible."
            )
        )
        == "ollama"
    )
    assert (
        classify_unavailable_example_exception(
            LocalEntryNotFoundError(
                "An error happened while trying to locate the files on the Hub, and we cannot find the appropriate snapshot folder for the specified revision on the local disk."
            )
        )
        == "model2vec"
    )
    assert classify_unavailable_example_exception(RuntimeError("boom")) is None


def test_run_example_restores_talkpipe_env(monkeypatch: pytest.MonkeyPatch):
    """Examples should not leak TALKPIPE_* environment changes into later examples."""
    monkeypatch.delenv("TALKPIPE_DOC_EXAMPLE_TEST", raising=False)

    success, exc = run_example(
        "memory-test",
        'import os\nos.environ["TALKPIPE_DOC_EXAMPLE_TEST"] = "set-in-example"',
    )

    assert success is True
    assert exc is None
    assert "TALKPIPE_DOC_EXAMPLE_TEST" not in os.environ


@pytest.fixture(autouse=True)
def _cleanup_doc_example_artifacts():
    """Remove artifacts created by doc examples (e.g. vector DBs) after each test."""
    yield
    for name in _DOC_EXAMPLE_ARTIFACTS:
        if not _is_safe_to_delete(_project_root, name):
            continue
        path = _project_root / name
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


def _safe_test_id(path: Path, line_num: int) -> str:
    """Produce a test ID safe for shell and IDE parsing (no brackets, colons, or dots in ID)."""
    # e.g. README.md:277 -> README_md-277, docs/foo.md:74 -> docs_foo_md-74
    stem = str(path).replace(".", "_").replace("/", "_").replace("\\", "_")
    return f"{stem}-{line_num}"


def _example_marks(config: pytest.Config, code: str) -> list[pytest.MarkDecorator]:
    """Mark examples that depend on unavailable external services."""
    requirements = detect_example_requirements(code)
    marks: list[object] = []

    availability = {
        "ollama": getattr(config, "is_ollama_available", False),
        "openai": getattr(config, "is_openai_available", False),
        "anthropic": getattr(config, "is_anthropic_available", False),
        "mongodb": getattr(config, "is_mongodb_available", False),
    }

    if "ollama" in requirements:
        marks.append(pytest.mark.requires_ollama)
        if not availability["ollama"]:
            marks.append(pytest.mark.skip(reason=unavailable_example_reason("ollama")))

    for requirement in sorted(requirements - {"ollama"}):
        if requirement in availability and not availability[requirement]:
            marks.append(pytest.mark.skip(reason=unavailable_example_reason(requirement)))

    return marks


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Collect doc examples at collection time so new examples are picked up after doc edits."""
    if "path" in metafunc.fixturenames and "line_num" in metafunc.fixturenames and "code" in metafunc.fixturenames:
        examples = extract_all_examples(_project_root)
        metafunc.parametrize(
            "path,line_num,code",
            [
                pytest.param(
                    path,
                    line_num,
                    code,
                    id=_safe_test_id(path, line_num),
                    marks=_example_marks(metafunc.config, code),
                )
                for path, line_num, code in examples
            ],
        )


def test_doc_example(path: Path, line_num: int, code: str) -> None:
    """Run a documentation example, skipping only examples with unavailable dependencies."""
    location = f"{path}:{line_num}"
    success, exc = run_example(location, code)
    if not success and exc is not None:
        requirement = classify_unavailable_example_exception(exc)
        if requirement is not None:
            pytest.skip(unavailable_example_reason(requirement))
        raise exc from None
    assert success, f"Example {location} failed"

"""Extract and run Python code examples from markdown documentation."""

import re
import subprocess  # nosec B404 - Required to run extracted doc examples script
import sys
from pathlib import Path

# Directories to skip when scanning markdown files
SKIP_DIRS = {".github", ".pytest_cache", ".claude", ".venv", "venv", "__pycache__"}

_EXAMPLE_REQUIREMENT_PATTERNS = {
    "ollama": ("ollama",),
    "openai": ("openai",),
    "anthropic": ("anthropic",),
    "model2vec": ("model2vec",),
    "mongodb": ("mongoclient", "mongoinsert", "mongosearch", "mongodb"),
}

_UNAVAILABLE_REQUIREMENT_REASONS = {
    "ollama": "requires Ollama, but it is not available",
    "openai": "requires OpenAI credentials, but they are not available",
    "anthropic": "requires Anthropic credentials, but they are not available",
    "model2vec": "requires a cached/downloadable model2vec model, but it is not available",
    "mongodb": "requires MongoDB, but it is not available",
}


def find_markdown_files(root: Path) -> list[Path]:
    """Find all markdown files, excluding skip directories."""
    md_files = []
    for path in root.rglob("*.md"):
        if any(part in path.parts for part in SKIP_DIRS):
            continue
        md_files.append(path)
    return sorted(md_files)


def extract_python_blocks(content: str) -> list[tuple[int, str]]:
    """
    Extract Python code blocks from markdown content.
    Returns list of (line_number, code) tuples.
    """
    pattern = re.compile(
        r"^\s*```\s*python\s*\n(.*?)^\s*```\s*$",
        re.DOTALL | re.MULTILINE | re.IGNORECASE,
    )
    blocks = []
    for match in pattern.finditer(content):
        code = match.group(1)
        line_num = content[: match.start()].count("\n") + 1
        code = _normalize_indentation(code)
        if code.strip().startswith("# skip-extract"):
            continue
        if code.strip():
            blocks.append((line_num, code))
    return blocks


def detect_example_requirements(code: str) -> set[str]:
    """Detect external runtime requirements implied by an example."""
    lower = code.lower()
    requirements = set()
    for requirement, patterns in _EXAMPLE_REQUIREMENT_PATTERNS.items():
        if any(pattern in lower for pattern in patterns):
            requirements.add(requirement)
    return requirements


def classify_unavailable_example_exception(exc: BaseException) -> str | None:
    """Map availability-related example failures to a dependency name."""
    message = str(exc).lower()
    exc_name = type(exc).__name__.lower()

    if "ollama" in message and ("failed to connect" in message or "not available" in message):
        return "ollama"
    if "openai" in message and (
        "missing credentials" in message
        or "api_key" in message
        or "api key" in message
        or "authentication" in message
    ):
        return "openai"
    if "anthropic" in message and (
        "authentication" in message or "api key" in message or "auth" in message
    ):
        return "anthropic"
    if exc_name == "localentrynotfounderror" or (
        "snapshot folder" in message and "hub" in message
    ):
        return "model2vec"
    if (
        "mongodb" in message
        or "localhost:27017" in message
        or exc_name in {"serverselectiontimeouterror", "autoreconnect"}
    ):
        return "mongodb"
    return None


def unavailable_example_reason(requirement: str) -> str:
    """Return a human-readable skip reason for an unavailable dependency."""
    return _UNAVAILABLE_REQUIREMENT_REASONS.get(
        requirement, f"requires {requirement}, but it is not available"
    )


def _normalize_indentation(code: str) -> str:
    """Strip common leading indentation while preserving relative indentation."""
    lines = code.split("\n")
    if not lines:
        return code
    min_indent = None
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            if min_indent is None or indent < min_indent:
                min_indent = indent
    if min_indent is None or min_indent == 0:
        return code.rstrip()
    result = []
    for line in lines:
        if line.strip() and len(line) >= min_indent:
            result.append(line[min_indent:])
        else:
            result.append(line)
    return "\n".join(result).rstrip()


def extract_all_examples(root: Path) -> list[tuple[Path, int, str]]:
    """
    Extract all Python examples from markdown files.
    Returns list of (file_path, line_number, code) tuples.
    """
    examples = []
    for md_path in find_markdown_files(root):
        try:
            content = md_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line_num, code in extract_python_blocks(content):
            rel_path = md_path.relative_to(root) if md_path.is_relative_to(root) else md_path
            examples.append((rel_path, line_num, code))
    return examples


def run_example(location: str, code: str) -> tuple[bool, BaseException | None]:
    """
    Execute code in a fresh namespace.

    Returns (True, None) if successful, (False, exception) on failure.
    Patches io.Prompt to use echo with default input (avoids blocking on user input).
    """
    io_module = None
    original_prompt = None
    try:
        import talkpipe.pipe.io as _io

        io_module = _io
        original_prompt = _io.Prompt
        _io.Prompt = lambda *args, **kwargs: _io.echo(data="Hello, world!")
    except ImportError:
        pass

    namespace = {"__name__": "__main__", "__package__": None}
    try:
        exec(code, namespace)  # nosec B102 - Code from project's own markdown docs, trusted source
        return (True, None)
    except Exception as e:
        return (False, e)
    finally:
        if io_module is not None and original_prompt is not None:
            io_module.Prompt = original_prompt


def generate_runner_script(examples: list[tuple[Path, int, str]], output_path: Path) -> None:
    """Generate a Python script that runs each example with location printed."""
    lines = [
        '"""',
        "Auto-generated script: runs all Python examples extracted from markdown files.",
        "Each example prints its source location before execution.",
        "Pipelines using the Prompt source are fed 'Hello, world!' instead of waiting for input.",
        "Exits on first failure.",
        '"""',
        "",
        "import sys",
        "from pathlib import Path",
        "from talkpipe.app.doc_examples import (",
        "    classify_unavailable_example_exception,",
        "    run_example,",
        "    unavailable_example_reason,",
        ")",
        "",
        "",
        "def main():",
        "    examples = [",
    ]

    for rel_path, line_num, code in examples:
        escaped = code.replace("\\", "\\\\").replace('"""', r"\"\"\"")
        path_str = str(rel_path).replace("\\", "\\\\")
        lines.append(f'        ("{path_str}", {line_num}, """')
        lines.append(escaped)
        lines.append('"""),')

    fail_index_path = "Path(__file__).resolve().parent / '.extracted_examples_fail_index'"
    lines.extend(
        [
            "    ]",
            "",
            f"    fail_index_path = {fail_index_path}",
            "    start_index = 0",
            "    skipped = 0",
            "    try:",
            "        with open(fail_index_path) as f:",
            "            start_index = int(f.read().strip())",
            "    except (FileNotFoundError, ValueError):",
            "        pass",
            "",
            "    for i in range(start_index, len(examples)):",
            "        path, line_num, code = examples[i]",
            '        print(f"\\n--- {Path(path).name}:{line_num} ---")',
            '        print(f"  {path}")',
            "        success, exc = run_example(f\"{path}:{line_num}\", code)",
            "        if not success:",
            "            requirement = classify_unavailable_example_exception(exc) if exc is not None else None",
            "            if requirement is not None:",
            "                skipped += 1",
            '                print(f"  SKIP: {unavailable_example_reason(requirement)}")',
            "                continue",
            '            print(f"  ERROR: {exc}", file=sys.stderr)',
            "            fail_index_path.write_text(str(i))",
            '            print(f"\\nExiting: example {i + 1} failed. Run again to retry from here.", file=sys.stderr)',
            "            sys.exit(1)",
            "",
            "    fail_index_path.unlink(missing_ok=True)",
            '    print(f"\\n--- All {len(examples) - skipped} runnable examples passed ({skipped} skipped) ---")',
            "",
            "",
            'if __name__ == "__main__":',
            "    main()",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Extract examples, generate runner script, and run all examples. Returns exit code."""
    root = Path.cwd()
    output_path = root / "extracted_examples.py"

    print(f"Scanning markdown files under {root}...")
    examples = extract_all_examples(root)
    print(f"Found {len(examples)} Python examples")

    generate_runner_script(examples, output_path)
    print(f"Wrote {output_path}")

    result = subprocess.run(  # nosec B603 - output_path is generated by this module, not user input
        [sys.executable, str(output_path)], cwd=str(root)
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

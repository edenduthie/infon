"""
test_section_sequence.py

Verifies that paper/main.tex contains \\input commands for all required
sections in the correct order.
"""

import re
from pathlib import Path

MAIN_TEX = Path(__file__).parent.parent / "main.tex"

EXPECTED_SEQUENCE = [
    "00_abstract",
    "01_introduction",
    "02_background",
    "03_architecture",
    "04_implementation",
    "05_experiments",
    "06_discussion",
    "07_related_work",
    "08_conclusion",
    "09_reproducibility",
    "system_card",
]

# Matches \input{sections/foo} or \input{sections/foo.tex}
_INPUT_RE = re.compile(r"\\input\{sections/([^}]+?)(?:\.tex)?\}")


def _parse_input_sequence(tex_text: str) -> list[str]:
    """Return the list of section names referenced by \\input in document order."""
    return _INPUT_RE.findall(tex_text)


def test_all_sections_present() -> None:
    tex_text = MAIN_TEX.read_text(encoding="utf-8")
    found = _parse_input_sequence(tex_text)
    for section in EXPECTED_SEQUENCE:
        assert section in found, (
            f"Section '{section}' not found in \\input commands of main.tex. "
            f"Found: {found}"
        )


def test_section_order() -> None:
    tex_text = MAIN_TEX.read_text(encoding="utf-8")
    found = _parse_input_sequence(tex_text)

    # Build index map for sections that appear in the expected list
    positions = {name: found.index(name) for name in EXPECTED_SEQUENCE if name in found}

    for i in range(len(EXPECTED_SEQUENCE) - 1):
        earlier = EXPECTED_SEQUENCE[i]
        later = EXPECTED_SEQUENCE[i + 1]
        assert earlier in positions, f"Section '{earlier}' missing from main.tex"
        assert later in positions, f"Section '{later}' missing from main.tex"
        assert positions[earlier] < positions[later], (
            f"Section '{earlier}' must appear before '{later}' in main.tex, "
            f"but found positions: {earlier}={positions[earlier]}, "
            f"{later}={positions[later]}"
        )


def test_no_unexpected_sections() -> None:
    """Warn if extra \\input{sections/...} entries appear outside the expected list."""
    tex_text = MAIN_TEX.read_text(encoding="utf-8")
    found = _parse_input_sequence(tex_text)
    unexpected = [name for name in found if name not in EXPECTED_SEQUENCE]
    assert unexpected == [], (
        f"Unexpected section(s) found in main.tex \\input commands: {unexpected}"
    )

#!/usr/bin/env bash
# build.sh — Create an arXiv-ready tarball: paper/arxiv-submission/paper.tar.gz
#
# What it includes:
#   - main.tex
#   - All .tex files referenced via \input{...} in main.tex
#   - references.bbl (if it exists — arXiv uses .bbl, not .bib)
#   - All figure files referenced via \includegraphics{...} in main.tex
#   - 00README.txt (generated if missing)
#
# What it excludes:
#   - .aux, .log, .out, .bib, .synctex.gz files
#   - Hidden files (starting with .)
#
# Checks:
#   - Reports total tarball size
#   - Warns if size exceeds 50 MB
#
# Usage (from repo root):
#   bash paper/arxiv-submission/build.sh [--paper-dir PATH] [--out PATH]
#
# Options:
#   --paper-dir PATH   Path to the paper/ directory (default: paper/)
#   --out PATH         Output tarball path (default: paper/arxiv-submission/paper.tar.gz)

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Default paper dir is one level up from this script (i.e., paper/)
PAPER_DIR="${SCRIPT_DIR%/arxiv-submission}"
# If script is called with absolute path reconstruction fails safely:
if [[ ! -f "${PAPER_DIR}/main.tex" ]]; then
    # Try relative to cwd
    PAPER_DIR="$(pwd)/paper"
fi

OUT_TARBALL="${PAPER_DIR}/arxiv-submission/paper.tar.gz"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --paper-dir)
            PAPER_DIR="$(cd "$2" && pwd)"
            shift 2
            ;;
        --out)
            OUT_TARBALL="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: bash build.sh [--paper-dir PATH] [--out PATH]" >&2
            exit 1
            ;;
    esac
done

MAIN_TEX="${PAPER_DIR}/main.tex"

if [[ ! -f "${MAIN_TEX}" ]]; then
    echo "ERROR: main.tex not found at ${MAIN_TEX}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Parse main.tex for included files
# ---------------------------------------------------------------------------

echo "==> Parsing ${MAIN_TEX} for included files ..."

# Extract \input{...} references (strips optional .tex suffix for lookup)
INPUT_FILES=()
while IFS= read -r line; do
    # Match \input{some/path} or \input{some/path.tex}
    if [[ "$line" =~ \\input\{([^}]+)\} ]]; then
        ref="${BASH_REMATCH[1]}"
        # Append .tex if not already present
        if [[ "$ref" != *.tex ]]; then
            ref="${ref}.tex"
        fi
        INPUT_FILES+=("$ref")
    fi
done < "${MAIN_TEX}"

# Extract \includegraphics{...} references (may lack extension — try common ones)
GRAPHICS_FILES=()
while IFS= read -r line; do
    if [[ "$line" =~ \\includegraphics(\[[^]]*\])?\{([^}]+)\} ]]; then
        ref="${BASH_REMATCH[2]}"
        GRAPHICS_FILES+=("$ref")
    fi
done < "${MAIN_TEX}"

# ---------------------------------------------------------------------------
# Build file list
# ---------------------------------------------------------------------------

# Work relative to PAPER_DIR
INCLUDE_FILES=()

# Always include main.tex
INCLUDE_FILES+=("main.tex")

# Input .tex files
for rel in "${INPUT_FILES[@]}"; do
    abs="${PAPER_DIR}/${rel}"
    if [[ -f "$abs" ]]; then
        INCLUDE_FILES+=("$rel")
    else
        echo "    WARNING: \\input file not found: ${rel}" >&2
    fi
done

# references.bbl (if it exists)
if [[ -f "${PAPER_DIR}/references.bbl" ]]; then
    INCLUDE_FILES+=("references.bbl")
    echo "    Including references.bbl"
else
    echo "    NOTE: references.bbl not found — skipping (run bibtex first if needed)"
fi

# Figure files referenced by \includegraphics
for rel in "${GRAPHICS_FILES[@]}"; do
    abs="${PAPER_DIR}/${rel}"
    if [[ -f "$abs" ]]; then
        INCLUDE_FILES+=("$rel")
    else
        # Try common extensions: .pdf .png .jpg .eps
        found=false
        for ext in .pdf .png .jpg .jpeg .eps; do
            if [[ -f "${abs}${ext}" ]]; then
                INCLUDE_FILES+=("${rel}${ext}")
                found=true
                break
            fi
        done
        if [[ "$found" == "false" ]]; then
            echo "    WARNING: \\includegraphics file not found: ${rel} (tried common extensions)" >&2
        fi
    fi
done

# 00README.txt — generate if missing
README_TXT="${PAPER_DIR}/arxiv-submission/00README.txt"
if [[ ! -f "${README_TXT}" ]]; then
    echo "    Generating 00README.txt ..."
    mkdir -p "${PAPER_DIR}/arxiv-submission"
    cat > "${README_TXT}" <<'EOF'
This archive contains the LaTeX source for:
  "Cognition: A Knowledge Graph Reasoner with Calibrated Uncertainty"

To compile:
  pdflatex main.tex
  bibtex main
  pdflatex main.tex
  pdflatex main.tex

The main entry point is main.tex.
EOF
fi
# Include 00README.txt from the tarball staging perspective
# We'll copy it into the staging dir so it appears at the top level

# ---------------------------------------------------------------------------
# Exclusion filter: never include these extensions or hidden files
# ---------------------------------------------------------------------------

EXCLUDED_EXTENSIONS=(".aux" ".log" ".out" ".bib" ".synctex.gz" ".blg" ".fdb_latexmk" ".fls")

_is_excluded() {
    local filepath="$1"
    local basename
    basename="$(basename "$filepath")"

    # Hidden files
    if [[ "$basename" == .* ]]; then
        return 0
    fi

    # Excluded extensions
    for ext in "${EXCLUDED_EXTENSIONS[@]}"; do
        if [[ "$filepath" == *"$ext" ]]; then
            return 0
        fi
    done

    return 1
}

# ---------------------------------------------------------------------------
# Stage files and create tarball
# ---------------------------------------------------------------------------

STAGING_DIR="$(mktemp -d)"
trap 'rm -rf "${STAGING_DIR}"' EXIT

echo ""
echo "==> Staging files for tarball ..."

FINAL_FILES=()
for rel in "${INCLUDE_FILES[@]}"; do
    if _is_excluded "$rel"; then
        echo "    EXCLUDED (filter): ${rel}"
        continue
    fi
    src="${PAPER_DIR}/${rel}"
    dst="${STAGING_DIR}/${rel}"
    mkdir -p "$(dirname "$dst")"
    if [[ -f "$src" ]]; then
        cp "$src" "$dst"
        FINAL_FILES+=("$rel")
        echo "    + ${rel}"
    fi
done

# Also include 00README.txt at top level of staging dir
if [[ -f "${README_TXT}" ]]; then
    cp "${README_TXT}" "${STAGING_DIR}/00README.txt"
    FINAL_FILES+=("00README.txt")
    echo "    + 00README.txt"
fi

# ---------------------------------------------------------------------------
# Create tarball
# ---------------------------------------------------------------------------

echo ""
echo "==> Creating tarball: ${OUT_TARBALL} ..."

mkdir -p "$(dirname "${OUT_TARBALL}")"
tar -czf "${OUT_TARBALL}" -C "${STAGING_DIR}" .

# ---------------------------------------------------------------------------
# Size check
# ---------------------------------------------------------------------------

TARBALL_SIZE_BYTES=$(stat -c%s "${OUT_TARBALL}" 2>/dev/null || stat -f%z "${OUT_TARBALL}")
TARBALL_SIZE_MB=$(awk "BEGIN {printf \"%.2f\", ${TARBALL_SIZE_BYTES}/1048576}")
MAX_SIZE_MB=50

echo ""
echo "==> Tarball contents: ${#FINAL_FILES[@]} file(s)"
echo "==> Tarball size: ${TARBALL_SIZE_MB} MB (${TARBALL_SIZE_BYTES} bytes)"

if awk "BEGIN {exit !(${TARBALL_SIZE_MB} > ${MAX_SIZE_MB})}"; then
    echo "ERROR: Tarball exceeds ${MAX_SIZE_MB} MB arXiv limit!" >&2
    exit 1
else
    echo "==> Size check PASSED (<= ${MAX_SIZE_MB} MB)"
fi

echo ""
echo "==> arXiv tarball created: ${OUT_TARBALL}"

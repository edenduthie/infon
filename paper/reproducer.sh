#!/usr/bin/env bash
# reproducer.sh — end-to-end reproduction script for the Cognition paper.
#
# Stages:
#   1. Install the reference_v4 package (pip install -e 'reference_v4/[study]')
#   2. Run Epic 03 evaluation matrix (reference_v4/experiments/benchmark_eval.py if present)
#   3. Generate H1/H2/H3 panel JSONs (reference_v4/experiments/panels.py if present)
#   4. Regenerate all figures (paper/figures/*.py with --input/--output args)
#   5. make lint
#   6. make pdf (skipped with a warning if latexmk is not available)
#
# Idempotent: each stage writes a sentinel file to .reproducer_checkpoints/
# so that re-running skips already-completed stages.
#
# Usage:
#   bash paper/reproducer.sh [--dry-run]
#
# Options:
#   --dry-run   Print all commands that would run, then exit 0.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_DIR="${REPO_ROOT}/.reproducer_checkpoints"
PAPER_DIR="${REPO_ROOT}/paper"
FIGURES_DIR="${PAPER_DIR}/figures"
EXPERIMENTS_DIR="${REPO_ROOT}/reference_v4/experiments"

DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --dry-run)
            DRY_RUN=true
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            echo "Usage: bash paper/reproducer.sh [--dry-run]" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STAGE_NAMES=(
    "stage_1: install reference_v4 package"
    "stage_2: run Epic 03 evaluation matrix"
    "stage_3: generate H1/H2/H3 panel JSONs"
    "stage_4: regenerate all figures"
    "stage_5: make lint"
    "stage_6: make pdf"
)

_sentinel() {
    echo "${CHECKPOINT_DIR}/.stage_${1}_done"
}

_stage_done() {
    local stage_num="$1"
    [[ -f "$(_sentinel "$stage_num")" ]]
}

_mark_done() {
    local stage_num="$1"
    mkdir -p "${CHECKPOINT_DIR}"
    touch "$(_sentinel "$stage_num")"
}

_run() {
    # Print the command, then execute it (or skip if dry-run).
    echo "+ $*"
    if [[ "$DRY_RUN" == "false" ]]; then
        "$@"
    fi
}

_run_shell() {
    # Run a shell command string (for pipelines, cd, etc.)
    local cmd="$1"
    echo "+ $cmd"
    if [[ "$DRY_RUN" == "false" ]]; then
        bash -c "$cmd"
    fi
}

# ---------------------------------------------------------------------------
# Dry-run: print all stage names and exit 0
# ---------------------------------------------------------------------------

if [[ "$DRY_RUN" == "true" ]]; then
    echo "=== dry-run mode: listing all stages ==="
    for name in "${STAGE_NAMES[@]}"; do
        echo "  $name"
    done
    echo ""
    echo "Stage commands:"
    echo ""

    echo "--- stage_1: install reference_v4 package ---"
    echo "+ pip install -e '${REPO_ROOT}/reference_v4/[study]'"
    echo ""

    echo "--- stage_2: run Epic 03 evaluation matrix ---"
    if [[ -f "${EXPERIMENTS_DIR}/benchmark_eval.py" ]]; then
        echo "+ python ${EXPERIMENTS_DIR}/benchmark_eval.py"
    else
        echo "(benchmark_eval.py not found — stage will be skipped)"
    fi
    echo ""

    echo "--- stage_3: generate H1/H2/H3 panel JSONs ---"
    if [[ -f "${EXPERIMENTS_DIR}/panels.py" ]]; then
        echo "+ python ${EXPERIMENTS_DIR}/panels.py"
    else
        echo "(panels.py not found — stage will be skipped)"
    fi
    echo ""

    echo "--- stage_4: regenerate all figures ---"
    for fig_script in "${FIGURES_DIR}"/*.py; do
        [[ -f "$fig_script" ]] || continue
        stem="$(basename "${fig_script%.py}")"
        echo "+ python ${fig_script} --output ${PAPER_DIR}/figures/${stem}.pdf"
    done
    echo ""

    echo "--- stage_5: make lint ---"
    echo "+ make lint (from ${REPO_ROOT})"
    echo ""

    echo "--- stage_6: make pdf ---"
    if command -v latexmk &>/dev/null; then
        echo "+ make pdf (from ${REPO_ROOT})"
    else
        echo "(latexmk not available — stage will be skipped with a warning)"
    fi

    exit 0
fi

# ---------------------------------------------------------------------------
# Stage 1: Install reference_v4 package
# ---------------------------------------------------------------------------

if _stage_done 1; then
    echo "==> stage_1 already done (checkpoint found), skipping."
else
    echo "==> stage_1: install reference_v4 package"
    _run pip install -e "${REPO_ROOT}/reference_v4/[study]"
    _mark_done 1
    echo "==> stage_1 complete."
fi

# ---------------------------------------------------------------------------
# Stage 2: Run Epic 03 evaluation matrix
# ---------------------------------------------------------------------------

if _stage_done 2; then
    echo "==> stage_2 already done (checkpoint found), skipping."
else
    echo "==> stage_2: run Epic 03 evaluation matrix"
    if [[ -f "${EXPERIMENTS_DIR}/benchmark_eval.py" ]]; then
        _run python "${EXPERIMENTS_DIR}/benchmark_eval.py"
    else
        echo "    WARNING: ${EXPERIMENTS_DIR}/benchmark_eval.py not found — skipping."
    fi
    _mark_done 2
    echo "==> stage_2 complete."
fi

# ---------------------------------------------------------------------------
# Stage 3: Generate H1/H2/H3 panel JSONs
# ---------------------------------------------------------------------------

if _stage_done 3; then
    echo "==> stage_3 already done (checkpoint found), skipping."
else
    echo "==> stage_3: generate H1/H2/H3 panel JSONs"
    if [[ -f "${EXPERIMENTS_DIR}/panels.py" ]]; then
        _run python "${EXPERIMENTS_DIR}/panels.py"
    else
        echo "    WARNING: ${EXPERIMENTS_DIR}/panels.py not found — skipping."
    fi
    _mark_done 3
    echo "==> stage_3 complete."
fi

# ---------------------------------------------------------------------------
# Stage 4: Regenerate all figures
# ---------------------------------------------------------------------------

if _stage_done 4; then
    echo "==> stage_4 already done (checkpoint found), skipping."
else
    echo "==> stage_4: regenerate all figures"
    for fig_script in "${FIGURES_DIR}"/*.py; do
        [[ -f "$fig_script" ]] || continue
        stem="$(basename "${fig_script%.py}")"
        output_pdf="${FIGURES_DIR}/${stem}.pdf"
        echo "    Generating ${stem}.pdf ..."
        _run python "${fig_script}" --output "${output_pdf}"
    done
    _mark_done 4
    echo "==> stage_4 complete."
fi

# ---------------------------------------------------------------------------
# Stage 5: make lint
# ---------------------------------------------------------------------------

if _stage_done 5; then
    echo "==> stage_5 already done (checkpoint found), skipping."
else
    echo "==> stage_5: make lint"
    _run_shell "cd '${REPO_ROOT}' && make lint"
    _mark_done 5
    echo "==> stage_5 complete."
fi

# ---------------------------------------------------------------------------
# Stage 6: make pdf (local only — skip if latexmk not available)
# ---------------------------------------------------------------------------

if _stage_done 6; then
    echo "==> stage_6 already done (checkpoint found), skipping."
else
    echo "==> stage_6: make pdf"
    if command -v latexmk &>/dev/null; then
        _run_shell "cd '${REPO_ROOT}' && make pdf"
        _mark_done 6
        echo "==> stage_6 complete."
    else
        echo "    WARNING: latexmk not found — skipping make pdf."
        _mark_done 6
    fi
fi

echo ""
echo "=== Reproducer finished successfully ==="

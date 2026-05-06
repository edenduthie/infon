# Cognition: A Knowledge Graph Reasoner with Calibrated Uncertainty

LaTeX source for the Cognition system paper targeting arXiv submission.

## Prerequisites

- `latexmk` and `pdflatex` (e.g. via `texlive-latex-extra`, `texlive-fonts-recommended`)
- Python 3.11+

Install on Ubuntu/Debian:

```bash
apt-get install -y latexmk texlive-latex-extra texlive-fonts-recommended
```

## Build Instructions

All commands are run from the **repo root** (`/home/ubuntu/infon`):

```bash
# Compile the paper to PDF
make pdf

# Run the numbers audit lint check
make lint

# Build the arXiv submission tarball
make arxiv-tarball

# Clean LaTeX build artefacts
make clean
```

The compiled PDF is written to `paper/main.pdf`.

## Reproduce

```bash
bash paper/reproducer.sh
```

## License

[CC BY 4.0](LICENSE)

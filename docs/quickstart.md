# Quickstart: running infon from source with Claude Code

This is the fastest path from a fresh clone to having Claude Code use infon
as an MCP knowledge base on one of your projects. It assumes you have
Python 3.11+ and `git`. No PyPI publish needed — you build the wheel
locally and install it from disk.

For background on what infon is and why you'd use it, see
[Concepts](concepts.md). For the full CLI reference, see
[CLI Reference](cli.md). For MCP server details, see
[MCP Server](mcp.md).

---

## 1. Build the wheel

Clone the repo and build a wheel from source. This step is one-time per
infon version.

```bash
git clone https://github.com/edenduthie/infon
cd infon

# Disposable venv just for the build tools — keeps your system Python clean.
python3 -m venv /tmp/infon-build-tools
/tmp/infon-build-tools/bin/pip install build
/tmp/infon-build-tools/bin/python -m build

ls dist/
# infon-0.1.0-py3-none-any.whl
# infon-0.1.0.tar.gz
```

If you only have one machine (this dev box and the box you'll run Claude
Code on are the same), you can skip the next note. If you build on one
machine and run on another, copy `dist/infon-0.1.0-py3-none-any.whl` over
— that one file is the only artifact you need.

## 2. Install into a long-lived venv

The wheel pulls in torch + transformers + duckdb + tree-sitter (~1 GB
total). Don't reinstall this per project — keep one venv and reuse it.

```bash
python3 -m venv ~/.local/share/infon-venv
~/.local/share/infon-venv/bin/pip install /path/to/infon-0.1.0-py3-none-any.whl

# Optional: symlink so `infon` works from any shell.
mkdir -p ~/.local/bin
ln -s ~/.local/share/infon-venv/bin/infon ~/.local/bin/infon

# Sanity check:
infon --version          # → infon, version 0.1.0
infon --help             # lists init, ingest, search, stats, serve
```

If `~/.local/bin` isn't on your PATH, either add it to your shell rc or
just call `~/.local/share/infon-venv/bin/infon` directly everywhere
below.

## 3. Initialize the kb on the project Claude Code will work on

`cd` into the project root — the directory that holds the source files
you want indexed.

```bash
cd ~/code/my-project
infon init                  # full default: discovery + text extraction
```

`init` creates four things:

- `.infon/kb.ddb` — DuckDB knowledge base.
- `.infon/schema.json` — the discovered or static anchor schema.
- `.mcp.json` in the project root — what Claude Code reads to find the
  MCP server.
- An entry in `.gitignore` for `.infon/` so you don't commit the kb.

### Time cost of the default

By default `init` runs Phase-6 schema discovery + Python docstring text
extraction + markdown text extraction. On CPU this takes **10–30+ minutes
on a moderate (~100-file) repo** because every line in the corpus and
every docstring sentence goes through the SPLADE encoder.

If you don't need conceptual queries (e.g. "how does this module work")
and just want structural search ("what calls FooBar"), use the fast path:

```bash
infon init --shallow        # ~10 seconds, AST-only, no SPLADE-driven extraction
```

You can always re-run `infon init --shallow` once for fast iteration and
later re-run `infon init` (without `--shallow`) when you want the
deep extraction.

## 4. Verify `.mcp.json` points at your venv binary

Look at the generated config:

```bash
cat .mcp.json
```

You want it to look like this (`command` is an absolute path into your
infon venv):

```json
{
  "mcpServers": {
    "infon": {
      "type": "stdio",
      "command": "/home/you/.local/share/infon-venv/bin/infon",
      "args": ["serve"]
    }
  }
}
```

If instead you see this — `init` thought you were running outside a venv:

```json
{
  "mcpServers": {
    "infon": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "infon", "infon", "serve"]
    }
  }
}
```

That form tries to fetch `infon` from PyPI on every launch and will fail
since you haven't published. Either:

- Edit `.mcp.json` by hand and replace `"command"` with the absolute path
  to your venv `infon` binary (and replace `"args"` with `["serve"]`); OR
- Re-run `infon init` from inside an activated venv:
  `source ~/.local/share/infon-venv/bin/activate && infon init`. Venv
  detection picks up the active environment and writes the binary-path
  form. (See the [installation guide](installation.md) for the detection
  logic.)

## 5. Smoke-test from the CLI before bringing in Claude Code

Confirm the kb is populated and search returns useful results:

```bash
infon stats
# Knowledge Base Statistics
# ==================================================
# Infons:      5474
# Edges:       9529
# Constraints: 5474
# Documents:   27

infon search "what calls SomeClass"
# Found 5 results:
#
# 1. UserService --[calls]--> SomeClass  (score=0.469)
#    src/services/user.py:42 (call)
#    next: ...
```

If `search` returns "No results found" for a symbol you know is in the
code, something didn't ingest correctly. Re-run `infon stats` to confirm
the infon count is non-zero. If it's zero, the AST extractor probably
hit an unhandled error during init — re-run `infon init` and watch the
output.

## 6. Open Claude Code in that project

```bash
cd ~/code/my-project
claude
```

Claude Code reads `.mcp.json` on startup, spawns `infon serve` as a
subprocess, and connects over stdio. To confirm it's wired up, type
`/mcp` inside Claude Code — `infon` should appear in the list.

The agent now has three tools available:

| Tool                  | Use when …                                                                                                                  |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `search`              | Open-ended natural-language questions: "where does X get configured", "what calls Y", "how does Z work".                    |
| `query_ast`           | You know the exact symbol: "all callers of `InfonStore`", "everything `service.py` imports", filter by predicate if needed. |
| `store_observation`   | The agent records a finding it wants to remember between sessions ("decided to use FastAPI here because…").                  |

You don't call these directly. The agent decides when to use them based
on what you're asking. To force-test, prompt Claude with something like
"Use the infon search tool to find what calls `MyClass` in this repo."

## Updating the kb

If you've changed a lot of code, refresh:

```bash
infon ingest          # full pipeline — slow but comprehensive
infon ingest --shallow # AST-only — fast, structural changes only
```

There's no auto-watch yet. You re-run `ingest` manually when you want
the kb to reflect recent changes.

## Gotchas

1. **Only one process can write the kb at a time.** Claude Code's
   `infon serve` holds a writer lock. If you try to run `infon search`
   from another terminal while Claude Code is open, you'll see
   `ConcurrentWriteError`. Close Claude Code (or the MCP session) first.

2. **The kb is per-project.** Each project gets its own `.infon/`
   directory. Switching projects means running `infon init` in the new
   project root.

3. **First run downloads ~500 MB.** The SPLADE model
   (`naver/splade-cocondenser-ensembledistil`) is fetched from
   HuggingFace Hub on first use, cached under `~/.cache/huggingface`.
   After that, init/serve don't need network access.

4. **Set `HF_TOKEN`** if you have one — without it you get an
   unauthenticated-rate-limit warning on every model load. It's not
   blocking, just noisy.

5. **CPU is slow for the deep default.** If init is taking forever and
   you don't need conceptual queries, kill it (Ctrl-C) and re-run with
   `--shallow`. Most "find this code" questions only need the structural
   AST infons that the shallow path produces.

6. **The kb is local only.** It lives in `.infon/kb.ddb` and is
   gitignored by default. Each developer on a team builds their own kb
   independently. Sharing kbs across machines is not supported in v0.1.x.

## What's next

- Browse the [CLI Reference](cli.md) for all commands and flags.
- Read [Concepts](concepts.md) for how anchors, infons, and grounding
  work.
- See [MCP Server](mcp.md) for the JSON-RPC tool schemas the agent
  consumes.
- Read [Schema Discovery](schema-discovery.md) for what the discovery
  pass actually produces and how to override it with a hand-tuned schema.

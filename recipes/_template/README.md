# TODO: recipe title

> Template. Copy this folder to `recipes/<name>`, then replace every TODO.

One-line description: trace a real **`<provider>`** call with the Bir SDK using
`bir.integrations.<name>.<entry_point>`.

## Key

TODO: which env var, and where to get a (preferably free) key.

## Run it

```bash
# Offline smoke — no key, no network (what CI runs):
uv run python main.py --smoke

# Real run:
export TODO_API_KEY="..."
uv run python main.py --prompt "..."
```

## Checklist before you delete this note

- [ ] `main.py` implements the real call + a same-shape in-file fake for `--smoke`
- [ ] Real provider library imported **lazily** (non-smoke branch only)
- [ ] Key read from env var only; `.env.example` lists it
- [ ] `pyproject.toml` renamed, provider dep added, `bir-sdk==0.3.0` pinned
- [ ] `uv run python main.py --smoke` passes locally
- [ ] Real path verified once with your own key
- [ ] Added to the root README table; removed from the backlog

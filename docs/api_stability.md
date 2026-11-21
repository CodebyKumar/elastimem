# API stability

Elastimem is pre-1.0. From 0.3.0 on:

- **Stable**: the names below. A breaking change to them needs a new
  minor version before 1.0 (a major one after), a `CHANGELOG.md` entry,
  and an upgrade section in
  [agent_integration_prompt.md](agent_integration_prompt.md). Adding a
  method, a field or a setting is not breaking.
- **Internal**: everything else, including every submodule
  (`elastimem.retrieval.search`, `elastimem.tiers`, ...), names starting with `_`,
  and the database schema (which migrates automatically). These can change
  in any release.

0.3.0 itself broke the 0.2 API (see the CHANGELOG). 0.2 was never
published, so no deprecation period was kept.

## Stable names

```python
elastimem.open, elastimem.Elastimem, elastimem.ElastimemConfig, elastimem.Tier,
elastimem.Status, elastimem.Budgets, elastimem.Context, elastimem.Fact,
elastimem.Hit, elastimem.Explanation, elastimem.__version__
```

`Elastimem` members: `context`, `record`, `pause`, `search`, `explain`,
`remember`, `forget`, `facts`, `history`, `learn`, `lessons`, `topics`,
`rejected`, `sessions`, `new_session`, `end_session`, `resume`, `status`,
`config`, `configure`, `report_pressure`, `flush`, `stats`, `close`, and the
context-manager protocol. Constructor arguments: `path`, `llm`, `embedder`,
`tokenizer`, `on_tier`, `config`, and every `ElastimemConfig` field.

Fields of `Context`, `Status`, `Budgets`, `Hit`, `Fact` and `Explanation`
are Stable, as are the section names in `Context.sections`.

## Not promised

- **Rankings and scores.** The equation's weights are tuned against
  measurements and may change in any release. Rely on order, not values.
- **The text of `Status.reasons`** and rendered headings of `str(Context)`.
- **`Explanation.items[].parts` keys** beyond the five current terms.
- **Tier thresholds** (RAM sizes, pressure rules, time budget defaults).

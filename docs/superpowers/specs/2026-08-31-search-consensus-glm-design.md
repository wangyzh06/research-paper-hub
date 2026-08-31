# Search, Consensus, and GLM Migration Design

## Status

Approved direction: implement comprehensive option A. GLM becomes the default LLM provider, while other configured providers remain available as non-default choices.

## Goals

- Make Chinese and English literature searches fail-safe, reproducible, and genuinely multi-source.
- Preserve relevant papers from OpenAlex, arXiv, Crossref, Semantic Scholar, and PubMed instead of allowing one source to fill the result cap.
- Expose source failures separately from legitimate zero-result searches.
- Replace the current single-prompt consensus guess with auditable per-paper evidence extraction and deterministic aggregation.
- Default all LLM-assisted operations to a supported GLM model without removing the existing provider choices.
- Add offline regression tests for every confirmed defect before changing production behavior.

## Non-goals

- Rebuilding the Streamlit application or changing its navigation.
- Implementing a formal statistical meta-analysis across incompatible studies.
- Claiming full-text evidence when only titles and abstracts are available.
- Adding paid literature providers or requiring a new external service.
- Migrating the existing SQLite schema in the first repair pass.

## Architecture

The repair introduces three explicit boundaries inside the existing Python application:

1. **Query planning** converts one user query into a validated bilingual query plan. AI translation and expansion are optional enhancements; a failed AI call never becomes a search term.
2. **Source retrieval and fusion** normalizes each provider response into one canonical paper shape, records per-source diagnostics, deduplicates identities, and merges ranked lists using reciprocal-rank fusion before truncation.
3. **Consensus evidence processing** evaluates selected papers one at a time into structured evidence records, computes a deterministic score from those records, and lets the LLM write only the narrative explanation.

The existing `SearchEngine.search()` list-returning interface remains available for current call sites. A new diagnostic interface returns papers plus query and source diagnostics for the search and consensus pages.

## GLM Migration

- Add a single `DEFAULT_LLM_PROVIDER = "glm"` constant and use it in the engine and Streamlit session defaults.
- Keep DeepSeek, OpenAI, Claude, Qwen, GLM, and Ollama in the settings page as optional providers.
- Search-time translation and expansion receive the current provider, model, and API key from the Streamlit session instead of constructing an implicit DeepSeek client.
- Cached search keys include provider, model, and a non-reversible API-key fingerprint. The raw API key is excluded from cache hashing and display.
- Strict LLM calls raise a typed error. Existing user-facing calls may convert that error into a visible message, but search planning must fall back to deterministic queries.
- Set the GLM default model identifier to `glm-4.7-flash`, which the current official API documents for `https://open.bigmodel.cn/api/paas/v4/chat/completions`. Remove retired identifiers from the GLM selector while leaving the other provider selectors intact.

## Canonical Paper and Source Results

Every source adapter produces the same dictionary fields:

- `id`, `source`, `title`, `abstract`, `year`, `citation_count`, `venue`
- `authors` as `[{"name": str}]`
- normalized `doi`, `url`
- optional source identifiers: `openalex_id`, `semantic_id`, `arxiv_id`, `pmid`
- internal ranking metadata: query variant, source rank, and fusion score

Each source call returns a `SourceBatch` containing `papers`, `total`, `source`, and an optional structured error with category, HTTP status, and retryability. A source timeout or rate limit therefore cannot masquerade as zero matches.

DOIs are normalized by removing URL and `doi:` prefixes, trimming whitespace, and case-folding. Records without a DOI use source identifiers, then a normalized title-plus-year key. Duplicate records merge richer metadata rather than dropping whichever record appears later.

OpenAlex inverted abstracts are reconstructed from every token position, including repeated words. Crossref dates prefer published/issued dates over record-creation dates. Semantic Scholar maps `externalIds` and `publicationVenue` correctly. PubMed participates in automatic searches and keeps PMID separate from DOI.

## Query Planning

The query plan always contains the trimmed original query. For Chinese input it may also contain a validated English translation; for all input it may contain a small number of sanitized academic synonyms.

Validation rejects empty strings, AI error markers, explanations, numbered prose, Markdown, and overlong variants. Failed translation or expansion leaves the original query intact. Acronyms, quoted phrases, DOIs, and mixed Chinese-English technical terms are preserved.

Exact identifiers and likely full titles take dedicated paths before broad topic search:

- DOI: provider identifier lookup where supported.
- arXiv/PMID/OpenAlex identifier: direct lookup.
- Long title-like query: title-biased search plus broad fallback.
- Topic query: bilingual multi-source search.

The query plan, including fallbacks and rejected variants, is recorded in diagnostics.

## Multi-source Retrieval and Ranking

Source and query-variant calls run concurrently with bounded per-source request counts. Pagination uses each provider's supported cursor or offset instead of repeating page one.

Each provider contributes a ranked list. Reciprocal-rank fusion combines those lists using `1 / (60 + rank)` with a modest weight for the original query. Exact normalized title matches and DOI matches receive deterministic bonuses. Citation count is displayed and may break close ties, but cannot replace relevance.

Fusion happens in this order:

1. Normalize each source response.
2. Record diagnostics.
3. Deduplicate and merge metadata.
4. Calculate fusion and lexical bilingual relevance scores.
5. Sort the full candidate pool.
6. Apply the requested result limit.

This ordering prevents OpenAlex from consuming the cap before other sources are considered. Partial results remain usable, with a warning listing failed sources. Total failure produces an actionable error rather than “no papers found.”

## Search UI

The search page continues to show paper cards and existing analyses. It additionally shows a compact source summary: returned count per provider, failed providers, query translation status, and whether the result is partial. Empty results distinguish “no matches” from “all providers failed.”

Changing the query resets pagination. Cached partial or failed searches have a shorter lifetime than successful searches and can be explicitly retried.

## Consensus Evidence Pipeline

The consensus page reuses one diagnostic search instead of running the same page twice. It deduplicates candidates, ranks the complete pool, and selects a bounded evidence set with source and study-type diversity.

For each selected paper, GLM receives the research question plus that paper's title and available abstract. It returns one JSON evidence record with:

- eligibility and exclusion reason
- study type: original experimental, observational, simulation, theory, systematic review, narrative review, commentary, or unknown
- population/system, intervention or exposure, comparator, and outcome
- stance: support, oppose, mixed, uncertain, or not applicable
- concise claim and effect direction
- reported sample size and statistical evidence, or explicit `not reported`
- methodological quality: high, medium, low, or unknown
- directness to the question and extraction confidence from 0 to 1
- an evidence excerpt traceable to the supplied abstract
- limitation flags, including abstract-only evidence

JSON parsing is strict. Invalid or failed extractions are retried once in a smaller repair prompt, then marked unavailable rather than silently treated as neutral. Reviews and commentaries inform the narrative but do not count as independent directional studies. Suspected duplicate cohorts are flagged and downweighted.

## Deterministic Consensus Calculation

Only eligible directional studies contribute to the score. Each paper receives a transparent weight:

`weight = study_type_weight * quality_weight * directness * extraction_confidence * independence_factor`

Support adds its weight to `S`; oppose adds to `O`; mixed evidence contributes half to each. Uncertain and not-applicable evidence contributes to coverage reporting but not direction.

When `S + O == 0`, the score is unavailable and the UI reports insufficient evidence. Otherwise:

- `directional_agreement = max(S, O) / (S + O)`
- `evidence_certainty = weighted_quality * min(1, effective_independent_studies / 8)`
- `confidence_adjusted_consensus = round(100 * directional_agreement * evidence_certainty)`

The UI presents all three concepts separately: dominant direction, directional agreement, and confidence-adjusted consensus. It also shows raw and weighted support/oppose/mixed/uncertain counts. The narrative model receives these fixed values and cannot alter them.

The report must state that abstract-only analysis cannot establish unreported sample sizes, effect sizes, p values, confidence intervals, or risk-of-bias details.

## Consensus UI

The existing report layout is retained but the score card is fed from the deterministic result object, never parsed from Markdown. Below it, an evidence table shows each paper's stance, study type, quality, confidence, key claim, and limitation flags. Excluded and failed extractions remain visible so users can audit selection bias.

## Error Handling and Observability

- Define typed errors for LLM failure, source timeout, authentication, rate limiting, response parsing, and invalid configuration.
- Log only provider name, operation, status, timing, and sanitized query metadata; never log API keys or complete private prompts.
- Close all HTTP clients after a search.
- Treat a single-source failure as a partial result; treat all-source failure as an error.
- Never cache authentication failures. Use short caching for retryable partial failures and normal caching for complete successes.

## Testing Strategy

All provider tests use local fake HTTP responses and consume no API quota. Tests are written before production changes and must reproduce these failures first:

- repeated tokens survive OpenAlex abstract reconstruction
- AI error strings never become search variants
- the Streamlit-selected GLM configuration reaches query planning
- Chinese queries rank their English translation matches
- all configured sources can contribute despite a full OpenAlex result set
- Semantic Scholar and PubMed normalize to the canonical schema
- DOI URL and bare DOI forms deduplicate
- ranking occurs before result truncation
- provider errors remain distinguishable from zero results
- pagination uses a new cursor or offset instead of repeating page one
- consensus batch indices remain globally correct, or the obsolete batch-title filter is removed
- repeated searches do not duplicate the evidence set
- a Markdown section number cannot become the consensus score
- zero directional evidence returns “unavailable” rather than 50
- deterministic aggregation produces expected values for support, oppose, mixed, uncertain, and duplicate-cohort fixtures

Integration tests exercise `SearchEngine` with fake source adapters and the consensus service with fake GLM JSON. A final smoke test compiles the application and imports both Streamlit and engine modules.

## Compatibility and Rollout

- Existing collections, notes, papers, and SQLite data remain untouched.
- Existing calls to `SearchEngine.search()` continue to receive a list of paper dictionaries.
- The diagnostic search and consensus service are introduced alongside existing interfaces, then the search and consensus pages migrate to them.
- If no GLM key is configured, non-AI search still works with the original query and explains that translation/expansion was skipped.
- The settings page clearly identifies GLM as the default and permits users to choose another supported provider.

## Acceptance Criteria

- All new and existing tests pass without external network access.
- A fake five-source search returns contributions from multiple sources even when OpenAlex alone exceeds the requested limit.
- Chinese-to-English fallback remains useful when GLM is unavailable.
- Search results display partial-source diagnostics instead of false “no results” messages.
- Consensus scores come only from structured evidence and deterministic calculations.
- Every consensus claim links back to one or more included paper records and identifies abstract-only limitations.
- No existing database file or unrelated untracked user file is modified.

# Search, Consensus, and GLM Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair bilingual multi-source literature retrieval, make GLM-4.7-Flash the default LLM, and replace Markdown-parsed consensus guesses with auditable structured evidence aggregation.

**Architecture:** Keep provider HTTP clients and the backward-compatible `SearchEngine.search()` entry point in `engine.py`, while adding typed query, source, and diagnostic results around them. Put the new evidence model, GLM extraction workflow, and deterministic score in a focused `consensus.py`; migrate `app.py` to those services without changing navigation or the SQLite schema.

**Tech Stack:** Python 3.10+, asyncio, httpx, dataclasses, Streamlit, pytest, SQLAlchemy, existing provider APIs.

**Spec:** `docs/superpowers/specs/2026-08-31-search-consensus-glm-design.md`

## Global Constraints

- `glm-4.7-flash` is the default model and `glm` is the default provider; existing providers remain optional.
- Existing calls to `SearchEngine.search()` continue returning `list[dict]`.
- Existing SQLite data and schema remain untouched.
- Search works without any LLM key by using the original query and reporting skipped AI enhancement.
- No API key, complete private prompt, or `.env` value is logged or stored in a cache key.
- All automated tests run without network access or API quota.
- Production behavior changes follow a witnessed RED → GREEN cycle.
- Unrelated untracked files in `D:\articles` are never staged or modified.

---

### Task 1: GLM Defaults and Fail-safe Query Planning

**Files:**
- Modify: `engine.py:24-30,724-845,1141-1193`
- Modify: `app.py:3-62,1197-1235`
- Create: `tests/test_query_planning.py`

**Interfaces:**
- Produces: `DEFAULT_LLM_PROVIDER: str`, `DEFAULT_LLM_MODEL: str`, `LLMConfig`, `LLMRequestError`, `LLMClient.complete_strict(...)`, `QueryPlan`, and `QueryPlanner.build(query: str) -> QueryPlan`.
- Produces: `SearchEngine(llm_config: LLMConfig | None = None, clients: dict[str, object] | None = None)` for later source-fusion tests.
- Consumes: existing `LLMClient.translate_query()` and `LLMClient.expand_query()` behavior through strict calls.

- [ ] **Step 1: Write failing query-planning tests**

```python
import pytest

from engine import (
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    LLMConfig,
    LLMRequestError,
    QueryPlanner,
)


class FailingLLM:
    async def translate_query(self, query):
        raise LLMRequestError("glm", "authentication", "unauthorized")

    async def expand_query(self, query):
        raise LLMRequestError("glm", "authentication", "unauthorized")


class UsefulLLM:
    async def translate_query(self, query):
        return "silicon photonic modulator"

    async def expand_query(self, query):
        return ["1. silicon photonics", "[AI 调用失败：timeout]", "optical modulator"]


def test_glm_is_the_default_provider_consumed_by_new_sessions():
    assert LLMConfig.default() == LLMConfig("glm", "glm-4.7-flash", "")
    assert (DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL) == ("glm", "glm-4.7-flash")


@pytest.mark.asyncio
async def test_llm_failure_keeps_original_query_instead_of_searching_error_text():
    plan = await QueryPlanner(lambda: FailingLLM()).build("硅光调制器")
    assert plan.variants == ("硅光调制器",)
    assert plan.ai_error == "authentication"


@pytest.mark.asyncio
async def test_chinese_plan_keeps_original_translation_and_clean_expansion():
    plan = await QueryPlanner(lambda: UsefulLLM()).build("硅光调制器")
    assert plan.variants == ("硅光调制器", "silicon photonic modulator", "optical modulator")
    assert plan.translation_used is True
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest tests/test_query_planning.py -q`

Expected: collection fails because `LLMConfig`, `QueryPlanner`, and the GLM default constants do not exist.

- [ ] **Step 3: Implement strict LLM errors and query planning**

Add immutable dataclasses and strict failure handling in `engine.py`:

```python
DEFAULT_LLM_PROVIDER = "glm"
DEFAULT_LLM_MODEL = "glm-4.7-flash"


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str = ""

    @classmethod
    def default(cls):
        return cls(DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL, "")


class LLMRequestError(RuntimeError):
    def __init__(self, provider, category, detail):
        super().__init__(detail)
        self.provider = provider
        self.category = category


@dataclass(frozen=True)
class QueryPlan:
    original: str
    variants: tuple[str, ...]
    translation_used: bool = False
    ai_error: str | None = None
    kind: str = "topic"
```

Implement `LLMClient.complete_strict()` so HTTP/authentication/parsing failures raise `LLMRequestError`, while `_complete()` remains a backward-compatible user-facing wrapper. Implement `QueryPlanner` with a 200-character variant cap, removal of numbered/Markdown/error lines, case-insensitive deduplication, and a maximum of four variants. Change `SearchEngine` to build LLM clients from its injected `LLMConfig` instead of calling implicit DeepSeek. Change Streamlit session defaults from DeepSeek to GLM, set the GLM selector to the currently documented `glm-4.7-flash` and `glm-4.7` identifiers, and select `glm-4.7-flash` when no prior model is stored.

- [ ] **Step 4: Run query-planning tests and the existing compile check**

Run: `python -m pytest tests/test_query_planning.py -q`

Expected: all query-planning tests pass.

Run: `python -m compileall -q engine.py app.py`

Expected: exit code 0.

- [ ] **Step 5: Commit the isolated change**

```bash
git add engine.py app.py tests/test_query_planning.py
git commit -m "fix: default search assistance to glm"
```

---

### Task 2: Canonical Papers, Abstract Reconstruction, and Source Diagnostics

**Files:**
- Modify: `engine.py:553-718,976-1133`
- Create: `tests/test_source_normalization.py`

**Interfaces:**
- Consumes: `LLMRequestError` only for shared error-category naming.
- Produces: `SourceError`, `SourceBatch`, `SearchResponse`, `normalize_doi(value: str | None) -> str`, `normalize_semantic_paper(raw: dict) -> dict`, and `normalize_pubmed_summary(pmid: str, raw: dict) -> dict`.
- Changes: every provider implements `search(query: str, limit: int = 20, page: int = 1, offset: int = 0) -> SourceBatch`; providers ignore the pagination argument they do not use. Citation/reference methods keep their existing return types.

- [ ] **Step 1: Write failing normalization and diagnostic tests**

```python
from engine import (
    SourceBatch,
    SourceError,
    _flatten_abstract,
    normalize_doi,
    normalize_pubmed_summary,
    normalize_semantic_paper,
)


def test_openalex_abstract_reconstruction_keeps_repeated_tokens():
    assert _flatten_abstract({"the": [0, 2], "cat": [1]}) == "the cat the"


def test_doi_url_and_bare_doi_normalize_to_the_same_identity():
    assert normalize_doi("https://doi.org/10.1000/XYZ") == "10.1000/xyz"
    assert normalize_doi("doi: 10.1000/xyz") == "10.1000/xyz"


def test_semantic_scholar_response_maps_external_ids_and_publication_venue():
    paper = normalize_semantic_paper({
        "paperId": "s2",
        "title": "Paper",
        "authors": [{"name": "A"}],
        "citationCount": 9,
        "externalIds": {"DOI": "10.1/X"},
        "publicationVenue": {"name": "Nature"},
    })
    assert paper["source"] == "semantic_scholar"
    assert paper["semantic_id"] == "s2"
    assert paper["doi"] == "10.1/x"
    assert paper["venue"] == "Nature"
    assert paper["citation_count"] == 9


def test_source_failure_is_not_an_empty_success():
    batch = SourceBatch("openalex", (), 0, SourceError("timeout", None, True, "timed out"))
    assert batch.ok is False
    assert batch.error.category == "timeout"
```

- [ ] **Step 2: Run source tests and verify RED**

Run: `python -m pytest tests/test_source_normalization.py -q`

Expected: repeated-token assertion fails and new diagnostic/normalization symbols are missing.

- [ ] **Step 3: Implement canonical normalization and typed source batches**

Reconstruct every OpenAlex token position with `(position, token)` pairs. Add frozen `SourceError` and `SourceBatch` dataclasses, with `SourceBatch.ok` true only when `error is None`. Add `SearchResponse` containing `papers`, `query_plan`, `sources`, and derived `partial`/`all_failed` properties. Normalize DOI prefixes and metadata. Change Semantic Scholar search to return canonical papers using `externalIds` and `publicationVenue`. Change PubMed normalization to preserve `pmid` and avoid storing an e-location string as an abstract. Make provider exceptions return a categorized `SourceBatch` error for authentication, rate limit, timeout, HTTP, or parsing failures.

- [ ] **Step 4: Run source tests and combined unit tests**

Run: `python -m pytest tests/test_source_normalization.py tests/test_query_planning.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the isolated change**

```bash
git add engine.py tests/test_source_normalization.py
git commit -m "fix: normalize literature source responses"
```

---

### Task 3: Concurrent Multi-source Retrieval, Pagination, Deduplication, and Fusion

**Files:**
- Modify: `engine.py:1136-1307`
- Create: `tests/test_search_fusion.py`

**Interfaces:**
- Consumes: `QueryPlan`, `QueryPlanner`, `SourceBatch`, `SearchResponse`, and injected `SearchEngine.clients` from Tasks 1-2.
- Produces: `SearchEngine.search_with_diagnostics(query: str, limit: int = 20, source: str = "auto", sort: str = "relevance") -> SearchResponse`.
- Preserves: `SearchEngine.search(...) -> list[dict]` by returning `list(response.papers)`.

- [ ] **Step 1: Write failing fusion tests with complete fake provider responses**

```python
import pytest

from engine import LLMConfig, SearchEngine, SourceBatch


class StaticPlanner:
    async def build(self, query):
        from engine import QueryPlan
        return QueryPlan(query, (query, "silicon photonic modulator"), True, None, "topic")


class FakeSource:
    def __init__(self, source, pages):
        self.source = source
        self.pages = pages
        self.calls = []

    async def search(self, query, limit=20, page=1, offset=0):
        cursor = offset if self.source == "semantic_scholar" else page
        self.calls.append((query, cursor))
        papers = tuple(self.pages.get(cursor, ()))
        return SourceBatch(self.source, papers, sum(map(len, self.pages.values())), None)

    async def aclose(self):
        return None


def paper(source, ident, title, doi=""):
    return {"id": ident, "source": source, "title": title, "abstract": "", "year": 2025,
            "citation_count": 0, "venue": "", "authors": [], "doi": doi, "url": ""}


@pytest.mark.asyncio
async def test_full_openalex_pool_does_not_starve_other_sources():
    clients = {
        "openalex": FakeSource("openalex", {1: [paper("openalex", f"oa{i}", f"OA {i}") for i in range(20)]}),
        "arxiv": FakeSource("arxiv", {1: [paper("arxiv", "ax", "arXiv match")]}),
        "crossref": FakeSource("crossref", {1: [paper("crossref", "cr", "Crossref match")]}),
        "semantic_scholar": FakeSource("semantic_scholar", {0: [paper("semantic_scholar", "ss", "S2 match")]}),
        "pubmed": FakeSource("pubmed", {1: [paper("pubmed", "pm", "PubMed match")]}),
    }
    engine = SearchEngine(LLMConfig.default(), clients=clients, query_planner=StaticPlanner())
    response = await engine.search_with_diagnostics("硅光调制器", limit=5)
    assert {p["source"] for p in response.papers} == set(clients)


@pytest.mark.asyncio
async def test_dedup_merges_doi_metadata_before_limit_is_applied():
    richer = paper("crossref", "cr", "Exact paper", "10.1/x") | {"abstract": "rich abstract"}
    duplicate = paper("openalex", "oa", "Exact paper", "https://doi.org/10.1/X")
    clients = {
        "openalex": FakeSource("openalex", {1: [duplicate]}),
        "crossref": FakeSource("crossref", {1: [richer]}),
    }
    engine = SearchEngine(LLMConfig.default(), clients=clients, query_planner=StaticPlanner())
    response = await engine.search_with_diagnostics("Exact paper", limit=1)
    assert len(response.papers) == 1
    assert response.papers[0]["abstract"] == "rich abstract"
    assert set(response.papers[0]["sources"]) == {"openalex", "crossref"}
```

Add separate tests that a Chinese query ranks the English translation match above an unrelated first result, a second requested page changes page/offset, one failed source yields `partial=True`, and all failed sources yield `all_failed=True`.

Add table-driven tests for a DOI query (`10.1000/xyz`), arXiv identifier (`arXiv:2401.00001`), PMID (`PMID:12345`), OpenAlex identifier (`W12345`), and a title-like query longer than eight tokens. Assert that the query plan labels the detected kind and that broad topic fallback remains present for title-like input.

- [ ] **Step 2: Run fusion tests and verify RED**

Run: `python -m pytest tests/test_search_fusion.py -q`

Expected: `SearchEngine` rejects injected clients/planner or lacks `search_with_diagnostics`; source-starvation behavior fails.

- [ ] **Step 3: Implement fusion before truncation**

Run source/query tasks concurrently with `asyncio.gather(..., return_exceptions=True)`. Route DOI, arXiv, PMID, and OpenAlex identifiers through the providers that support direct lookup; route title-like input through a title-biased request plus the broad fallback. Fetch the next page/offset only when a provider reports more results and its first batch is smaller than its assigned candidate budget. Merge identities by normalized DOI, source identifier, then normalized title plus year. Preserve the richest abstract/authors/venue and a `sources` list. Score each source/query ranking with weighted RRF `query_weight / (60 + rank)`, add exact title and DOI bonuses, add bilingual token overlap, then sort the complete pool. When `limit >= successful_source_count`, reserve one highest-ranked unique paper per successful source before filling remaining positions by global score. Log only source, operation, status category, duration, and query kind. Close every injected or constructed client in `finally`.

- [ ] **Step 4: Run all search tests and verify GREEN**

Run: `python -m pytest tests/test_query_planning.py tests/test_source_normalization.py tests/test_search_fusion.py -q`

Expected: all search tests pass with no network access.

- [ ] **Step 5: Commit the isolated change**

```bash
git add engine.py tests/test_search_fusion.py
git commit -m "fix: fuse ranked results from all literature sources"
```

---

### Task 4: Search UI Configuration, Diagnostics, and Cache Safety

**Files:**
- Modify: `app.py:33-62,127-252,1197-1295`
- Create: `app_support.py`
- Create: `tests/test_app_support.py`

**Interfaces:**
- Consumes: `LLMConfig`, `SearchEngine.search_with_diagnostics()`, and `SearchResponse`.
- Produces: `build_llm_config(provider: str, model: str, api_key: str) -> LLMConfig`, `api_key_fingerprint(api_key: str) -> str`, and `format_search_diagnostics(response: SearchResponse) -> dict` in `app_support.py`.

- [ ] **Step 1: Write failing cache-safety and diagnostics tests**

```python
from app_support import api_key_fingerprint, build_llm_config, format_search_diagnostics
from engine import LLMConfig, QueryPlan, SearchResponse, SourceBatch, SourceError


def test_api_key_fingerprint_is_stable_and_does_not_expose_secret():
    fingerprint = api_key_fingerprint("super-secret-key")
    assert fingerprint == api_key_fingerprint("super-secret-key")
    assert "super-secret-key" not in fingerprint


def test_blank_ui_configuration_uses_glm_defaults():
    assert build_llm_config("", "", "") == LLMConfig("glm", "glm-4.7-flash", "")


def test_partial_search_diagnostics_name_failed_source():
    response = SearchResponse(
        (), QueryPlan("q", ("q",)),
        (SourceBatch("openalex", (), 0, SourceError("timeout", None, True, "timeout")),
         SourceBatch("arxiv", (), 0, None)),
    )
    summary = format_search_diagnostics(response)
    assert summary["partial"] is True
    assert summary["failed_sources"] == ["openalex"]
```

- [ ] **Step 2: Run app-support tests and verify RED**

Run: `python -m pytest tests/test_app_support.py -q`

Expected: `app_support` does not exist.

- [ ] **Step 3: Implement pure UI support and migrate the search page**

Use SHA-256 and retain only the first 12 hexadecimal characters for cache identity. Replace the unconditional `st.cache_data` search result with a session-state cache keyed by query, sort, provider, model, and fingerprint; store an expiry timestamp beside successful results, use a shorter expiry for retryable partial results, and never insert a response containing an authentication error. Provide a Retry button that removes that exact entry. Store and render provider counts and partial failures. Reset `search_page` when the normalized query changes. Distinguish no matches from all-provider failure. Keep paper cards and downstream charts consuming `response.papers`.

- [ ] **Step 4: Run UI support tests and compile the app**

Run: `python -m pytest tests/test_app_support.py -q`

Expected: all tests pass.

Run: `python -m compileall -q app.py app_support.py engine.py`

Expected: exit code 0.

- [ ] **Step 5: Commit the isolated change**

```bash
git add app.py app_support.py tests/test_app_support.py
git commit -m "fix: expose literature search diagnostics"
```

---

### Task 5: Deterministic Consensus Evidence Model and Score

**Files:**
- Create: `consensus.py`
- Create: `tests/test_consensus_scoring.py`

**Interfaces:**
- Produces: `EvidenceItem`, `ConsensusScore`, `ConsensusScore.insufficient(raw_counts: dict[str, int])`, and `calculate_consensus(items: Sequence[EvidenceItem]) -> ConsensusScore`.
- Produces literal stance values: `support`, `oppose`, `mixed`, `uncertain`, `not_applicable`.
- Provides fields consumed by Task 6: eligibility, study type, quality, directness, confidence, independence factor, claim, excerpt, and limitations.

- [ ] **Step 1: Write failing hand-calculated score tests**

```python
from consensus import EvidenceItem, calculate_consensus


def item(stance, *, quality="high", independence=1.0, eligible=True):
    return EvidenceItem(
        paper_id=stance, title=stance, eligible=eligible, exclusion_reason="",
        study_type="experimental", stance=stance, claim="claim", excerpt="evidence",
        quality=quality, directness=1.0, confidence=1.0,
        independence_factor=independence, limitations=("abstract_only",),
    )


def test_no_directional_evidence_returns_unavailable_not_fifty():
    result = calculate_consensus([item("uncertain")])
    assert result.score is None
    assert result.direction == "insufficient_evidence"


def test_eight_high_quality_independent_supporting_studies_score_one_hundred():
    result = calculate_consensus([item("support") for _ in range(8)])
    assert result.direction == "support"
    assert result.directional_agreement == 1.0
    assert result.evidence_certainty == 1.0
    assert result.score == 100


def test_four_support_and_four_oppose_have_fifty_percent_directional_agreement():
    result = calculate_consensus([item("support") for _ in range(4)] + [item("oppose") for _ in range(4)])
    assert result.direction == "mixed"
    assert result.directional_agreement == 0.5
    assert result.score == 50


def test_duplicate_cohort_reduces_effective_study_count():
    result = calculate_consensus([item("support"), item("support", independence=0.25)])
    assert result.effective_independent_studies == 1.25
    assert result.score == 16
```

- [ ] **Step 2: Run scoring tests and verify RED**

Run: `python -m pytest tests/test_consensus_scoring.py -q`

Expected: `consensus` module does not exist.

- [ ] **Step 3: Implement immutable evidence and deterministic aggregation**

Define `EvidenceItem` with the constructor fields shown in the test plus defaulted evidence details: `population`, `intervention`, `comparator`, `outcome`, `effect_direction`, `sample_size`, `statistical_evidence`, and `cohort_key`, all defaulting to `"not reported"` except `cohort_key=""`. Use study weights `experimental=1.0`, `observational=0.8`, `simulation=0.6`, `theory=0.6`, `unknown=0.4`, and zero for reviews/commentary. Use quality weights `high=1.0`, `medium=0.75`, `low=0.4`, `unknown=0.5`. Clamp directness, confidence, and independence to `[0, 1]`. Mixed adds half its directional weight to each side. Calculate agreement, weighted quality, effective independent study count, certainty, direction, raw counts, weighted totals, and rounded score exactly as specified in the design. Return `score=None` through `ConsensusScore.insufficient()` when no directional weight exists.

- [ ] **Step 4: Run scoring tests and verify GREEN**

Run: `python -m pytest tests/test_consensus_scoring.py -q`

Expected: all score fixtures pass.

- [ ] **Step 5: Commit the isolated change**

```bash
git add consensus.py tests/test_consensus_scoring.py
git commit -m "feat: calculate auditable consensus scores"
```

---

### Task 6: GLM Structured Evidence Extraction and Narrative Boundaries

**Files:**
- Modify: `consensus.py`
- Create: `tests/test_consensus_analysis.py`

**Interfaces:**
- Consumes: `EvidenceItem`, `ConsensusScore`, `calculate_consensus`, and an injected strict LLM with `complete_strict(messages, temperature=..., response_format=...)`.
- Produces: `ConsensusAnalysis`, `ConsensusAnalyzer.extract_one(question: str, paper: dict) -> EvidenceItem`, and `ConsensusAnalyzer.analyze(question: str, papers: Sequence[dict]) -> ConsensusAnalysis`.

- [ ] **Step 1: Write failing extraction and retry tests**

```python
import pytest

from consensus import ConsensusAnalyzer


class SequenceLLM:
    def __init__(self, responses):
        self.responses = iter(responses)

    async def complete_strict(self, messages, **kwargs):
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return value


PAPER = {"id": "p1", "title": "Trial", "abstract": "Treatment improved outcome in 80 participants."}
VALID = '''{"eligible":true,"exclusion_reason":"","study_type":"experimental",
"population":"participants","intervention":"treatment","comparator":"control","outcome":"outcome",
"stance":"support","claim":"Treatment improved outcome","effect_direction":"positive",
"sample_size":"80","statistical_evidence":"not reported","quality":"medium",
"directness":1.0,"confidence":0.9,"excerpt":"Treatment improved outcome in 80 participants.",
"limitations":["abstract_only"],"cohort_key":""}'''


@pytest.mark.asyncio
async def test_invalid_json_is_repaired_once_and_returns_traceable_evidence():
    analyzer = ConsensusAnalyzer(lambda: SequenceLLM(["not-json", VALID]))
    evidence = await analyzer.extract_one("Does treatment improve outcome?", PAPER)
    assert evidence.stance == "support"
    assert evidence.sample_size == "80"
    assert evidence.excerpt in PAPER["abstract"]


@pytest.mark.asyncio
async def test_unreported_statistics_remain_unreported():
    analyzer = ConsensusAnalyzer(lambda: SequenceLLM([VALID]))
    evidence = await analyzer.extract_one("Does treatment improve outcome?", PAPER)
    assert evidence.statistical_evidence == "not reported"
    assert "abstract_only" in evidence.limitations
```

Add tests that two failed parses produce an unavailable evidence record, an excerpt absent from the abstract is removed and flagged `untraceable_excerpt`, reviews receive zero directional weight, duplicate non-empty cohort keys get independence factors `1.0` then `0.25`, and narrative generation receives the fixed calculated score.

- [ ] **Step 2: Run analysis tests and verify RED**

Run: `python -m pytest tests/test_consensus_analysis.py -q`

Expected: `ConsensusAnalyzer` and `ConsensusAnalysis` do not exist.

- [ ] **Step 3: Implement strict per-paper JSON extraction**

Use one prompt per paper with title and available abstract only, a documented JSON schema, temperature 0, and GLM JSON response format when supported. Parse the first response strictly; retry once with a schema-repair prompt containing the invalid response. Convert missing facts to `not reported`, validate enum values and numeric ranges, verify excerpts are substrings of the supplied abstract, and return an excluded/unavailable record after two failures. Bound concurrency with `asyncio.Semaphore(4)`. Detect duplicate `cohort_key` values after extraction. Calculate the score before narrative generation and pass the immutable numbers into a final narrative prompt that cannot replace them.

- [ ] **Step 4: Run consensus tests and verify GREEN**

Run: `python -m pytest tests/test_consensus_scoring.py tests/test_consensus_analysis.py -q`

Expected: all consensus tests pass without network access.

- [ ] **Step 5: Commit the isolated change**

```bash
git add consensus.py tests/test_consensus_analysis.py
git commit -m "feat: extract structured consensus evidence with glm"
```

---

### Task 7: Migrate the Consensus Page to Structured Evidence

**Files:**
- Modify: `app.py:753-911`
- Modify: `app_support.py`
- Modify: `tests/test_app_support.py`

**Interfaces:**
- Consumes: `SearchEngine.search_with_diagnostics()`, `ConsensusAnalyzer.analyze()`, and `ConsensusAnalysis`.
- Produces: `consensus_rows(analysis: ConsensusAnalysis) -> list[dict]` for Streamlit evidence-table rendering.
- Removes: duplicate two-round search, 100-title batch-index filter, TF-IDF on untranslated Chinese, Markdown score parsing, and default score 50.

- [ ] **Step 1: Write a failing evidence-table behavior test**

```python
from app_support import consensus_rows
from consensus import ConsensusAnalysis, ConsensusScore, EvidenceItem


def test_consensus_rows_keep_excluded_and_unavailable_evidence_visible():
    evidence = EvidenceItem(
        paper_id="p", title="Paper", eligible=False, exclusion_reason="no direct outcome",
        study_type="unknown", stance="not_applicable", claim="", excerpt="",
        quality="unknown", directness=0.0, confidence=0.0, independence_factor=1.0,
        limitations=("abstract_only",),
    )
    score = ConsensusScore.insufficient({"not_applicable": 1})
    rows = consensus_rows(ConsensusAnalysis((evidence,), score, "Insufficient evidence"))
    assert rows == [{
        "论文": "Paper", "纳入": "否", "立场": "不适用", "研究类型": "unknown",
        "质量": "unknown", "抽取置信度": 0.0, "核心证据": "no direct outcome",
        "限制": "abstract_only",
    }]
```

- [ ] **Step 2: Run the focused UI test and verify RED**

Run: `python -m pytest tests/test_app_support.py::test_consensus_rows_keep_excluded_and_unavailable_evidence_visible -q`

Expected: `consensus_rows` does not exist.

- [ ] **Step 3: Replace the old consensus workflow**

Search once with the current GLM configuration and diagnostics, deduplicate through the search service, take at most 30 high-ranked diverse papers, and call `ConsensusAnalyzer`. Render `score=None` as “证据不足”; otherwise render fixed score, direction, agreement, certainty, effective study count, and weighted stance totals. Render the pure `consensus_rows()` result with `st.dataframe`, followed by the narrative and source paper expanders. Keep exclusions and failed extractions visible. Display the abstract-only limitation above the score. Do not parse any number from generated Markdown.

- [ ] **Step 4: Run all unit tests and compile the application**

Run: `python -m pytest tests -q`

Expected: all tests pass.

Run: `python -m compileall -q app.py app_support.py engine.py consensus.py`

Expected: exit code 0.

- [ ] **Step 5: Commit the isolated change**

```bash
git add app.py app_support.py tests/test_app_support.py
git commit -m "fix: base consensus ui on structured evidence"
```

---

### Task 8: Documentation, Offline Regression Suite, and Final Smoke Verification

**Files:**
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `requirements.txt`
- Modify: `tests/test_search_fusion.py`
- Modify: `tests/test_consensus_analysis.py`

**Interfaces:**
- Consumes all public interfaces delivered by Tasks 1-7.
- Produces user documentation for GLM default configuration, source diagnostics, and score limitations.

- [ ] **Step 1: Add end-to-end offline regression tests**

Add one fake five-source `SearchEngine` test that starts from a Chinese query, uses a fake GLM translation, experiences one retryable provider failure, returns deduplicated multi-source papers, and reports `partial=True`. Add one fake `ConsensusAnalyzer` test with support, oppose, uncertain, excluded review, and duplicate-cohort JSON fixtures; assert the literal final weighted totals and score, and assert no Markdown parsing is involved in the returned object.

- [ ] **Step 2: Run the new regression tests and verify they fail for missing integration behavior**

Run: `python -m pytest tests/test_search_fusion.py tests/test_consensus_analysis.py -q`

Expected: newly added integration assertions fail until the remaining wiring or metadata merge gaps are implemented.

- [ ] **Step 3: Implement only the integration gaps and update documentation**

Update README features and configuration to name GLM-4.7-Flash as default, describe five-source partial diagnostics, and explain that consensus is abstract-based and confidence-adjusted rather than a meta-analysis. Update `.env.example` to lead with `ZHIPUAI_API_KEY`, retain optional provider keys, and never include a real key. Add `pytest>=8.0` and `pytest-asyncio>=0.23` to requirements so the documented test command works in a clean environment.

- [ ] **Step 4: Run complete fresh verification**

Run: `python -m pytest tests -q`

Expected: all tests pass, zero failures.

Run: `python -m compileall -q app.py app_support.py engine.py consensus.py tests`

Expected: exit code 0.

Run: `git diff --check`

Expected: no whitespace errors.

Run: `git status --short`

Expected: only intended tracked implementation files plus the user's pre-existing unrelated untracked files are listed; no database, `.env`, document, chart, or report file is staged.

- [ ] **Step 5: Commit final documentation and regression coverage**

```bash
git add README.md .env.example requirements.txt tests/test_search_fusion.py tests/test_consensus_analysis.py
git commit -m "test: cover bilingual search and consensus workflow"
```

- [ ] **Step 6: Review the completed branch against the design**

Run: `git log --oneline --decorate main..HEAD`

Expected: task-scoped commits for GLM defaults, source normalization, multi-source fusion, diagnostics UI, consensus scoring, evidence extraction, consensus UI, and final regression coverage.

Run: `git diff --stat main...HEAD`

Expected: changes are limited to the plan's production, test, and documentation files.

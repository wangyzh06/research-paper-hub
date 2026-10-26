"""Google Scholar 客户端与引用核验测试。

Google Scholar 背景（实测 2026-10）：
  - 无官方 API，纯 HTML 解析；
  - 经代理 24 次请求成功率 96%，失败为 HTTP 429 + 人机验证（间歇性）；
  - 触发后人机验证会把 IP 放入 google.com/sorry 惩罚页，需等待恢复；
  - 因此设计为**默认关闭、失败静默降级**，绝不阻塞主检索。

引用核验背景：LLM 可能编造 DOI/引用。核验层负责在报告产出前
把这些不可追溯的证据标记出来（借鉴 PaperPilot 的 verification 设计）。
"""
from __future__ import annotations

import pytest

from engine import (
    GS_PAGE_SIZE,
    GoogleScholarClient,
    parse_google_scholar_html,
    parse_google_scholar_meta,
)
from consensus import (
    DOI_PATTERN,
    EvidenceItem,
    build_quality_gate,
    verify_evidence,
)

# 真实 Google Scholar 结果块（取自实测 HTML，已裁剪）
REAL_BLOCK = '''
<div class="gs_r gs_or gs_scl" data-cid="abc" data-rp="0">
  <div class="gs_ri">
    <h3 class="gs_rt"><a href="https://pubs.aip.org/aip/jap/article/93/1/1">Observation of enhanced transport in carbon-doped InGaAsN</a></h3>
    <div class="gs_a">C Monier, AG Baca, SZ Sun, E Armour… - Applied physics …, 2002 - pubs.aip.org</div>
    <div class="gs_rs">We report enhanced transport after in situ anneal in carbon-doped InGaAsN base layers.</div>
    <div class="gs_fl"><a href="#">Cited by 28</a></div>
  </div>
</div>
<div class="gs_r gs_or gs_scl" data-cid="def" data-rp="1">
  <div class="gs_ri">
    <h3 class="gs_rt"><a href="https://ieeexplore.ieee.org/x">Excess noise reduction in GaInP/GaAs HBTs</a></h3>
    <div class="gs_a">H Yamada, N Fukuhara - Japanese journal of applied …, 2006 - iopscience.iop.org</div>
    <div class="gs_rs">Noise reduction is demonstrated.</div>
    <div class="gs_fl"><a href="#">Cited by 5</a></div>
  </div>
</div>
'''

EMPTY_BLOCK = '''
<div class="gs_r gs_or gs_scl">
  <div class="gs_ri">
    <h3 class="gs_rt">A citation entry without a link</h3>
    <div class="gs_a">Someone - Venue, 2010</div>
  </div>
</div>
'''


# ============ meta 解析 ============
def test_parse_meta_extracts_year_via_regex_not_split():
    """年份必须用正则提取：venue 与 year 以逗号相连，按 ' - ' 分割会错位。"""
    meta = "C Monier, AG Baca - Applied physics …, 2002 - pubs.aip.org"
    authors, venue, year = parse_google_scholar_meta(meta)

    assert authors == "C Monier, AG Baca"
    assert year == 2002
    assert "Applied physics" in venue
    assert "2002" not in venue, "年份不应残留在 venue 中"


def test_parse_meta_without_publisher_segment():
    authors, venue, year = parse_google_scholar_meta(
        "M Mimila-Arroyo - Journal of applied physics, 2003"
    )
    assert authors == "M Mimila-Arroyo"
    assert year == 2003
    assert venue == "Journal of applied physics"


def test_parse_meta_handles_empty():
    authors, venue, year = parse_google_scholar_meta("")
    assert authors == ""
    assert venue == ""
    assert year is None


# ============ HTML 解析 ============
def test_parse_html_extracts_two_results():
    papers = parse_google_scholar_html(REAL_BLOCK)

    assert len(papers) == 2
    first = papers[0]
    assert "carbon-doped InGaAsN" in first["title"]
    assert first["year"] == 2002
    assert first["citation_count"] == 28
    assert first["source"] == "google_scholar"
    assert "enhanced transport" in first["abstract"]
    assert first["authors"][0]["name"] == "C Monier"


def test_parse_html_handles_entry_without_link():
    papers = parse_google_scholar_html(EMPTY_BLOCK)

    assert len(papers) == 1
    assert papers[0]["url"] == ""
    assert papers[0]["title"] == "A citation entry without a link"


def test_parse_html_empty_input():
    assert parse_google_scholar_html("") == []
    assert parse_google_scholar_html("<html>no results</html>") == []


def test_parse_html_strips_type_markers():
    html = '''
    <div class="gs_r gs_or gs_scl">
      <div class="gs_ri">
        <h3 class="gs_rt"><span class="gs_ctc">[PDF]</span> <a href="http://x">Real Title</a></h3>
        <div class="gs_a">A Author - V, 2019</div>
      </div>
    </div>'''
    papers = parse_google_scholar_html(html)

    assert papers[0]["title"] == "Real Title"
    assert "[PDF]" not in papers[0]["title"]


def test_parse_html_is_deterministic():
    assert parse_google_scholar_html(REAL_BLOCK) == parse_google_scholar_html(REAL_BLOCK)


# ============ 客户端行为 ============
def test_gs_page_size_is_ten():
    """实测：Google Scholar 每页最多 10 条，即使 num=20。"""
    assert GS_PAGE_SIZE == 10


@pytest.mark.asyncio
async def test_client_handles_429_as_rate_limit(monkeypatch):
    """429 必须归类为 rate_limit（可重试/可跳过），而非崩溃。"""

    class FakeResponse:
        status_code = 429
        text = "unusual traffic detected"

        def raise_for_status(self):
            raise RuntimeError("429")

    class FakeHTTP:
        async def get(self, url, params=None, **kw):
            return FakeResponse()

    client = GoogleScholarClient()
    client._client = None  # unused
    client.client = FakeHTTP()  # type: ignore

    batch = await client.search("test query", limit=10)

    assert not batch.ok
    assert batch.error.category == "rate_limit"
    assert batch.papers == ()


@pytest.mark.asyncio
async def test_client_detects_captcha_page_even_on_200():
    """Google 有时返回 200 但内容是验证页，必须识别为限流而非空结果。"""

    class FakeResponse:
        status_code = 200
        text = "<html>Please show you're not a robot — unusual traffic</html>"

        def raise_for_status(self):
            pass

    class FakeHTTP:
        async def get(self, url, params=None, **kw):
            return FakeResponse()

    client = GoogleScholarClient()
    client.client = FakeHTTP()  # type: ignore

    batch = await client.search("test", limit=10)

    assert not batch.ok
    assert batch.error.category == "rate_limit"


@pytest.mark.asyncio
async def test_client_returns_papers_on_success():
    class FakeResponse:
        status_code = 200
        text = REAL_BLOCK

        def raise_for_status(self):
            pass

    class FakeHTTP:
        async def get(self, url, params=None, **kw):
            return FakeResponse()

    client = GoogleScholarClient()
    client.client = FakeHTTP()  # type: ignore

    batch = await client.search("InGaP GaAs HBT", limit=10)

    assert batch.ok
    assert len(batch.papers) == 2


@pytest.mark.asyncio
async def test_client_empty_query_returns_empty():
    client = GoogleScholarClient()
    batch = await client.search("   ", limit=10)
    assert batch.papers == ()


# ============ 引用核验 ============
def _ev(pid, stance="support", eligible=True, excerpt="excerpt", limitations=()):
    return EvidenceItem(
        paper_id=pid, title=f"P {pid}", eligible=eligible, exclusion_reason="",
        study_type="bench_test", stance=stance, claim="c", excerpt=excerpt,
        quality="high", directness=1.0, confidence=1.0, independence_factor=1.0,
        limitations=tuple(limitations))


def _papers():
    abs_ok = "We study annealing of InGaP GaAs HBT devices in detail. " * 4
    return [
        {"id": "good", "doi": "10.1063/1.1140312", "url": "https://doi.org/x", "abstract": abs_ok},
        {"id": "bad_doi", "doi": "not-a-doi", "url": "https://x", "abstract": abs_ok},
        {"id": "no_url", "doi": "10.1016/j.jcrysgro.2006.10.114", "url": "", "abstract": abs_ok},
        {"id": "no_abs", "doi": "10.1109/16.737435", "url": "https://ieee.org", "abstract": ""},
    ]


def test_doi_pattern_accepts_valid_and_rejects_fake():
    assert DOI_PATTERN.match("10.1063/1.1140312")
    assert DOI_PATTERN.match("10.1016/j.jcrysgro.2006.10.114")
    assert not DOI_PATTERN.match("not-a-doi")
    assert not DOI_PATTERN.match("10.123/too-short-prefix")


def test_verify_evidence_flags_invalid_doi():
    """核心价值：抓出模型编造的 DOI。"""
    result = verify_evidence([_ev("bad_doi")], _papers())
    record = result["records"][0]

    assert record["doi_status"] == "format_invalid"
    assert "invalid_doi_format" in record["issues"]
    assert record["status"] == "plausible_with_warnings"


def test_verify_evidence_flags_missing_url_and_abstract():
    result = verify_evidence([_ev("no_url"), _ev("no_abs")], _papers())
    by_id = {r["paper_id"]: r for r in result["records"]}

    assert "missing_url" in by_id["no_url"]["issues"]
    assert by_id["no_abs"]["evidence_basis"] == "metadata_only"
    assert "no_abstract" in by_id["no_abs"]["issues"]


def test_verify_evidence_marks_untraceable_excerpt():
    """excerpt 无法在摘要中定位时，必须标记（防止编造证据）。"""
    ev = _ev("no_abs", excerpt="fabricated quote", limitations=("untraceable_excerpt",))
    result = verify_evidence([ev], _papers())

    assert result["records"][0]["excerpt_status"] == "untraceable"
    assert "untraceable_excerpt" in result["records"][0]["issues"]


def test_verify_evidence_clean_case_is_verified():
    ev = _ev("good", excerpt="We study annealing of InGaP GaAs HBT devices in detail.")
    result = verify_evidence([ev], _papers())

    assert result["records"][0]["status"] == "verified"
    assert result["verified"] == 1
    assert result["records"][0]["excerpt_status"] == "traceable"
    assert result["records"][0]["url_status"] == "present"
    assert result["records"][0]["url_validation"] == "not_performed"


def test_verify_evidence_excluded_records_are_separated():
    result = verify_evidence([_ev("good", eligible=False)], _papers())

    assert result["records"][0]["status"] == "excluded"
    assert result["excluded"] == 1


def test_verify_evidence_warns_on_invalid_doi_present():
    result = verify_evidence([_ev("bad_doi")], _papers())

    assert any("DOI 格式非法" in w for w in result["verification_warnings"])


def test_verify_evidence_is_deterministic():
    a = verify_evidence([_ev("good")], _papers())
    b = verify_evidence([_ev("good")], _papers())
    assert a == b


# ============ 质量门控 ============
def test_quality_gate_fails_when_too_few_verified():
    verification = {"total": 3, "verified": 1, "excluded": 0, "verification_warnings": []}
    gate = build_quality_gate(verification, min_papers=5)

    assert not gate["pass"]
    assert "too_few_verified_papers" in gate["issues"]


def test_quality_gate_flags_invalid_doi():
    verification = {"total": 10, "verified": 8, "excluded": 0,
                    "verification_warnings": ["存在 DOI 格式非法的引用，可能为模型编造"]}
    gate = build_quality_gate(verification, min_papers=5)

    assert "invalid_doi_present" in gate["issues"]
    assert any("剔除" in r for r in gate["recommendations"])


def test_quality_gate_flags_insufficient_evidence_score():
    """证据不足的门控方向需经 apply_min_study_gate 才会置为 insufficient_evidence。"""
    from consensus import apply_min_study_gate, calculate_consensus

    evidence = [_ev("a"), _ev("b")]
    score = apply_min_study_gate(calculate_consensus(evidence), evidence, minimum=5)
    assert score.direction == "insufficient_evidence"

    verification = {"total": 2, "verified": 2, "excluded": 0, "verification_warnings": []}
    gate = build_quality_gate(verification, score=score, min_papers=5)

    assert "insufficient_evidence" in gate["issues"]


def test_quality_gate_passes_on_clean_large_corpus():
    verification = {"total": 20, "verified": 18, "excluded": 2, "verification_warnings": []}
    gate = build_quality_gate(verification, score=None, min_papers=5)

    assert gate["pass"]
    assert gate["issues"] == []


def test_low_verification_ratio_fails_even_without_invalid_dois():
    abstract = "A traceable abstract reporting experimental observations. " * 3
    evidence = [_ev(f"p{i}", excerpt=abstract[:60]) for i in range(20)]
    papers = [
        {
            "id": f"p{i}",
            "url": "https://example.org/paper" if i < 8 else "",
            "abstract": abstract,
        }
        for i in range(20)
    ]

    verification = verify_evidence(evidence, papers)
    gate = build_quality_gate(verification)

    assert verification["verified"] == 8
    assert verification["plausible_with_warnings"] == 12
    assert any("一半" in warning for warning in verification["verification_warnings"])
    assert gate["pass"] is False
    assert "low_verification_rate" in gate["issues"]

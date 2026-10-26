from __future__ import annotations

import os
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
import fitz
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey, Table, create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Mapped, mapped_column, relationship

logger = logging.getLogger(__name__)
load_dotenv()


def _httpx_timeout_types() -> tuple[type[BaseException], ...]:
    names = ("TimeoutException", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout")
    return tuple(
        error_type
        for name in names
        if isinstance((error_type := getattr(httpx, name, None)), type)
        and issubclass(error_type, BaseException)
    )


def _httpx_network_types() -> tuple[type[BaseException], ...]:
    names = (
        "NetworkError", "ProtocolError", "ConnectError", "ReadError",
        "WriteError", "CloseError", "ProxyError", "StreamError",
    )
    modules = (httpx, getattr(httpx, "_exceptions", None))
    error_types = []
    for module in modules:
        if module is None:
            continue
        for name in names:
            error_type = getattr(module, name, None)
            if (
                isinstance(error_type, type)
                and issubclass(error_type, BaseException)
                and error_type not in error_types
            ):
                error_types.append(error_type)
    return tuple(error_types)

# ============================================================
# Config
# ============================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = os.getenv("DATA_DIR", str(BASE_DIR / "data"))
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
DB_PATH = os.path.join(DATA_DIR, "papers.db")
os.makedirs(PDF_DIR, exist_ok=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-v4-flash"
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "")
DEFAULT_LLM_PROVIDER = "deepseek"
DEFAULT_LLM_MODEL = "deepseek-v4-flash"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

# ============================================================
# Database
# ============================================================
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _utcnow():
    return datetime.now(timezone.utc)


# ============================================================
# Association Tables
# ============================================================
paper_author = Table("paper_author", Base.metadata,
    Column("paper_id", Integer, ForeignKey("paper.id", ondelete="CASCADE"), primary_key=True),
    Column("author_id", Integer, ForeignKey("author.id", ondelete="CASCADE"), primary_key=True),
)
paper_collection = Table("paper_collection", Base.metadata,
    Column("paper_id", Integer, ForeignKey("paper.id", ondelete="CASCADE"), primary_key=True),
    Column("collection_id", Integer, ForeignKey("collection.id", ondelete="CASCADE"), primary_key=True),
)
paper_tag = Table("paper_tag", Base.metadata,
    Column("paper_id", Integer, ForeignKey("paper.id", ondelete="CASCADE"), primary_key=True),
    Column("tag_id", Integer, ForeignKey("tag.id", ondelete="CASCADE"), primary_key=True),
)

# ============================================================
# Models
# ============================================================
class Paper(Base):
    __tablename__ = "paper"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    venue: Mapped[str | None] = mapped_column(String, nullable=True)
    citation_count: Mapped[int] = mapped_column(Integer, default=0)
    pdf_path: Mapped[str | None] = mapped_column(String, nullable=True)
    semantic_id: Mapped[str | None] = mapped_column(String, nullable=True)
    openalex_id: Mapped[str | None] = mapped_column(String, nullable=True)
    url: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    authors: Mapped[list["Author"]] = relationship(secondary=paper_author, back_populates="papers")
    collections: Mapped[list["Collection"]] = relationship(secondary=paper_collection, back_populates="papers")
    tags: Mapped[list["Tag"]] = relationship(secondary=paper_tag, back_populates="papers")
    key_points: Mapped[list["KeyPoint"]] = relationship(back_populates="paper", cascade="all, delete-orphan")
    chats: Mapped[list["ChatMessage"]] = relationship(back_populates="paper", cascade="all, delete-orphan")
    notes: Mapped[list["Note"]] = relationship(back_populates="paper", cascade="all, delete-orphan")


class Author(Base):
    __tablename__ = "author"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    affiliation: Mapped[str | None] = mapped_column(String, nullable=True)

    papers: Mapped[list["Paper"]] = relationship(secondary=paper_author, back_populates="authors")


class Citation(Base):
    __tablename__ = "citation"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    paper_id: Mapped[int] = mapped_column(Integer, ForeignKey("paper.id", ondelete="CASCADE"))
    cited_paper_id: Mapped[int] = mapped_column(Integer, ForeignKey("paper.id", ondelete="CASCADE"))
    relationship_type: Mapped[str] = mapped_column(String, default="citation")


class Collection(Base):
    __tablename__ = "collection"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    papers: Mapped[list["Paper"]] = relationship(secondary=paper_collection, back_populates="collections")


class KeyPoint(Base):
    __tablename__ = "key_point"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    paper_id: Mapped[int] = mapped_column(Integer, ForeignKey("paper.id", ondelete="CASCADE"))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String, default="general")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    paper: Mapped["Paper"] = relationship(back_populates="key_points")


class ChatMessage(Base):
    __tablename__ = "chat_message"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    paper_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("paper.id", ondelete="CASCADE"), nullable=True)
    role: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    paper: Mapped["Paper | None"] = relationship(back_populates="chats")


class Consensus(Base):
    __tablename__ = "consensus"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    topic: Mapped[str] = mapped_column(String, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    papers_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class Note(Base):
    __tablename__ = "note"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    paper_id: Mapped[int] = mapped_column(Integer, ForeignKey("paper.id", ondelete="CASCADE"))
    content: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)

    paper: Mapped["Paper"] = relationship(back_populates="notes")


class Tag(Base):
    __tablename__ = "tag"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    color: Mapped[str] = mapped_column(String, default="#4361ee")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    papers: Mapped[list["Paper"]] = relationship(secondary=paper_tag, back_populates="tags")


# ============================================================
# CRUD Helpers (同步，供 Streamlit 调用)
# ============================================================
def save_paper_to_db(paper_dict: dict) -> int:
    db = SessionLocal()
    try:
        oaid = paper_dict.get("id", "") or paper_dict.get("openalex_id", "")
        existing = None
        if oaid:
            existing = db.query(Paper).filter(Paper.openalex_id == oaid).first()
        if not existing:
            sem_id = paper_dict.get("semantic_id", "") or paper_dict.get("id", "")
            if sem_id and sem_id.startswith("arxiv"):
                existing = db.query(Paper).filter(Paper.semantic_id == sem_id).first()
        if existing:
            return existing.id
        p = Paper(
            title=paper_dict.get("title", ""),
            abstract=paper_dict.get("abstract", ""),
            year=paper_dict.get("year"),
            venue=paper_dict.get("venue", ""),
            citation_count=paper_dict.get("citation_count", 0),
            openalex_id=oaid or None,
            semantic_id=(paper_dict.get("semantic_id") or None),
            url=paper_dict.get("url", ""),
            created_at=_utcnow(),
        )
        db.add(p)
        db.commit()
        db.refresh(p)
        return p.id
    finally:
        db.close()


def add_to_collection(collection_id: int, paper_id: int) -> None:
    db = SessionLocal()
    try:
        c = db.query(Collection).get(collection_id)
        p = db.query(Paper).get(paper_id)
        if c and p and p not in c.papers:
            c.papers.append(p)
            db.commit()
    finally:
        db.close()


def remove_from_collection(collection_id: int, paper_id: int) -> None:
    db = SessionLocal()
    try:
        c = db.query(Collection).get(collection_id)
        p = db.query(Paper).get(paper_id)
        if c and p and p in c.papers:
            c.papers.remove(p)
            db.commit()
    finally:
        db.close()


def get_all_collections() -> list[dict]:
    db = SessionLocal()
    try:
        result = []
        for c in db.query(Collection).order_by(Collection.created_at.desc()).all():
            result.append({
                "id": c.id, "name": c.name, "description": c.description or "",
                "papers_count": len(c.papers),
                "created_at": c.created_at.isoformat() if c.created_at else "",
            })
        return result
    finally:
        db.close()


def get_collection_detail(collection_id: int) -> dict | None:
    db = SessionLocal()
    try:
        c = db.query(Collection).get(collection_id)
        if not c:
            return None
        papers_data = []
        for p in c.papers:
            papers_data.append({
                "id": p.id, "title": p.title, "year": p.year,
                "venue": p.venue, "citation_count": p.citation_count,
                "openalex_id": p.openalex_id, "semantic_id": p.semantic_id,
                "abstract": p.abstract, "url": p.url,
            })
        return {
            "id": c.id, "name": c.name, "description": c.description or "",
            "papers": papers_data,
        }
    finally:
        db.close()


def create_collection(name: str, description: str = "") -> int:
    db = SessionLocal()
    try:
        c = Collection(name=name, description=description, created_at=_utcnow())
        db.add(c)
        db.commit()
        db.refresh(c)
        return c.id
    finally:
        db.close()


def delete_collection(collection_id: int) -> None:
    db = SessionLocal()
    try:
        c = db.query(Collection).get(collection_id)
        if c:
            db.delete(c)
            db.commit()
    finally:
        db.close()


def export_bibtex(collection_id: int) -> str:
    """Export collection papers as BibTeX"""
    db = SessionLocal()
    try:
        c = db.query(Collection).get(collection_id)
        if not c or not c.papers:
            return ""
        entries = []
        for p in c.papers:
            author_str = " and ".join(a.name for a in p.authors[:5]) if p.authors else "Unknown"
            cite_key = f"{author_str.split(' and ')[0].split(',')[0].strip().replace(' ','')}{p.year or '0000'}{p.title[:3].replace(' ','')}"
            entry = f"@article{{{cite_key},\n"
            entry += f"  title = {{{p.title}}},\n"
            entry += f"  author = {{{author_str}}},\n"
            entry += f"  year = {{{p.year or '0000'}}},\n"
            if p.venue: entry += f"  journal = {{{p.venue}}},\n"
            if p.abstract: entry += f"  abstract = {{{p.abstract[:200]}}},\n"
            if p.url: entry += f"  url = {{{p.url}}},\n"
            if p.openalex_id: entry += f"  note = {{OpenAlex: {p.openalex_id}}},\n"
            entry += "}\n"
            entries.append(entry)
        return "\n".join(entries)
    finally:
        db.close()


# ============================================================
# Note CRUD
# ============================================================
def save_note(paper_id: int, content: str) -> None:
    db = SessionLocal()
    try:
        note = db.query(Note).filter(Note.paper_id == paper_id).first()
        if note:
            note.content = content
            note.updated_at = _utcnow()
        else:
            note = Note(paper_id=paper_id, content=content, created_at=_utcnow(), updated_at=_utcnow())
            db.add(note)
        db.commit()
    finally:
        db.close()


def get_note(paper_id: int) -> str:
    db = SessionLocal()
    try:
        note = db.query(Note).filter(Note.paper_id == paper_id).first()
        return note.content if note else ""
    finally:
        db.close()


def delete_note(paper_id: int) -> None:
    db = SessionLocal()
    try:
        note = db.query(Note).filter(Note.paper_id == paper_id).first()
        if note:
            db.delete(note)
            db.commit()
    finally:
        db.close()


def get_all_notes() -> list[dict]:
    db = SessionLocal()
    try:
        notes = db.query(Note).order_by(Note.updated_at.desc()).all()
        result = []
        for n in notes:
            paper = db.query(Paper).get(n.paper_id)
            result.append({
                "paper_id": n.paper_id,
                "paper_title": paper.title if paper else "未知论文",
                "content": n.content,
                "updated_at": n.updated_at.isoformat() if n.updated_at else "",
            })
        return result
    finally:
        db.close()


# ============================================================
# Tag CRUD
# ============================================================
def create_tag(name: str, color: str = "#4361ee") -> int:
    db = SessionLocal()
    try:
        existing = db.query(Tag).filter(Tag.name == name).first()
        if existing:
            return existing.id
        tag = Tag(name=name, color=color, created_at=_utcnow())
        db.add(tag)
        db.commit()
        db.refresh(tag)
        return tag.id
    finally:
        db.close()


def get_all_tags() -> list[dict]:
    db = SessionLocal()
    try:
        tags = db.query(Tag).order_by(Tag.name).all()
        return [{"id": t.id, "name": t.name, "color": t.color, "papers_count": len(t.papers)} for t in tags]
    finally:
        db.close()


def delete_tag(tag_id: int) -> None:
    db = SessionLocal()
    try:
        tag = db.query(Tag).get(tag_id)
        if tag:
            db.delete(tag)
            db.commit()
    finally:
        db.close()


def add_tag_to_paper(paper_id: int, tag_id: int) -> None:
    db = SessionLocal()
    try:
        paper = db.query(Paper).get(paper_id)
        tag = db.query(Tag).get(tag_id)
        if paper and tag and tag not in paper.tags:
            paper.tags.append(tag)
            db.commit()
    finally:
        db.close()


def remove_tag_from_paper(paper_id: int, tag_id: int) -> None:
    db = SessionLocal()
    try:
        paper = db.query(Paper).get(paper_id)
        tag = db.query(Tag).get(tag_id)
        if paper and tag and tag in paper.tags:
            paper.tags.remove(tag)
            db.commit()
    finally:
        db.close()


def get_paper_tags(paper_id: int) -> list[dict]:
    db = SessionLocal()
    try:
        paper = db.query(Paper).get(paper_id)
        if not paper:
            return []
        return [{"id": t.id, "name": t.name, "color": t.color} for t in paper.tags]
    finally:
        db.close()


# ============================================================
# CiteSpace-Inspired Analysis Functions
# ============================================================
from datetime import datetime as _dt

def detect_burst(papers: list[dict], top_n: int = 10) -> list[dict]:
    """突变检测：发现引用量激增的论文（按年均引用排序）"""
    current_year = _dt.now().year
    scored = []
    for p in papers:
        year = p.get("year") or current_year
        citations = p.get("citation_count", 0)
        age = max(1, current_year - year + 1)
        cpy = citations / age  # citations per year
        scored.append({**p, "_cpy": round(cpy, 1), "_age": age})
    scored.sort(key=lambda x: x["_cpy"], reverse=True)
    return scored[:top_n]


def cluster_papers(papers: list[dict], n_clusters: int = 5) -> dict:
    """聚类分析：使用嵌入将论文自动分组"""
    if len(papers) < n_clusters:
        n_clusters = max(2, len(papers) // 2)
    if len(papers) < 3:
        return {"clusters": [{"id": 0, "label": "全部", "papers": papers}]}

    try:
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
        has_sklearn = True
    except ImportError:
        has_sklearn = False

    if not has_sklearn:
        # Fallback: group by year range
        by_decade = {}
        for p in papers:
            year = p.get("year") or 2020
            decade = f"{(year // 10) * 10}s"
            by_decade.setdefault(decade, []).append(p)
        clusters = [{"id": i, "label": k, "papers": v} for i, (k, v) in enumerate(by_decade.items())]
        return {"clusters": clusters}

    # Use TF-IDF on titles for clustering
    titles = [(p.get("title") or "") for p in papers]
    try:
        vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
        X = vectorizer.fit_transform(titles)
        km = KMeans(n_clusters=n_clusters, random_state=42)
        labels = km.fit_predict(X)

        # Extract top keywords per cluster
        feature_names = vectorizer.get_feature_names_out()
        clusters = []
        for i in range(n_clusters):
            mask = labels == i
            cluster_papers = [p for p, m in zip(papers, mask) if m]
            if not cluster_papers:
                continue
            # Top keywords for this cluster
            center = km.cluster_centers_[i]
            top_indices = center.argsort()[-3:][::-1]
            keywords = [feature_names[j] for j in top_indices if center[j] > 0]
            label = " / ".join(keywords) if keywords else f"簇 {i+1}"
            clusters.append({"id": i, "label": label, "papers": cluster_papers})
        return {"clusters": clusters}
    except Exception:
        return {"clusters": [{"id": 0, "label": "全部", "papers": papers}]}


def timeline_analysis(papers: list[dict]) -> dict:
    """时间线分析：按年份统计论文数量和引用趋势"""
    by_year = {}
    for p in papers:
        year = p.get("year")
        if not year:
            continue
        if year not in by_year:
            by_year[year] = {"count": 0, "total_citations": 0, "papers": []}
        by_year[year]["count"] += 1
        by_year[year]["total_citations"] += p.get("citation_count", 0)
        by_year[year]["papers"].append(p.get("title", "")[:50])

    timeline = []
    for year in sorted(by_year.keys()):
        data = by_year[year]
        timeline.append({
            "year": year,
            "count": data["count"],
            "total_citations": data["total_citations"],
            "avg_citations": round(data["total_citations"] / max(1, data["count"]), 1),
            "papers": data["papers"][:5],
        })
    return {"timeline": timeline, "years": [t["year"] for t in timeline],
            "counts": [t["count"] for t in timeline],
            "citations": [t["total_citations"] for t in timeline]}


# ============================================================
# OpenAlex Client
# ============================================================
@dataclass(frozen=True)
class SourceError:
    category: str
    status_code: int | None
    retryable: bool
    detail: str


@dataclass(frozen=True)
class SourceBatch:
    source: str
    papers: tuple[dict, ...]
    total: int = 0
    error: SourceError | None = None
    # 分页游标（目前仅 S2 bulk 使用；供后续翻页取更多结果）
    next_token: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None


def _source_error(error: Exception) -> SourceError:
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    http_status_error = getattr(httpx, "HTTPStatusError", ())
    timeout_errors = _httpx_timeout_types()
    network_errors = _httpx_network_types()
    if status_code is not None or (http_status_error and isinstance(error, http_status_error)):
        if status_code in {401, 403}:
            category = "authentication"
        elif status_code == 429:
            category = "rate_limit"
        else:
            category = "http"
    elif timeout_errors and isinstance(error, timeout_errors):
        category = "timeout"
    elif (network_errors and isinstance(error, network_errors)) or isinstance(
        error, (ConnectionError, OSError)
    ):
        category = "network"
    elif isinstance(error, (ValueError, KeyError, TypeError, ET.ParseError)):
        category = "response_parsing"
    else:
        category = "request"
    retryable = category in {"timeout", "rate_limit", "network"} or bool(
        status_code and status_code >= 500
    )
    return SourceError(category, status_code, retryable, str(error)[:240])


def normalize_doi(value: str | None) -> str:
    doi = (value or "").strip().casefold()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi.strip()


def _normalize_openalex_work(w: dict) -> dict:
    """OpenAlex work → 统一 paper dict（供检索与引文图共用）。"""
    authors = [
        {"name": a.get("author", {}).get("display_name", "")}
        for a in (w.get("authorships") or [])
    ]
    source = ((w.get("primary_location") or {}).get("source") or {})
    return {
        "id": (w.get("id") or "").split("/")[-1],
        "source": "openalex",
        "title": w.get("title") or "",
        "abstract": _flatten_abstract(w.get("abstract_inverted_index")) or "",
        "year": w.get("publication_year"),
        "citation_count": w.get("cited_by_count", 0) or 0,
        "venue": source.get("display_name") or "",
        "authors": authors,
        "doi": normalize_doi(w.get("doi")),
        "url": w.get("id"),
        "openalex_id": (w.get("id") or "").split("/")[-1],
    }


def _flatten_abstract(inverted_index: dict | None) -> str | None:
    if not inverted_index:
        return None
    words = sorted(
        (position, word)
        for word, positions in inverted_index.items()
        for position in positions
    )
    return " ".join(w for _, w in words)


class OpenAlexClient:
    def __init__(self):
        headers = {"Accept": "application/json"}
        if OPENALEX_EMAIL:
            headers["User-Agent"] = f"ResearchPaperHub/4.0 (mailto:{OPENALEX_EMAIL})"
        self.client = httpx.AsyncClient(base_url="https://api.openalex.org", headers=headers, timeout=10.0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.client.aclose()

    async def aclose(self):
        await self.client.aclose()

    async def search(self, query: str, limit: int = 20, page: int = 1,
                     offset: int = 0, filter_abstract: bool = False,
                     scope_title_abstract: bool = False) -> SourceBatch:
        try:
            params = {"per_page": limit, "page": page}
            if scope_title_abstract:
                # 严格限定在标题与摘要内检索
                params["filter"] = f"title_and_abstract.search:{query}"
            else:
                params["search"] = query

            if filter_abstract:
                current_filter = params.get("filter", "")
                add_on = "has_abstract:true,type:article"
                params["filter"] = f"{current_filter},{add_on}" if current_filter else add_on

            r = await self.client.get("/works", params=params)
            r.raise_for_status()
            data = r.json()
            results = []
            for w in data.get("results", []):
                authors = [{"name": a.get("author", {}).get("display_name", "")}
                          for a in w.get("authorships", [])]
                openalex_id = w.get("id", "").split("/")[-1]
                results.append({
                    "id": openalex_id,
                    "source": "openalex",
                    "title": w.get("title", ""),
                    "abstract": _flatten_abstract(w.get("abstract_inverted_index")) or "",
                    "year": w.get("publication_year"),
                    "citation_count": w.get("cited_by_count", 0),
                    "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                    "authors": authors,
                    "doi": normalize_doi(w.get("doi")),
                    "url": w.get("id"),
                    "openalex_id": openalex_id,
                })
            return SourceBatch(
                "openalex", tuple(results), data.get("meta", {}).get("count", 0), None
            )
        except Exception as error:
            return SourceBatch("openalex", (), 0, _source_error(error))

    async def get_work(self, work_id: str) -> dict | None:
        try:
            r = await self.client.get(f"/works/{work_id}")
            r.raise_for_status()
            w = r.json()
            authors = [{"name": a.get("author", {}).get("display_name", "")}
                      for a in w.get("authorships", [])]
            return {
                "id": w.get("id", "").split("/")[-1],
                "title": w.get("title", ""),
                "abstract": _flatten_abstract(w.get("abstract_inverted_index")),
                "year": w.get("publication_year"),
                "citation_count": w.get("cited_by_count", 0),
                "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                "authors": authors,
                "doi": w.get("doi"),
                "url": w.get("id"),
            }
        except Exception:
            return None

    async def get_cited_by(self, work_id: str, limit: int = 20, page: int = 1) -> dict:
        try:
            r = await self.client.get("/works", params={"filter": f"cites:{work_id}", "per_page": limit, "page": page})
            r.raise_for_status()
            data = r.json()
            results = []
            for w in data.get("results", []):
                results.append({"id": w.get("id", "").split("/")[-1], "title": w.get("title", ""),
                                "year": w.get("publication_year"), "citation_count": w.get("cited_by_count", 0)})
            return {"results": results, "total": data.get("meta", {}).get("count", 0)}
        except Exception:
            return {"results": [], "total": 0}

    async def get_references(self, work_id: str, limit: int = 20, page: int = 1) -> dict:
        try:
            r = await self.client.get("/works", params={"filter": f"referenced_works:{work_id}", "per_page": limit, "page": page})
            r.raise_for_status()
            data = r.json()
            results = []
            for w in data.get("results", []):
                results.append({"id": w.get("id", "").split("/")[-1], "title": w.get("title", ""),
                                "year": w.get("publication_year")})
            return {"results": results, "total": data.get("meta", {}).get("count", 0)}
        except Exception:
            return {"results": [], "total": 0}

    async def get_author(self, author_id: str) -> dict | None:
        try:
            r = await self.client.get(f"/authors/{author_id}")
            r.raise_for_status()
            a = r.json()
            return {"id": a.get("id", "").split("/")[-1], "name": a.get("display_name", ""),
                    "works_count": a.get("works_count", 0), "cited_by_count": a.get("cited_by_count", 0)}
        except Exception:
            return None

    # === 引文图通道（成本远低于关键词检索）===
    # OpenAlex 关键词检索每次消耗 10 积分（每日 1000 积分 ≈ 100 次调用），
    # 但引文图查询实测仅 1 积分/次、单篇查询 0 积分，
    # 因此引文扩展是配额受限时性价比最高的召回方式。

    async def get_citing_works(self, work_id: str, limit: int = 200,
                               sort_by_citations: bool = True) -> SourceBatch:
        """向前引文：引用该论文的后续工作（1 积分/次）。

        ``sort_by_citations``：按引用量降序返回。当一篇论文有数千篇引用者而我们
        只需要 200 篇时，**排序决定了截断质量**——降序能保留最有影响力的引用者，
        而非任意子集。此为 findpapers 的成熟做法。
        """
        wid = str(work_id or "").strip().split("/")[-1]
        if not wid:
            return SourceBatch("openalex", (), 0, None)
        try:
            params = {
                "filter": f"cites:{wid}",
                "per_page": max(1, min(int(limit), 200)),
                "select": "id,doi,title,abstract_inverted_index,publication_year,"
                          "cited_by_count,primary_location,authorships,type",
            }
            if sort_by_citations:
                params["sort"] = "cited_by_count:desc"
            r = await self.client.get("/works", params=params)
            r.raise_for_status()
            data = r.json()
            results = [_normalize_openalex_work(w) for w in data.get("results", []) if isinstance(w, dict)]
            total = (data.get("meta") or {}).get("count", len(results))
            return SourceBatch("openalex", tuple(results), int(total or len(results)), None)
        except Exception as error:
            return SourceBatch("openalex", (), 0, _source_error(error))

    async def hydrate_works(self, work_ids: Sequence[str], limit: int = 50) -> SourceBatch:
        """批量补全论文记录（单次调用最多 50 篇，1 积分）。

        实测：``filter=openalex_id:A|B|C`` 管道 OR 可行（DOI 的 OR 不受支持）。
        """
        ids = []
        for value in work_ids:
            wid = str(value or "").strip().split("/")[-1]
            if wid and wid not in ids:
                ids.append(wid)
        if not ids:
            return SourceBatch("openalex", (), 0, None)
        chunk = ids[:max(1, min(int(limit), 50))]
        try:
            r = await self.client.get("/works", params={
                "filter": "openalex_id:" + "|".join(chunk),
                "per_page": len(chunk),
                "select": "id,doi,title,abstract_inverted_index,publication_year,"
                          "cited_by_count,primary_location,authorships,type",
            })
            r.raise_for_status()
            data = r.json()
            results = [_normalize_openalex_work(w) for w in data.get("results", []) if isinstance(w, dict)]
            return SourceBatch("openalex", tuple(results), len(results), None)
        except Exception as error:
            return SourceBatch("openalex", (), 0, _source_error(error))

    async def get_referenced_works(self, work_id: str, limit: int = 200) -> SourceBatch:
        """向后引文：该论文引用过的奠基性工作（1 积分取 id + 1 积分批量补全）。"""
        wid = str(work_id or "").strip().split("/")[-1]
        if not wid:
            return SourceBatch("openalex", (), 0, None)
        try:
            r = await self.client.get(f"/works/{wid}", params={"select": "referenced_works"})
            r.raise_for_status()
            refs = (r.json().get("referenced_works") or [])[:max(1, min(int(limit), 200))]
            if not refs:
                return SourceBatch("openalex", (), 0, None)
            return await self.hydrate_works(refs, limit=50)
        except Exception as error:
            return SourceBatch("openalex", (), 0, _source_error(error))


# ============================================================
# Semantic Scholar Client
# ============================================================
_PAPER_FIELDS = "title,authors,year,abstract,citationCount,externalIds,url,publicationVenue"


def normalize_semantic_paper(raw: dict) -> dict:
    paper_id = raw.get("paperId", "")
    external_ids = raw.get("externalIds") or {}
    publication_venue = raw.get("publicationVenue") or {}
    doi = normalize_doi(external_ids.get("DOI"))
    return {
        "id": paper_id,
        "source": "semantic_scholar",
        "title": raw.get("title", "") or "",
        "abstract": raw.get("abstract", "") or "",
        "year": raw.get("year"),
        "citation_count": raw.get("citationCount", 0) or 0,
        "venue": publication_venue.get("name", "") or "",
        "authors": [
            {"name": author.get("name", "")}
            for author in (raw.get("authors") or [])
        ],
        "doi": doi,
        "url": raw.get("url", "") or "",
        "semantic_id": paper_id,
    }


class SemanticScholarClient:
    def __init__(self):
        headers = {"Accept": "application/json"}
        if SEMANTIC_SCHOLAR_API_KEY:
            headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
        self.client = httpx.AsyncClient(base_url="https://api.semanticscholar.org", headers=headers, timeout=10.0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.client.aclose()

    async def aclose(self):
        await self.client.aclose()

    async def search(self, query: str, limit: int = 20, page: int = 1,
                     offset: int = 0) -> SourceBatch:
        try:
            r = await self.client.get("/graph/v1/paper/search", params={"query": query, "limit": limit, "offset": offset, "fields": _PAPER_FIELDS})
            r.raise_for_status()
            data = r.json()
            papers = tuple(normalize_semantic_paper(raw) for raw in data.get("data", []))
            return SourceBatch("semantic_scholar", papers, data.get("total", len(papers)), None)
        except Exception as error:
            return SourceBatch("semantic_scholar", (), 0, _source_error(error))

    async def get_paper(self, paper_id: str) -> dict | None:
        try:
            r = await self.client.get(f"/graph/v1/paper/{paper_id}", params={"fields": _PAPER_FIELDS})
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    async def get_citations(self, paper_id: str, limit: int = 50) -> list[dict]:
        try:
            r = await self.client.get(f"/graph/v1/paper/{paper_id}/citations", params={"limit": limit, "fields": "title,authors,year,citationCount"})
            r.raise_for_status()
            data = r.json()
            return [e["citingPaper"] for e in data.get("data", []) if "citingPaper" in e]
        except Exception:
            return []

    async def get_citation_neighbors(self, paper_id: str, limit: int = 50,
                                     direction: str = "citations") -> list[dict]:
        """按 DOI / S2 id 取引文邻居，**直接返回带摘要的论文**。

        与 Europe PMC 不同，S2 的 citations/references 端点可同时返回 abstract，
        因此无需二次补全。实测该库对本领域的半导体器件文献覆盖优于 Europe PMC
        （Europe PMC 常返回 0 条，且这些论文多无 pmid）。
        """
        if direction not in {"citations", "references"}:
            return []
        try:
            r = await self.client.get(
                f"/graph/v1/paper/{paper_id}/{direction}",
                params={"limit": max(1, min(int(limit), 500)), "fields": _PAPER_FIELDS},
            )
            r.raise_for_status()
            rows = r.json().get("data") or []
            key = "citingPaper" if direction == "citations" else "citedPaper"
            out = []
            for entry in rows:
                raw = entry.get(key) if isinstance(entry, dict) else None
                if isinstance(raw, dict) and raw.get("title"):
                    out.append(normalize_semantic_paper(raw))
            return out
        except Exception:
            return []

    async def get_references(self, paper_id: str, limit: int = 50) -> list[dict]:
        try:
            r = await self.client.get(f"/graph/v1/paper/{paper_id}/references", params={"limit": limit, "fields": "title,authors,year,citationCount"})
            r.raise_for_status()
            data = r.json()
            return [e["citedPaper"] for e in data.get("data", []) if "citedPaper" in e]
        except Exception:
            return []

    async def get_similar(self, paper_id: str, limit: int = 10) -> list[dict]:
        try:
            r = await self.client.get(f"/graph/v1/paper/{paper_id}/similar", params={"limit": limit})
            r.raise_for_status()
            data = r.json()
            return data.get("papers", [])
        except Exception:
            return []

    async def search_bulk(self, query: str, limit: int = 200,
                          token: str = "") -> SourceBatch:
        """S2 bulk 检索端点（独立于 /search 的配额）。

        实测：单次调用返回 337 篇（/search 十次成功调用都拿不到这么多），
        官方上限 1000 篇/次。用于宽召回层，显著降低请求次数与限流风险。
        注意：bulk 不带相关性排序，需由本地重排处理。
        """
        try:
            page_size = max(1, min(int(limit), 1000))
            params = {
                "query": query,
                "fields": _PAPER_FIELDS,
            }
            if token:
                params["token"] = token
            r = await self.client.get("/graph/v1/paper/search/bulk", params=params)
            r.raise_for_status()
            data = r.json()
            rows = data.get("data") or []
            # bulk 可能返回 dict（按年份分组）或 list，统一处理
            if isinstance(rows, dict):
                flat = []
                for value in rows.values():
                    if isinstance(value, list):
                        flat.extend(value)
                rows = flat
            papers = tuple(
                normalize_semantic_paper(raw)
                for raw in rows[:page_size]
                if isinstance(raw, dict)
            )
            total = data.get("total")
            try:
                total = int(total) if total is not None else len(papers)
            except (TypeError, ValueError):
                total = len(papers)
            return SourceBatch("semantic_scholar", papers, total, None,
                               next_token=data.get("token") or "")
        except Exception as error:
            return SourceBatch("semantic_scholar", (), 0, _source_error(error))


# ============================================================
# LLM Provider Configurations
# ============================================================
LLM_PROVIDERS = {
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-v4-flash", "deepseek-v4-pro"],
        "default_model": "deepseek-v4-flash",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "default_model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
    },
    "claude": {
        "name": "Claude (Anthropic)",
        "base_url": "https://api.anthropic.com/v1",
        "models": ["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022"],
        "default_model": "claude-sonnet-4-20250514",
        "env_key": "ANTHROPIC_API_KEY",
        "api_format": "anthropic",
    },
    "qwen": {
        "name": "通义千问 (阿里)",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": ["qwen-plus", "qwen-turbo", "qwen-max", "qwen-long"],
        "default_model": "qwen-plus",
        "env_key": "DASHSCOPE_API_KEY",
    },
    "glm": {
        "name": "智谱清言 (GLM)",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "models": ["glm-4.7-flash", "glm-4.7"],
        "default_model": "glm-4.7-flash",
        "env_key": "ZHIPUAI_API_KEY",
    },
    "ollama": {
        "name": "Ollama (本地)",
        "base_url": "http://localhost:11434/v1",
        "models": ["llama3", "qwen2.5", "deepseek-v2", "mistral"],
        "default_model": "llama3",
        "env_key": "",
    },
}


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str = ""

    @classmethod
    def default(cls) -> "LLMConfig":
        return cls(DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL, "")


class LLMRequestError(RuntimeError):
    def __init__(self, provider: str, category: str, detail: str):
        super().__init__(detail)
        self.provider = provider
        self.category = category


def _llm_error_category(error: Exception) -> str:
    # 已是 LLMRequestError 时保留其分类（如 rate_limit / empty_response），
    # 否则重试逻辑会误判为不可重试。
    if isinstance(error, LLMRequestError):
        return error.category
    timeout_errors = _httpx_timeout_types()
    network_errors = _httpx_network_types()
    http_status_error = getattr(httpx, "HTTPStatusError", ())
    if timeout_errors and isinstance(error, timeout_errors):
        return "timeout"
    if (network_errors and isinstance(error, network_errors)) or isinstance(
        error, (ConnectionError, OSError)
    ):
        return "network"
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if status is not None or (http_status_error and isinstance(error, http_status_error)):
        if status in {401, 403}:
            return "authentication"
        if status == 429:
            return "rate_limit"
        return "http"
    if isinstance(error, (KeyError, ValueError, TypeError)):
        return "response_parsing"
    return "request"


# ============================================================
# Generic LLM Client
# ============================================================
class LLMClient:
    def __init__(self, provider: str = DEFAULT_LLM_PROVIDER, api_key: str = "", model: str = "",
                 thinking: bool = True):
        cfg = LLM_PROVIDERS.get(provider, LLM_PROVIDERS[DEFAULT_LLM_PROVIDER])
        self.provider = provider if provider in LLM_PROVIDERS else DEFAULT_LLM_PROVIDER
        self.model = model or cfg["default_model"]
        self.base_url = cfg["base_url"]
        self.api_format = cfg.get("api_format", "openai")
        # 是否允许隐藏推理。推理型模型在结构化抽取任务上会生成大量
        # reasoning token（实测 1729 字符/次），拖慢 2.35 倍却对结果无益。
        # 调用方可按任务类型覆盖：抽取/分类关闭，复杂综述生成保持开启。
        self.thinking = thinking

        # Priority: explicit api_key > env var > config default
        if not api_key:
            env_key = cfg.get("env_key", "")
            api_key = os.getenv(env_key, "") if env_key else ""

        self.api_key = api_key
        self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                if self.api_format == "anthropic":
                    headers["x-api-key"] = self.api_key
                    headers["anthropic-version"] = "2023-06-01"
                else:
                    headers["Authorization"] = f"Bearer {self.api_key}"
            # trust_env=False: 绕过系统代理(127.0.0.1:7890)，国内 API 直连更快更稳
            self._client = httpx.AsyncClient(
                base_url=self.base_url, headers=headers,
                timeout=httpx.Timeout(60.0),
                trust_env=False,
            )
        return self._client

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    async def aclose(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def complete_strict(self, messages: list[dict], temperature: float = 0.3,
                              response_format: dict | None = None,
                              thinking: bool | None = None,
                              max_tokens: int | None = None) -> str:
        """严格补全：返回纯文本内容，失败抛 LLMRequestError。

        ``thinking``：是否允许模型进行隐藏推理（默认沿用构造参数）。
        证据抽取这类**结构化分类**任务不需要推理链，关掉后实测快 2.35 倍
        且输出更短（651→241 completion tokens），统计结果不受影响。

        ``max_tokens``：输出 token 上限。结构化抽取设为 ~700 足以容纳全部
        18 个字段（实测精简 schema 仅需 142），同时能在模型跑偏时尽早截断，
        避免无谓等待——这是"约束输出格式以减少无效 token"在远程 API 上的
        可行落法（DeepSeek 不支持 json_schema 解码期约束）。
        """
        client = self._get_client()
        allow_thinking = self.thinking if thinking is None else thinking
        max_attempts = 4
        for attempt in range(max_attempts):
            try:
                if self.api_format == "anthropic":
                    system_msg = ""
                    user_messages = []
                    for message in messages:
                        if message["role"] == "system":
                            system_msg = message["content"]
                        else:
                            user_messages.append(message)
                    payload = {
                        "model": self.model,
                        "max_tokens": max_tokens or 4096,
                        "temperature": temperature,
                        "messages": user_messages,
                    }
                    if system_msg:
                        payload["system"] = system_msg
                    response = await client.post("/messages", json=payload)
                    response.raise_for_status()
                    return response.json()["content"][0]["text"].strip()

                endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    # 默认 8192：大 prompt + 推理模式下预算不足会导致空响应。
                    # 调用方可通过 max_tokens 收紧（如结构化抽取用 700）。
                    "max_tokens": max_tokens or 8192,
                }
                # 显式关闭隐藏推理：DeepSeek 推理型模型（v4 系列）默认会先生成
                # 大量 reasoning token 再输出，对"读摘要→填 JSON"这类任务纯属浪费。
                # 实测：开启时 3.50s/次、651 completion tokens；
                #       关闭后 1.49s/次、241 tokens，且 stance 判定准确率不变。
                if not allow_thinking:
                    payload["thinking"] = {"type": "disabled"}
                if response_format:
                    payload["response_format"] = response_format
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
                data = response.json()
                choice = (data.get("choices") or [{}])[0]
                content = ((choice.get("message") or {}).get("content") or "").strip()
                if not content:
                    # 空内容不是有效结果：可能因推理占满预算被截断，需重试
                    finish = choice.get("finish_reason")
                    raise LLMRequestError(
                        self.provider,
                        "rate_limit" if finish == "length" else "empty_response",
                        f"模型返回空内容 (finish_reason={finish})",
                    )
                return content
            except Exception as error:
                category = _llm_error_category(error)
                retryable = category in {"rate_limit", "empty_response", "timeout", "network"}
                if retryable and attempt < max_attempts - 1:
                    # 指数退避 + 基础等待，应对限流与推理占满预算被截断的情况
                    wait_time = 2.0 * (attempt + 1)
                    await _asyncio.sleep(wait_time)
                    continue
                raise LLMRequestError(
                    self.provider,
                    category,
                    str(error),
                ) from error
        raise LLMRequestError(self.provider, "rate_limit", "重试次数用尽")

    async def _complete(self, messages: list[dict], temperature: float = 0.3) -> str:
        try:
            return await self.complete_strict(messages, temperature)
        except LLMRequestError as error:
            return f"[AI 调用失败：{error}]"

    async def summarize(self, text: str) -> str:
        prompt = ("请用中文总结以下学术论文的核心内容，按以下格式输出：\n"
                  "### 研究目的\n...\n### 方法\n...\n### 主要结果\n...\n### 结论\n...")
        return await self._complete([{"role": "system", "content": "你是专业的学术论文分析助手。"},
                                      {"role": "user", "content": f"{prompt}\n\n论文内容：\n{text[:8000]}"}])

    async def chat(self, messages: list[dict], context: str = "") -> str:
        system_msg = {"role": "system", "content": "你是专业的学术论文分析助手。请基于提供的论文内容回答问题。如果答案不在论文中，请如实说明。" +
                      (f"\n\n论文内容：\n{context[:6000]}" if context else "")}
        return await self._complete([system_msg] + messages, temperature=0.5)

    async def extract_key_points(self, text: str) -> list[dict]:
        prompt = ("从以下论文内容中提取3-5个关键发现点，每点用一两句话概括。\n"
                  "用JSON数组格式返回，每个元素包含 category (可选值: method/result/conclusion/general) 和 content 字段。\n"
                  "只返回JSON数组，不要其他内容。")
        result = await self._complete([{"role": "system", "content": "你是专业的学术论文分析助手，只返回JSON格式。"},
                                        {"role": "user", "content": f"{prompt}\n\n论文内容：\n{text[:6000]}"}])
        import json
        try:
            start, end = result.find("["), result.rfind("]") + 1
            if start >= 0 and end > start:
                return json.loads(result[start:end])
        except (json.JSONDecodeError, ValueError):
            pass
        return [{"category": "general", "content": result}]

    async def compare_papers(self, papers: list[dict]) -> str:
        paper_texts = "\n\n---\n\n".join(
            f"论文：{p.get('title', '无标题')}\n{p.get('abstract', p.get('text', '无内容'))[:3000]}" for p in papers)
        prompt = ("请对比分析以下论文，用中文输出对比表格：\n"
                  "| 维度 | 论文1 | 论文2 | ... |\n|------|-------|-------|-----|\n"
                  "| 研究方法 | ... | ... | ... |\n| 数据集 | ... | ... | ... |\n"
                  "| 主要结果 | ... | ... | ... |\n| 创新点 | ... | ... | ... |\n| 局限 | ... | ... | ... |\n\n表格后附简要分析。")
        return await self._complete([{"role": "system", "content": "你是专业的学术论文对比分析助手。"},
                                      {"role": "user", "content": f"{prompt}\n\n{paper_texts}"}])

    async def translate_query(self, chinese_query: str) -> str:
        prompt = (
            "将以下中文学术搜索查询翻译成英文学术搜索关键词。"
            "保留原有的英文术语（如 GaAs、HBT 等），只翻译中文部分。"
            "输出纯英文关键词，用空格分隔，不要任何解释。\n\n"
            f"中文查询：{chinese_query}"
        )
        return (await self.complete_strict([
            {"role": "system", "content": "你是学术搜索翻译助手，只输出翻译后的英文关键词。"},
            {"role": "user", "content": prompt},
        ], temperature=0.1)).strip()

    async def expand_query(self, query: str) -> list[str]:
        prompt = (
            "将以下学术搜索查询扩展为3-5个同义或相关的英文搜索关键词短语。"
            "每个一行。保留专业术语和缩写。"
            "输出纯英文关键词，不要编号、不要解释、不要用 markdown。\n\n"
            f"查询：{query}"
        )
        result = await self.complete_strict([
            {"role": "system", "content": "你是学术搜索专家，只输出搜索关键词，每行一个。"},
            {"role": "user", "content": prompt},
        ], temperature=0.3)
        terms = [line.strip() for line in result.split("\n")
                 if line.strip() and not line.startswith("#") and len(line.strip()) > 3]
        return terms[:5] if terms else [query]


# Keep DeepSeekClient as alias for backward compatibility
def DeepSeekClient():
    return LLMClient("deepseek")


# ============================================================
# PDF Processor
# ============================================================
class PDFProcessor:
    def extract_text(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()

    def extract_metadata(self, pdf_path: str) -> dict:
        doc = fitz.open(pdf_path)
        meta = doc.metadata
        pages = doc.page_count
        doc.close()
        return {"title": meta.get("title", ""), "author": meta.get("author", ""),
                "subject": meta.get("subject", ""), "pages": pages}

    def save_pdf(self, file_content: bytes, paper_id: int) -> str:
        path = os.path.join(PDF_DIR, f"{paper_id}.pdf")
        with open(path, "wb") as f:
            f.write(file_content)
        return path


# ============================================================
# Embedding Service (lazy singleton)
# ============================================================
class EmbeddingService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
        return cls._instance

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    def embed_text(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(384, dtype=np.float32)
        return self.model.encode(text, normalize_embeddings=True)

    def compute_similarities(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        q = self.embed_text(query)
        c = self.model.encode(texts, normalize_embeddings=True)
        return np.dot(c, q).tolist()


# ============================================================
# arXiv Client
# ============================================================
import xml.etree.ElementTree as ET

_ARXIV_NS = "http://www.w3.org/2005/Atom"
_OPENSEARCH_NS = "http://a9.com/-/spec/opensearch/1.1/"


class ArxivClient:
    def __init__(self):
        self.client = httpx.AsyncClient(base_url="https://export.arxiv.org", timeout=15.0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.client.aclose()

    async def aclose(self):
        await self.client.aclose()

    async def search(self, query: str, limit: int = 20, page: int = 1,
                     offset: int = 0, max_results: int | None = None) -> SourceBatch:
        try:
            result_limit = max_results or limit
            start = offset or max(0, page - 1) * result_limit
            r = await self.client.get("/api/query", params={
                "search_query": f"all:{query}",
                "start": start, "max_results": result_limit,
                "sortBy": "relevance",
            })
            r.raise_for_status()
            root = ET.fromstring(r.text)
            papers = []
            for entry in root.findall(f"{{{_ARXIV_NS}}}entry"):
                title_el = entry.find(f"{{{_ARXIV_NS}}}title")
                summary_el = entry.find(f"{{{_ARXIV_NS}}}summary")
                published_el = entry.find(f"{{{_ARXIV_NS}}}published")
                id_el = entry.find(f"{{{_ARXIV_NS}}}id")

                title = title_el.text.strip() if title_el is not None and title_el.text else ""
                abstract = summary_el.text.strip() if summary_el is not None and summary_el.text else ""
                year = int(published_el.text[:4]) if published_el is not None and published_el.text else None
                arx_id = id_el.text.strip().split("/abs/")[-1] if id_el is not None and id_el.text else ""

                authors = []
                for a_el in entry.findall(f"{{{_ARXIV_NS}}}author"):
                    name_el = a_el.find(f"{{{_ARXIV_NS}}}name")
                    if name_el is not None and name_el.text:
                        authors.append({"name": name_el.text.strip()})

                papers.append({
                    "id": f"arxiv:{arx_id}",
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "citation_count": 0,
                    "venue": "arXiv",
                    "authors": authors,
                    "url": f"https://arxiv.org/abs/{arx_id}",
                    "source": "arxiv",
                    "doi": "",
                    "arxiv_id": arx_id,
                })
            total_element = root.find(f"{{{_OPENSEARCH_NS}}}totalResults")
            try:
                total = int(total_element.text) if total_element is not None else len(papers)
            except (TypeError, ValueError):
                total = len(papers)
            return SourceBatch("arxiv", tuple(papers), total, None)
        except Exception as error:
            return SourceBatch("arxiv", (), 0, _source_error(error))


import xml.etree.ElementTree as ET2

# ============================================================
# Crossref Client
# ============================================================
class CrossrefClient:
    def __init__(self):
        self.client = httpx.AsyncClient(
            base_url="https://api.crossref.org",
            headers={"User-Agent": "ResearchPaperHub/4.0 (mailto:research@example.com)"},
            timeout=15.0,
        )

    async def __aenter__(self): return self
    async def __aexit__(self, *args): await self.client.aclose()
    async def aclose(self): await self.client.aclose()

    async def search(self, query: str, limit: int = 20, page: int = 1,
                     offset: int = 0) -> SourceBatch:
        try:
            r = await self.client.get("/works", params={
                "query": query,
                "rows": min(limit, 100),
                "offset": offset or max(0, page - 1) * min(limit, 100),
            })
            r.raise_for_status()
            data = r.json()
            items = data.get("message", {}).get("items", [])
            results = []
            for w in items:
                doi = normalize_doi(w.get("DOI", ""))
                publication = (
                    w.get("published")
                    or w.get("published-print")
                    or w.get("published-online")
                    or w.get("issued")
                    or w.get("created")
                    or {}
                )
                date_parts = publication.get("date-parts", [[None]]) or [[None]]
                year = (date_parts[0] or [None])[0] if date_parts else None
                authors = [{"name": f"{(a.get('given',''))} {(a.get('family',''))}".strip()}
                          for a in w.get("author", [])]
                results.append({
                    "id": doi,
                    "source": "crossref",
                    "title": (w.get("title") or [""])[0],
                    "abstract": w.get("abstract", ""),
                    "year": year,
                    "citation_count": w.get("is-referenced-by-count", 0),
                    "venue": (w.get("container-title") or [""])[0] if w.get("container-title") else "",
                    "authors": authors,
                    "doi": doi,
                    "url": f"https://doi.org/{doi}" if doi else "",
                })
            return SourceBatch(
                "crossref", tuple(results), data.get("message", {}).get("total-results", 0), None
            )
        except Exception as error:
            return SourceBatch("crossref", (), 0, _source_error(error))


# ============================================================
# PubMed Client
# ============================================================
def normalize_pubmed_summary(pmid: str, raw: dict) -> dict:
    publication = raw.get("pubdate", "") or ""
    match = re.search(r"\b(19|20)\d{2}\b", publication)
    elocation = raw.get("elocationid", "") or ""
    return {
        "id": f"pmid:{pmid}",
        "source": "pubmed",
        "title": raw.get("title", "") or "",
        "abstract": "",
        "year": int(match.group(0)) if match else None,
        "citation_count": 0,
        "venue": raw.get("source", "") or "",
        "authors": [
            {"name": author.get("name", "")}
            for author in (raw.get("authors") or [])
        ],
        "doi": normalize_doi(elocation) if "doi" in elocation.casefold() else "",
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "pmid": pmid,
    }


def _parse_pubmed_abstracts(xml_text: str) -> dict[str, str]:
    abstracts = {}
    root = ET.fromstring(xml_text)
    for article in root.findall(".//PubmedArticle"):
        pmid_element = article.find(".//MedlineCitation/PMID")
        if pmid_element is None or not pmid_element.text:
            continue
        sections = []
        for abstract_element in article.findall(".//Article/Abstract/AbstractText"):
            text = "".join(abstract_element.itertext()).strip()
            if not text:
                continue
            label = (abstract_element.get("Label") or "").strip()
            sections.append(f"{label}: {text}" if label else text)
        if sections:
            abstracts[pmid_element.text.strip()] = " ".join(sections)
    return abstracts


class PubMedClient:
    def __init__(self):
        self.client = httpx.AsyncClient(
            base_url="https://eutils.ncbi.nlm.nih.gov",
            timeout=15.0,
        )

    async def __aenter__(self): return self
    async def __aexit__(self, *args): await self.client.aclose()
    async def aclose(self): await self.client.aclose()

    async def search(self, query: str, limit: int = 20, page: int = 1,
                     offset: int = 0) -> SourceBatch:
        try:
            # Step 1: Search for PubMed IDs
            r = await self.client.get("/entrez/eutils/esearch.fcgi", params={
                "db": "pubmed", "term": query, "retmax": min(limit, 100),
                "retstart": offset or max(0, page - 1) * min(limit, 100),
                "retmode": "json", "sort": "relevance",
            })
            r.raise_for_status()
            search_result = r.json().get("esearchresult", {})
            id_list = search_result.get("idlist", [])
            try:
                total = int(search_result.get("count", len(id_list)))
            except (TypeError, ValueError):
                total = len(id_list)
            if not id_list:
                return SourceBatch("pubmed", (), 0, None)

            # Step 2: Fetch summaries
            r2 = await self.client.get("/entrez/eutils/esummary.fcgi", params={
                "db": "pubmed", "id": ",".join(id_list), "retmode": "json",
            })
            r2.raise_for_status()
            result_data = r2.json().get("result", {})
            results = [
                normalize_pubmed_summary(pid, result_data.get(pid, {}))
                for pid in id_list
                if result_data.get(pid)
            ]
            try:
                r3 = await self.client.get("/entrez/eutils/efetch.fcgi", params={
                    "db": "pubmed", "id": ",".join(id_list),
                    "retmode": "xml", "rettype": "abstract",
                })
                r3.raise_for_status()
                abstracts = _parse_pubmed_abstracts(r3.text)
                for paper in results:
                    paper["abstract"] = abstracts.get(paper["pmid"], "")
            except Exception:
                # Summary search remains useful when the optional abstract fetch fails.
                pass
            return SourceBatch("pubmed", tuple(results), total, None)
        except Exception as error:
            return SourceBatch("pubmed", (), 0, _source_error(error))


# ============================================================
# Google Scholar Client（可选源，默认关闭）
# ============================================================
# 重要声明与风险：
#   1. Google Scholar **没有官方 API**，此处为 HTML 解析，Google 可随时改动结构；
#   2. Google 服务条款禁止自动化抓取，使用者需自行评估合规风险；
#   3. 实测（2026-10）经代理访问 24 次请求成功率 96%，失败表现为 HTTP 429 +
#      "unusual traffic" 验证页，属**间歇性限流**而非永久封禁，等待后可恢复；
#   4. 必须经代理访问，直连会超时（国内网络环境）。
# 因此本客户端设计为：默认关闭、失败静默降级、绝不阻塞主检索流程。

GS_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

# 单次返回上限（Google Scholar 实际每页最多 10 条，即使 num=20）
GS_PAGE_SIZE = 10


def parse_google_scholar_meta(meta: str) -> tuple[str, str, int | None]:
    """解析 Google Scholar 的 ``div.gs_a`` 文本。

    格式为 ``authors - venue, year - publisher_domain``，例如::

        "C Monier, AG Baca, SZ Sun… - Applied physics …, 2002 - pubs.aip.org"

    注意：**年份必须用正则提取**——直接用 " - " 分割会把 venue 与 year 混在一起
    （它们以逗号相连），这也是 paper-search-mcp 等实现常见解析错误的来源。
    """
    text = " ".join(str(meta or "").split())
    parts = [p.strip() for p in text.split(" - ")]
    authors = parts[0] if parts else ""
    venue = parts[1] if len(parts) > 1 else ""
    year_match = re.search(r"\b(19|20)\d{2}\b", text)
    year = int(year_match.group(0)) if year_match else None
    # 从 venue 段剥离年份及其后的域名
    venue = re.sub(r",?\s*\b(19|20)\d{2}\b.*$", "", venue).strip().rstrip(",")
    return authors, venue, year


def parse_google_scholar_html(html: str) -> list[dict]:
    """从搜索结果页解析论文条目（无需 BeautifulSoup，纯正则）。

    选择器与 scholarly 保持一致：结果块 ``div.gs_r.gs_or.gs_scl``，
    标题 ``h3.gs_rt``，元信息 ``div.gs_a``，摘要片段 ``div.gs_rs``。
    """
    import re as _re
    if not html:
        return []

    papers: list[dict] = []
    blocks = _re.split(r'<div class="gs_r gs_or gs_scl"', html)[1:]
    for block in blocks:
        # 标题 + 链接（部分条目为纯引用，无 <a>）
        title_match = _re.search(
            r'<h3 class="gs_rt".*?<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            block, _re.S,
        )
        if title_match:
            url = title_match.group(1)
            raw_title = title_match.group(2)
        else:
            fallback = _re.search(r'<h3 class="gs_rt"[^>]*>(.*?)</h3>', block, _re.S)
            if not fallback:
                continue
            url = ""
            raw_title = fallback.group(1)

        # 去掉 [PDF] / [HTML] / [BOOK] 之类的类型标记
        raw_title = _re.sub(r'<span class="gs_ct[ac]"[^>]*>.*?</span>', " ", raw_title, flags=_re.S)
        title = _re.sub(r"<[^>]+>", "", raw_title)
        title = " ".join(title.split())
        if not title:
            continue

        meta_match = _re.search(r'<div class="gs_a">(.*?)</div>', block, _re.S)
        meta_text = _re.sub(r"<[^>]+>", "", meta_match.group(1)) if meta_match else ""
        authors_raw, venue, year = parse_google_scholar_meta(meta_text)

        abstract_match = _re.search(r'<div class="gs_rs">(.*?)</div>', block, _re.S)
        abstract = ""
        if abstract_match:
            abstract = _re.sub(r"<[^>]+>", "", abstract_match.group(1))
            abstract = " ".join(abstract.split())

        cite_match = _re.search(r"Cited by (\d+)", block)
        citation_count = int(cite_match.group(1)) if cite_match else 0

        authors = [
            {"name": name.strip()}
            for name in authors_raw.split(",")
            if name.strip() and "…" not in name
        ]

        papers.append({
            "id": f"gs:{abs(hash(title)) & 0xFFFFFFFF:x}",
            "source": "google_scholar",
            "title": title,
            "abstract": abstract,
            "year": year,
            "citation_count": citation_count,
            "venue": venue,
            "authors": authors,
            "doi": "",  # Google Scholar 不直接提供 DOI
            "url": url,
        })
    return papers


class GoogleScholarClient:
    """Google Scholar 抓取客户端（可选、默认关闭、需代理）。

    仅用于补充 OpenAlex/Europe PMC 未覆盖的文献，**不作为主力源**。
    """

    def __init__(self, proxy: str = "", timeout: float = 20.0):
        # 优先使用显式代理；否则交由 httpx 读取环境变量（trust_env=True）
        kwargs: dict = {"base_url": "https://scholar.google.com", "headers": GS_HEADERS,
                        "timeout": timeout}
        if proxy:
            kwargs["proxies"] = proxy
        else:
            kwargs["trust_env"] = True
        self.client = httpx.AsyncClient(**kwargs)

    async def __aenter__(self): return self
    async def __aexit__(self, *args): await self.client.aclose()
    async def aclose(self): await self.client.aclose()

    async def search(self, query: str, limit: int = 10, page: int = 1,
                     offset: int = 0, year_low: int | None = None,
                     year_high: int | None = None, title_only: bool = False
                     ) -> SourceBatch:
        """检索 Google Scholar。

        支持官方高级检索参数（实测 6/6 可用）：
        ``as_ylo``/``as_yhi``（年份范围）、``as_occt=title``（限定标题）。
        分页用 ``start``；注意实测相邻页存在少量重复条目，需依赖上层去重。
        """
        query = " ".join((query or "").split())
        if not query:
            return SourceBatch("google_scholar", (), 0, None)

        per_page = max(1, min(int(limit), GS_PAGE_SIZE))
        start = max(0, int(offset)) if offset else max(0, (int(page) - 1) * per_page)
        params: dict = {"q": query, "hl": "en", "num": per_page}
        if start:
            params["start"] = start
        if year_low:
            params["as_ylo"] = int(year_low)
        if year_high:
            params["as_yhi"] = int(year_high)
        if title_only:
            params["as_occt"] = "title"

        try:
            r = await self.client.get("/scholar", params=params)
            if r.status_code == 429:
                # 间歇性限流：明确归类为 rate_limit，交由上层退避或跳过
                return SourceBatch(
                    "google_scholar", (), 0,
                    SourceError("rate_limit", 429, True,
                                "Google Scholar 返回 429（unusual traffic / 验证码）"),
                )
            r.raise_for_status()
            papers = parse_google_scholar_html(r.text)
            if not papers and ("unusual traffic" in r.text.lower()
                               or "not a robot" in r.text.lower()):
                return SourceBatch(
                    "google_scholar", (), 0,
                    SourceError("rate_limit", 200, True, "Google Scholar 触发人机验证"),
                )
            return SourceBatch("google_scholar", tuple(papers), len(papers), None)
        except Exception as error:
            return SourceBatch("google_scholar", (), 0, _source_error(error))

    async def search_with_abstract(self, query: str, limit: int = 10, **kwargs
                                   ) -> SourceBatch:
        """仅返回带摘要的条目（Google Scholar 的摘要片段是本源的独特价值）。"""
        batch = await self.search(query, limit=limit, **kwargs)
        if not batch.ok:
            return batch
        papers = tuple(p for p in batch.papers if (p.get("abstract") or "").strip())
        return SourceBatch("google_scholar", papers, len(papers), None)


def normalize_europepmc_result(raw: dict) -> dict:
    """Europe PMC core result → 统一 paper dict。

    Europe PMC 单次可返回 pageSize=100 且实测无严格限流，字段丰富：
    abstractText / citedByCount / journalInfo / pubTypeList / inEPMC（全文可用）。
    """
    journal = ((raw.get("journalInfo") or {}).get("journal") or {})
    doi = normalize_doi(raw.get("doi"))
    pmid = raw.get("pmid") or ""
    pmcid = raw.get("pmcid") or ""
    year = raw.get("pubYear")
    try:
        year = int(year) if year not in (None, "") else None
    except (TypeError, ValueError):
        year = None
    identifier = pmid or pmcid or doi or raw.get("id", "")
    pub_types = ((raw.get("pubTypeList") or {}).get("pubType") or [])
    if isinstance(pub_types, str):
        pub_types = [pub_types]
    return {
        "id": f"epmc:{identifier}",
        "source": "europepmc",
        "title": raw.get("title", "") or "",
        "abstract": raw.get("abstractText", "") or "",
        "year": year,
        "citation_count": raw.get("citedByCount", 0) or 0,
        "venue": journal.get("title", "") or "",
        "authors": [
            {"name": (a.get("fullName") or a.get("collectiveName") or "").strip()}
            for a in ((raw.get("authorList") or {}).get("author") or [])
        ],
        "doi": doi,
        "url": (
            f"https://europepmc.org/article/MED/{pmid}" if pmid
            else f"https://europepmc.org/article/PMC/{pmcid}" if pmcid
            else ""
        ),
        "pmid": pmid,
        "pmcid": pmcid,
        # 具有全文（Europe PMC / PMC）时可用于 Methods/Results 抽取
        "has_full_text": (raw.get("inEPMC") == "Y") or bool(raw.get("isOpenAccess") == "Y"),
        "publication_types": [str(t) for t in pub_types],
    }


class EuropePMCClient:
    """Europe PMC REST 客户端（免 Key、高吞吐）。

    实测：8 次连续 pageSize=100 请求全部 200，无节流；
    是 OpenAlex 每日 100 次配额之外最有价值的补充召回源。
    """

    def __init__(self):
        # 注意：httpx 在请求路径以 "/" 开头时会**丢弃 base_url 的路径部分**，
        # 因此这里把完整路径并入 base_url，并以相对路径（无前导 "/"）发起请求。
        self.client = httpx.AsyncClient(
            base_url="https://www.ebi.ac.uk/europepmc/webservices/rest/",
            timeout=20.0,
        )

    async def __aenter__(self): return self
    async def __aexit__(self, *args): await self.client.aclose()
    async def aclose(self): await self.client.aclose()

    async def search(self, query: str, limit: int = 100, page: int = 1,
                     offset: int = 0) -> SourceBatch:
        try:
            page_size = max(1, min(int(limit), 100))
            current = max(1, int(page))
            params = {
                "query": query,
                "format": "json",
                "pageSize": page_size,
                "resultType": "core",
                # 引用量降序：优先高质量证据，配合本地质量精排
                "sort": "CITED desc",
            }
            if (offset or 0) > 0 and page_size:
                current = max(1, (int(offset) // page_size) + 1)
            if current > 1:
                params["page"] = current
            r = await self.client.get("search", params=params)
            r.raise_for_status()
            data = r.json()
            result_list = (data.get("resultList") or {}).get("result") or []
            results = [
                normalize_europepmc_result(item)
                for item in result_list
                if isinstance(item, dict)
            ]
            try:
                total = int(data.get("hitCount") or len(results))
            except (TypeError, ValueError):
                total = len(results)
            return SourceBatch("europepmc", tuple(results), total, None)
        except Exception as error:
            return SourceBatch("europepmc", (), 0, _source_error(error))

    async def get_citations(self, pmid: str, limit: int = 50) -> list[dict]:
        """向前引文：谁引用了这篇论文（免费、无配额）。

        返回的是**无摘要的引用存根**（title/pubYear/id/citedByCount），
        需配合 ``hydrate`` 补齐摘要后才能进入证据抽取。
        """
        try:
            r = await self.client.get(f"MED/{pmid}/citations",
                                      params={"format": "json", "pageSize": min(int(limit), 1000)})
            r.raise_for_status()
            data = r.json()
            return list((data.get("citationList") or {}).get("citation") or [])
        except Exception:
            return []

    async def get_references(self, pmid: str, limit: int = 50) -> list[dict]:
        """向后引文：这篇论文引用了谁（免费、无配额）。"""
        try:
            r = await self.client.get(f"MED/{pmid}/references",
                                      params={"format": "json", "pageSize": min(int(limit), 1000)})
            r.raise_for_status()
            data = r.json()
            return list((data.get("referenceList") or {}).get("reference") or [])
        except Exception:
            return []

    async def hydrate(self, ids: Sequence[str], limit: int = 50) -> SourceBatch:
        """把引文存根批量补全为带摘要的完整记录（单次调用完成）。

        实测：``EXT_ID:a OR b OR c`` 形式一次可返回多条 core 记录且含 abstractText，
        远优于逐条查询。
        """
        cleaned = [str(i).strip() for i in ids if str(i).strip()]
        if not cleaned:
            return SourceBatch("europepmc", (), 0, None)
        chunk = cleaned[:max(1, min(int(limit), 50))]
        try:
            query = " OR ".join(f"EXT_ID:{i}" for i in chunk)
            r = await self.client.get("search", params={
                "query": query, "format": "json",
                "pageSize": len(chunk), "resultType": "core",
            })
            r.raise_for_status()
            data = r.json()
            rows = ((data.get("resultList") or {}).get("result") or [])
            results = [normalize_europepmc_result(x) for x in rows if isinstance(x, dict)]
            return SourceBatch("europepmc", tuple(results), len(results), None)
        except Exception as error:
            return SourceBatch("europepmc", (), 0, _source_error(error))


# ============================================================
# Search Engine
# ============================================================
import re as _re
import asyncio as _asyncio

def _contains_chinese(text: str) -> bool:
    return bool(_re.search(r'[\u4e00-\u9fff]', text))


# 实测上限：超过此词数的检索式在 OpenAlex / Europe PMC / S2 上召回为 0
MAX_QUERY_WORDS = 12


@dataclass(frozen=True)
class QueryPlan:
    original: str
    variants: tuple[str, ...]
    translation_used: bool = False
    ai_error: str | None = None
    kind: str = "topic"


@dataclass(frozen=True)
class SearchResponse:
    papers: tuple[dict, ...]
    query_plan: QueryPlan
    sources: tuple[SourceBatch, ...]

    @property
    def partial(self) -> bool:
        return any(not batch.ok for batch in self.sources) and any(
            batch.ok or bool(batch.papers) for batch in self.sources
        )

    @property
    def all_failed(self) -> bool:
        return (
            bool(self.sources)
            and all(not batch.ok for batch in self.sources)
            and not any(batch.papers for batch in self.sources)
        )


def _query_kind(query: str) -> str:
    value = query.strip()
    lowered = value.lower()
    if re.fullmatch(r"(?:https?://(?:dx\.)?doi\.org/|doi:\s*)?10\.\d{4,9}/\S+", lowered):
        return "doi"
    if re.fullmatch(r"(?:arxiv:\s*)?\d{4}\.\d{4,5}(?:v\d+)?", lowered):
        return "arxiv"
    if re.fullmatch(r"pmid:\s*\d+", lowered):
        return "pmid"
    if re.fullmatch(r"W\d+", value, re.IGNORECASE):
        return "openalex"
    if len(value.split()) > 8:
        return "title"
    return "topic"


def _clean_query_variant(value: str) -> str | None:
    cleaned = " ".join((value or "").strip().split())
    if not cleaned or len(cleaned) > 200:
        return None
    if "AI 调用失败" in cleaned or cleaned.startswith("["):
        return None
    if re.match(r"^(?:\d+[.)]|[-*#])\s*", cleaned):
        return None
    return cleaned


class QueryPlanner:
    def __init__(self, llm_factory):
        self.llm_factory = llm_factory

    async def build(self, query: str) -> QueryPlan:
        original = " ".join(query.strip().split())
        if not original:
            return QueryPlan("", ())

        variants = [original]
        translation_used = False
        ai_error = None
        client = self.llm_factory()
        effective = original
        try:
            if _contains_chinese(original):
                translated = _clean_query_variant(await client.translate_query(original))
                if translated and translated.casefold() != original.casefold():
                    variants.append(translated)
                    effective = translated
                    translation_used = True
            for candidate in await client.expand_query(effective):
                cleaned = _clean_query_variant(candidate)
                if cleaned:
                    variants.append(cleaned)
        except LLMRequestError as error:
            ai_error = error.category
        except Exception:
            ai_error = "request"
        finally:
            close = getattr(client, "aclose", None)
            if close:
                await close()

        unique = []
        seen = set()
        for variant in variants:
            key = variant.casefold()
            if key not in seen:
                seen.add(key)
                unique.append(variant)
            if len(unique) == 4:
                break
        return QueryPlan(original, tuple(unique), translation_used, ai_error, _query_kind(original))


class SearchEngine:
    def __init__(self, llm_config: LLMConfig | None = None,
                 clients: dict[str, object] | None = None, query_planner=None,
                 retry_delays: tuple[float, ...] = (1.0,),
                 source_intervals: dict[str, float] | None = None,
                 source_request_caps: dict[str, int] | None = None):
        self.llm_config = llm_config or LLMConfig.default()
        self.retry_delays = retry_delays
        self.source_intervals = dict(source_intervals) if source_intervals is not None else {
            "arxiv": 1.0,
            "semantic_scholar": 1.05,
            "crossref": 0.25,
            "pubmed": 1.05,
            # Europe PMC 实测无严格限流（8/8 连续 100 篇请求通过）
            "europepmc": 0.2,
        }
        self.source_request_caps = (
            dict(source_request_caps)
            if source_request_caps is not None
            else {"crossref": 20}
        )
        self._source_last_request: dict[str, float] = {}
        self._clients_injected = clients is not None
        clients = dict(clients or {})
        self.clients = clients
        self._oa = None  # OpenAlex
        self._arx = None # arXiv
        self._ss = None  # Semantic Scholar
        self._cr = None  # Crossref
        self._pm = None  # PubMed
        self._epmc = None  # Europe PMC
        self._oa = clients.get("openalex")
        self._arx = clients.get("arxiv")
        self._ss = clients.get("semantic_scholar")
        self._cr = clients.get("crossref")
        self._pm = clients.get("pubmed")
        self._epmc = clients.get("europepmc")
        self.query_planner = query_planner or QueryPlanner(self._new_llm_client)

    def _new_llm_client(self):
        return LLMClient(
            provider=self.llm_config.provider,
            api_key=self.llm_config.api_key,
            model=self.llm_config.model,
        )

    async def _ensure_clients(self):
        if not self._clients_injected:
            if self._oa is None: self._oa = OpenAlexClient()
            if self._arx is None: self._arx = ArxivClient()
            if self._ss is None and SEMANTIC_SCHOLAR_API_KEY:
                self._ss = SemanticScholarClient()
            if self._cr is None: self._cr = CrossrefClient()
            if self._pm is None: self._pm = PubMedClient()
            if self._epmc is None: self._epmc = EuropePMCClient()
        self.clients = {
            name: client
            for name, client in {
                "openalex": self._oa,
                "arxiv": self._arx,
                "semantic_scholar": self._ss,
                "crossref": self._cr,
                "pubmed": self._pm,
                "europepmc": self._epmc,
            }.items()
            if client is not None
        }

    async def _translate_to_english(self, chinese_query: str) -> str:
        try:
            async with self._new_llm_client() as c:
                return await c.translate_query(chinese_query)
        except Exception:
            return chinese_query

    async def _expand_query(self, query: str) -> list[str]:
        try:
            async with self._new_llm_client() as c:
                return await c.expand_query(query)
        except Exception:
            return [query]

    async def search(self, query: str, limit: int = 20, source: str = "auto",
                     sort: str = "relevance") -> list[dict]:
        response = await self.search_with_diagnostics(query, limit, source, sort)
        return list(response.papers)

    async def search_with_diagnostics(self, query: str, limit: int = 20,
                                      source: str = "auto",
                                      sort: str = "relevance") -> SearchResponse:
        plan = await self.query_planner.build(query)
        if not plan.original or limit <= 0:
            return SearchResponse((), plan, ())

        await self._ensure_clients()
        selected_clients = self.clients
        if source != "auto":
            selected_clients = {
                source: self.clients[source]
            } if source in self.clients else {}

        variant_count = max(1, len(plan.variants))
        source_budget = max(20, min(200, limit * 2))
        per_variant_budget = max(
            10, (source_budget + variant_count - 1) // variant_count
        )
        request_limit = min(100, per_variant_budget)
        source_items = list(selected_clients.items())
        tasks = []
        for source_name, client in source_items:
            source_cap = self.source_request_caps.get(source_name, 100)
            source_request_limit = min(request_limit, source_cap)
            source_variant_budget = min(per_variant_budget, source_cap)
            # Semantic Scholar 限流严格（1 req/s），只发前2个变体
            if source_name == "semantic_scholar":
                variants_to_send = plan.variants[:2]
            else:
                variants_to_send = plan.variants
            tasks.append(self._fetch_source_variants(
                source_name,
                client,
                variants_to_send,
                source_request_limit,
                source_variant_budget,
            ))

        occurrences = []
        source_calls: dict[str, list[SourceBatch]] = {
            name: [] for name in selected_clients
        }
        try:
            gathered = await _asyncio.gather(*tasks, return_exceptions=True)
            for (source_name, _client), source_result in zip(source_items, gathered):
                if isinstance(source_result, Exception):
                    source_result = (
                        (0, SourceBatch(source_name, (), 0, _source_error(source_result))),
                    )
                for variant_index, result in source_result:
                    source_calls[source_name].append(result)
                    for rank, paper in enumerate(result.papers, start=1):
                        occurrences.append((source_name, variant_index, rank, paper))
        finally:
            await self._close_clients(selected_clients.values())

        source_batches = tuple(
            self._combine_source_batches(name, source_calls[name])
            for name in selected_clients
        )
        merged = self._merge_and_rank(plan, occurrences, sort)
        selected = self._select_diverse(merged, source_batches, limit)
        return SearchResponse(tuple(selected), plan, source_batches)

    async def _fetch_source_variants(self, source_name: str, client,
                                     variants: tuple[str, ...], request_limit: int,
                                     per_variant_budget: int):
        """Run sources in parallel while keeping each provider's calls serial."""
        results = []
        for variant_index, variant in enumerate(variants):
            try:
                batch = await self._fetch_source_query(
                    source_name,
                    client,
                    variant,
                    request_limit,
                    per_variant_budget,
                )
            except Exception as error:
                batch = SourceBatch(source_name, (), 0, _source_error(error))
            results.append((variant_index, batch))
            if batch.error and batch.error.category in {
                "authentication", "rate_limit", "network", "timeout"
            }:
                break
        return tuple(results)

    async def _request_source_batch(self, source_name: str, client, query: str,
                                    limit: int, page: int, offset: int) -> SourceBatch:
        async def request_once() -> SourceBatch:
            try:
                interval = self.source_intervals.get(source_name, 0.0)
                last_request = self._source_last_request.get(source_name)
                if interval > 0 and last_request is not None:
                    remaining = interval - (time.monotonic() - last_request)
                    if remaining > 0:
                        await _asyncio.sleep(remaining)
                self._source_last_request[source_name] = time.monotonic()
                result = await client.search(
                    query, limit=limit, page=page, offset=offset
                )
                if isinstance(result, SourceBatch):
                    return result
                return SourceBatch(
                    source_name,
                    (),
                    0,
                    SourceError("response_parsing", None, False, "invalid source response"),
                )
            except Exception as error:
                return SourceBatch(source_name, (), 0, _source_error(error))

        batch = await request_once()
        for delay in self.retry_delays:
            if batch.error is None or not batch.error.retryable:
                break
            await _asyncio.sleep(max(0.0, delay))
            batch = await request_once()
        return batch

    async def _fetch_source_query(self, source_name: str, client, query: str,
                                  request_limit: int, result_limit: int) -> SourceBatch:
        first = await self._request_source_batch(
            source_name, client, query, request_limit, page=1, offset=0
        )
        if not isinstance(first, SourceBatch) or not first.ok:
            return first
        papers = list(first.papers)
        error = None
        if first.total > len(papers) and len(papers) < result_limit:
            second_offset = len(papers) if source_name == "semantic_scholar" else 0
            second = await self._request_source_batch(
                source_name,
                client,
                query,
                request_limit,
                page=2,
                offset=second_offset,
            )
            if isinstance(second, SourceBatch):
                if second.ok:
                    papers.extend(second.papers)
                else:
                    error = second.error
        return SourceBatch(source_name, tuple(papers), first.total, error)

    async def _close_clients(self, clients) -> None:
        seen = set()
        for client in clients:
            if id(client) in seen:
                continue
            seen.add(id(client))
            close = getattr(client, "aclose", None)
            if close:
                try:
                    await close()
                except Exception:
                    logger.warning("literature source close failed: %s", type(client).__name__)

    @staticmethod
    def _combine_source_batches(source_name: str,
                                batches: list[SourceBatch]) -> SourceBatch:
        papers = []
        total = 0
        error = None
        for batch in batches:
            papers.extend(batch.papers)
            total = max(total, batch.total)
            if batch.error and error is None:
                error = batch.error
        return SourceBatch(source_name, tuple(papers), total, error)

    @staticmethod
    def _normalized_title(value: str) -> str:
        return " ".join(re.findall(r"[\w\u4e00-\u9fff]+", (value or "").casefold()))

    def _paper_identity(self, paper: dict) -> str:
        doi = normalize_doi(paper.get("doi"))
        if doi:
            return f"doi:{doi}"
        title = self._normalized_title(paper.get("title", ""))
        if title:
            return f"title:{title}:{paper.get('year') or ''}"
        return f"{paper.get('source', 'unknown')}:{paper.get('id', '')}"

    @staticmethod
    def _merge_paper(existing: dict, incoming: dict) -> None:
        source = incoming.get("source", "")
        if source and source not in existing["sources"]:
            existing["sources"].append(source)
        for field in ("abstract", "title"):
            if len(incoming.get(field, "") or "") > len(existing.get(field, "") or ""):
                existing[field] = incoming[field]
        for field in ("authors",):
            if len(incoming.get(field) or []) > len(existing.get(field) or []):
                existing[field] = incoming[field]
        for field in ("venue", "url", "doi", "year"):
            if not existing.get(field) and incoming.get(field):
                existing[field] = incoming[field]
        existing["citation_count"] = max(
            existing.get("citation_count", 0) or 0,
            incoming.get("citation_count", 0) or 0,
        )

    def _merge_and_rank(self, plan: QueryPlan, occurrences, sort: str,
                        extra_query_terms: Sequence[str] | None = None) -> list[dict]:
        by_identity = {}
        for source_name, variant_index, rank, raw in occurrences:
            paper = dict(raw)
            paper["doi"] = normalize_doi(paper.get("doi"))
            paper.setdefault("source", source_name)
            identity = self._paper_identity(paper)
            if identity not in by_identity:
                # 保留调用方已标注的来源列表（如引文扩展带入的多源信息），
                # 否则跨源一致性信号会被清零。
                existing_sources = [s for s in (paper.get("sources") or []) if s]
                if paper["source"] not in existing_sources:
                    existing_sources.insert(0, paper["source"])
                paper["sources"] = existing_sources or [paper["source"]]
                paper["_fusion_score"] = 0.0
                by_identity[identity] = paper
            else:
                self._merge_paper(by_identity[identity], paper)
            # 主检索(slot 0)权重最高；维度召回(slot 1)稍低但仍参与融合
            query_weight = 1.25 if variant_index == 0 else 1.0
            by_identity[identity]["_fusion_score"] += query_weight / (60 + rank)

        variants = [self._normalized_title(value) for value in plan.variants]
        query_terms = {
            token
            for variant in variants
            for token in variant.split()
            if len(token) > 1
        }
        # 维度检索式的词项也纳入词面重叠计算，否则维度召回的论文词面分恒为 0
        for extra in (extra_query_terms or ()):
            for token in self._normalized_title(extra).split():
                if len(token) > 1:
                    query_terms.add(token)
        original_title = self._normalized_title(plan.original)
        for paper in by_identity.values():
            text = self._normalized_title(
                f"{paper.get('title', '')} {paper.get('abstract', '')}"
            )
            overlap = sum(1 for term in query_terms if term in text)
            paper["_lexical_score"] = overlap / max(1, len(query_terms))
            if self._normalized_title(paper.get("title", "")) == original_title:
                paper["_fusion_score"] += 2.0

            # 跨源一致性信号：多个独立数据源都收录 → 更可能是真实核心文献。
            # 借鉴 findpapers 的 Paper.found_in（我们此前采集了 sources 却未用于排序）。
            source_count = len({s for s in (paper.get("sources") or []) if s})
            paper["_source_agreement"] = min(max(source_count - 1, 0), 3)
            paper["_score"] = (
                paper["_fusion_score"]
                + 0.25 * paper["_lexical_score"]
                + 0.3 * paper["_source_agreement"]
            )

        papers = list(by_identity.values())
        if sort == "year":
            papers.sort(key=lambda p: (p.get("year") or 0, p["_score"]), reverse=True)
        elif sort == "citations":
            papers.sort(key=lambda p: (p.get("citation_count") or 0, p["_score"]), reverse=True)
        else:
            papers.sort(key=lambda p: (p["_score"], p.get("citation_count") or 0), reverse=True)
        return papers

    @staticmethod
    def _select_diverse(papers: list[dict], source_batches: tuple[SourceBatch, ...],
                        limit: int) -> list[dict]:
        successful_sources = [batch.source for batch in source_batches if batch.papers]
        if limit < len(successful_sources):
            return papers[:limit]
        selected = []
        selected_ids = set()
        for source_name in successful_sources:
            candidate = next(
                (
                    paper for paper in papers
                    if source_name in paper.get("sources", [])
                    and id(paper) not in selected_ids
                ),
                None,
            )
            if candidate is not None:
                selected.append(candidate)
                selected_ids.add(id(candidate))
        for paper in papers:
            if len(selected) >= limit:
                break
            if id(paper) not in selected_ids:
                selected.append(paper)
                selected_ids.add(id(paper))
        return selected[:limit]

    async def _search_oa_expanded(self, queries: list[str], limit: int) -> list[dict]:
        per_q = max(30, limit // max(1, len(queries)) + 5)
        tasks = [self._oa.search(q, per_q) for q in queries]
        results_list = await _asyncio.gather(*tasks, return_exceptions=True)
        all_papers = []
        for r in results_list:
            if isinstance(r, dict):
                for p in r.get("results", []):
                    p["source"] = "openalex"
                    all_papers.append(p)
        return all_papers

    def _doi_deduplicate(self, papers: list[dict], limit: int) -> list[dict]:
        """DOI优先去重，再用标题去重"""
        seen_doi = set()
        seen_title = set()
        unique = []
        for p in papers:
            doi = (p.get("doi") or "").lower().strip()
            if doi and doi in seen_doi:
                continue
            if doi:
                seen_doi.add(doi)
            title_key = (p.get("title") or "").lower().strip()[:80]
            if title_key and title_key in seen_title:
                continue
            if title_key:
                seen_title.add(title_key)
            unique.append(p)
        return unique[:limit]

    def _sort_results(self, query: str, papers: list[dict], sort: str) -> list[dict]:
        if sort == "year":
            papers.sort(key=lambda p: p.get("year") or 0, reverse=True)
        elif sort == "citations":
            papers.sort(key=lambda p: p.get("citation_count", 0), reverse=True)
        elif sort == "ai_rank":
            # LLM rerank happens lazily at call site
            papers.sort(key=lambda p: p.get("citation_count", 0), reverse=True)
        else:  # "relevance"
            stop = {"the","a","an","is","are","of","in","on","to","for","with",
                    "and","or","by","at","from","as","be","it","we","this","that"}
            q_terms = {w for w in query.lower().replace("/", " ").split()
                       if w not in stop and len(w) > 1}
            def _score(p: dict) -> int:
                text = (p.get("title", "") + " " + (p.get("abstract") or "")).lower()
                return sum(1 for t in q_terms if t in text)
            papers.sort(key=_score, reverse=True)
        return papers

    def _deduplicate(self, papers: list[dict], limit: int) -> list[dict]:
        seen = set()
        unique = []
        for p in papers:
            key = p.get("title", "").lower().strip()[:80]
            if key and key not in seen:
                seen.add(key)
                unique.append(p)
        return unique[:limit]

    async def search_consensus_candidates(
        self,
        query: str,
        domain: str = "general",
        limit: int = 80,
        s2_api_key: str = "",
        query_variants: Sequence[str] | None = None,
        dimension_queries: Sequence[str] | None = None,
        citation_expansion: bool = True,
        citation_seeds: int = 5,
        citation_neighbors: int = 40,
        citation_depth: int = 2,
        citation_papers_per_level: int = 60,
        citation_expansion_per_level: int = 8,
    ) -> SearchResponse:
        """为共识分析定制的高通量、极低请求安全定向检索通道（对标 Consensus 搜索策略）。

        1. 支持多级学术检索式变体（主词检索 + 深度长句），兼顾召回广度与语义深度；
        2. 全领域激活 Semantic Scholar（S2），带有 1.05s 串行锁与静默容错；
        3. 按学科路由：计算机/物理加入 arXiv，生物医学加入 PubMed；
        4. 彻底排除 Crossref（无摘要率高，避免浪费请求与配额）；
        5. OpenAlex 原生开启 has_abstract 与标题摘要字段限定；
        6. ``dimension_queries``：Consensus Deep 式 7 维度短检索式，逐维度召回后合并。
           **关键**：实测长自然语言句子（>12 词）在各引擎召回为 0，因此维度检索式
           必须短；此处也会过滤掉过长的检索式以节省配额。
        """
        clean_q = " ".join(query.strip().split())
        # 清除输入中可能夹带的成对多余双引号
        clean_q = clean_q.replace('"', '').replace("'", '').strip()

        # 整理检索变体
        variants_list = [clean_q] if clean_q else []
        if query_variants:
            for v in query_variants:
                cv = " ".join(v.strip().split()).replace('"', '').replace("'", '').strip()
                if cv and cv not in variants_list:
                    variants_list.append(cv)

        # 维度检索式：过滤过长项（长句召回为 0，纯属浪费配额）
        dimension_list: list[str] = []
        if dimension_queries:
            for dq in dimension_queries:
                cdq = " ".join(str(dq).strip().split()).replace('"', '').replace("'", '').strip()
                if not cdq:
                    continue
                if len(cdq.split()) > MAX_QUERY_WORDS:
                    continue
                if cdq not in dimension_list:
                    dimension_list.append(cdq)

        if not variants_list:
            variants_list = [clean_q]

        plan = QueryPlan(clean_q, tuple(variants_list), False, None, _query_kind(clean_q))
        if (not clean_q and not dimension_list) or limit <= 0:
            return SearchResponse((), plan, ())

        await self._ensure_clients()

        # 动态支持外部传入的 S2 Key（如设置页面配置）
        effective_s2_key = s2_api_key or SEMANTIC_SCHOLAR_API_KEY
        if effective_s2_key and ("semantic_scholar" not in self.clients or self.clients["semantic_scholar"] is None):
            self.clients["semantic_scholar"] = SemanticScholarClient()

        # 确定启用的源（严格排除 crossref：无摘要，浪费 LLM 配额）
        target_sources: list[str] = ["openalex"]
        if "semantic_scholar" in self.clients and self.clients["semantic_scholar"] is not None:
            target_sources.append("semantic_scholar")
        # Europe PMC 免 Key、高吞吐、覆盖材料/器件与生物医学，作为通用第二召回源
        if "europepmc" in self.clients and self.clients["europepmc"] is not None:
            target_sources.append("europepmc")

        d = (domain or "general").lower()
        if d in {"cs_physics", "computer_science", "physics"}:
            if "arxiv" in self.clients and self.clients["arxiv"] is not None:
                target_sources.append("arxiv")
        elif d in {"biomedicine", "medicine", "biology"}:
            if "pubmed" in self.clients and self.clients["pubmed"] is not None:
                target_sources.append("pubmed")

        occurrences = []
        source_calls: dict[str, list[SourceBatch]] = {name: [] for name in target_sources}

        async def fetch_one_source(source_name: str) -> tuple[str, SourceBatch]:
            client = self.clients[source_name]
            try:
                # 遵循各源流控间隔保护（如 S2 1.05s 锁）
                interval = self.source_intervals.get(source_name, 0.0)
                last_request = self._source_last_request.get(source_name)
                if interval > 0 and last_request is not None:
                    remaining = interval - (time.monotonic() - last_request)
                    if remaining > 0:
                        await _asyncio.sleep(remaining)
                self._source_last_request[source_name] = time.monotonic()

                if source_name == "openalex":
                    # OpenAlex 每次调用固定消耗 10 积分（实测 per_page 1~200 同价），
                    # 因此单次拉满 200 篇 = 零边际成本的双倍召回。
                    oa_papers = []
                    oa_seen_ids = set()
                    oa_total = 0
                    primary_oa = variants_list[0]
                    batch_p = await client.search(
                        primary_oa, limit=200, page=1, filter_abstract=True, scope_title_abstract=False
                    )
                    oa_total += batch_p.total
                    for p in batch_p.papers:
                        pid = p.get("id") or p.get("title")
                        if pid and pid not in oa_seen_ids:
                            oa_seen_ids.add(pid)
                            oa_papers.append(p)
                    batch = SourceBatch("openalex", tuple(oa_papers), oa_total, None)

                elif source_name == "europepmc":
                    # Europe PMC 免 Key、高吞吐，单次 100 篇；用于补充 OpenAlex 未覆盖的
                    # 生物医学/材料文献。为避免过载，主检索词取核心短句。
                    epmc_query = clean_q
                    words = clean_q.split()
                    if len(words) > 12:
                        epmc_query = " ".join(words[:12])
                    batch = await client.search(epmc_query, limit=100, page=1)

                elif source_name == "semantic_scholar":
                    # 优先使用 bulk 端点（独立配额，单次可返 1000 篇），
                    # /search 在共享池限流下实测成功率仅约 30%，仅作兜底。
                    s2_query = clean_q
                    words = clean_q.split()
                    if len(words) > 6:
                        s2_query = " ".join(words[:5])
                    bulk_fn = getattr(client, "search_bulk", None)
                    batch = None
                    if callable(bulk_fn):
                        batch = await bulk_fn(s2_query, limit=200)
                    if batch is None or not batch.papers:
                        try:
                            fallback = await client.search(s2_query, limit=30, page=1, offset=0)
                            if fallback.papers or batch is None:
                                batch = fallback
                        except Exception:
                            if batch is None:
                                raise

                elif source_name == "arxiv":
                    batch = await client.search(clean_q, limit=20, page=1)
                elif source_name == "pubmed":
                    batch = await client.search(clean_q, limit=20, page=1)
                else:
                    batch = SourceBatch(source_name, (), 0, None)

                return source_name, batch
            except Exception as err:
                return source_name, SourceBatch(source_name, (), 0, _source_error(err))

        tasks = [fetch_one_source(s) for s in target_sources]
        results = await _asyncio.gather(*tasks, return_exceptions=True)

        for res in results:
            if isinstance(res, Exception):
                continue
            source_name, batch = res
            source_calls[source_name].append(batch)
            for rank, paper in enumerate(batch.papers, start=1):
                occurrences.append((source_name, 0, rank, paper))

        # === 7 维度多维召回（对标 Consensus Deep）===
        # 逐维度查询 OpenAlex + Europe PMC，把各维度结果并入统一候选池。
        # 实测：单条长句召回 0 篇，而 7 维度 × 短检索式可召回上千篇。
        if dimension_list:
            # 源健康状态：某源出现限流/网络错误后即停用，避免 14 次维度查询
            # 全部撞在同一个已耗尽的配额上（OpenAlex 每日仅 1000 积分 = 100 次调用）。
            source_disabled: dict[str, bool] = {}

            async def fetch_dimension(dim_query: str) -> list[tuple[str, int, dict]]:
                """返回 (source, rank, paper) 三元组；单维度失败不影响整体。"""
                rows: list[tuple[str, int, dict]] = []
                for src in ("openalex", "europepmc"):
                    if source_disabled.get(src):
                        continue
                    client = self.clients.get(src)
                    if client is None:
                        continue
                    try:
                        interval = self.source_intervals.get(src, 0.0)
                        last_request = self._source_last_request.get(src)
                        if interval > 0 and last_request is not None:
                            remaining = interval - (time.monotonic() - last_request)
                            if remaining > 0:
                                await _asyncio.sleep(remaining)
                        self._source_last_request[src] = time.monotonic()

                        if src == "openalex":
                            batch = await client.search(
                                dim_query, limit=200, page=1,
                                filter_abstract=True, scope_title_abstract=False,
                            )
                        else:
                            batch = await client.search(dim_query, limit=100, page=1)
                        if batch.error is not None:
                            # 限流/认证类错误短期内不会自愈，直接停用该源
                            if batch.error.category in {"rate_limit", "authentication"}:
                                source_disabled[src] = True
                            continue
                        for rank, paper in enumerate(batch.papers, start=1):
                            rows.append((src, rank, paper))
                    except Exception:
                        # 维度检索失败静默跳过，保证整体可用性
                        continue
                return rows

            dimension_results = await _asyncio.gather(
                *(fetch_dimension(dq) for dq in dimension_list),
                return_exceptions=True,
            )
            for dim_result in dimension_results:
                if isinstance(dim_result, Exception) or not dim_result:
                    continue
                for source_name, rank, paper in dim_result:
                    # dimension=1 标记为多维召回，参与融合排序但权重略低于主检索
                    occurrences.append((source_name, 1, rank, paper))

        source_batches = tuple(
            self._combine_source_batches(name, source_calls[name])
            for name in target_sources
        )
        merged = self._merge_and_rank(
            plan, occurrences, sort="relevance",
            extra_query_terms=tuple(dimension_list) + tuple(variants_list[1:]),
        )

        # === 引文网络扩展（Citation Graph / Snowballing）===
        # 对标 Consensus Deep：「Discover additional papers by following citation
        # connections」。关键词检索只能命中字面匹配，引文网络能找到主题相关但
        # 用词不同的工作（含奠基性经典文献）。
        #
        # 成本优势（实测）：OpenAlex 关键词检索 10 积分/次，引文图仅 1 积分/次；
        # Europe PMC 引文接口完全免费。因此这是配额受限时性价比最高的召回方式。
        if citation_expansion and merged:
            try:
                expansion = await self._expand_citation_graph(
                    merged, seed_count=citation_seeds,
                    max_neighbors=citation_neighbors,
                    max_depth=citation_depth,
                    max_papers_per_level=citation_papers_per_level,
                    max_expansion_per_level=citation_expansion_per_level,
                )
                for paper in expansion:
                    paper["_fusion_score"] = paper.get("_fusion_score", 0.0) + 0.35
                    paper["_score"] = paper.get("_score", 0.0) + 0.35
                merged.extend(expansion)
            except Exception:
                # 引文扩展失败绝不阻塞主检索
                pass

        selected = self._select_diverse(merged, source_batches, limit)
        return SearchResponse(tuple(selected), plan, source_batches)

    async def _fetch_citation_neighbours(self, papers: Sequence[dict],
                                         max_neighbors: int = 40) -> list[dict]:
        """取给定论文集合的直接引文邻居（前向 + 后向），已去重。

        三源分工（按成本从低到高）：
          1. Europe PMC——完全免费，但需要 pmid，且只返回无摘要存根，需 hydrate
          2. S2 按 DOI——能覆盖 Europe PMC 未收录的半导体器件文献（常无 pmid）
          3. OpenAlex——1 积分/次，作为兜底
        仅返回**带摘要**的论文，因为证据抽取必须有摘要。
        """
        outgoing: dict[str, dict] = {}
        stub_ids: list[str] = []

        def keep(paper: dict) -> None:
            if not paper or not paper.get("title"):
                return
            if not (paper.get("abstract") or "").strip():
                return
            outgoing.setdefault(self._paper_identity(paper), paper)

        epmc = self.clients.get("europepmc")
        s2 = self.clients.get("semantic_scholar")
        openalex = self.clients.get("openalex")

        # --- 1) Europe PMC（免费）---
        if epmc is not None and hasattr(epmc, "get_citations"):
            for paper in papers:
                pmid = paper.get("pmid")
                if not pmid:
                    continue
                try:
                    citations = await epmc.get_citations(pmid, limit=max_neighbors)
                    references = await epmc.get_references(pmid, limit=max_neighbors)
                except Exception:
                    continue
                for stub in list(citations) + list(references):
                    stub_id = str(stub.get("id") or "").strip()
                    if stub_id:
                        stub_ids.append(stub_id)

        # --- 2) S2 按 DOI（覆盖无 pmid 的文献）---
        if s2 is not None and hasattr(s2, "get_citation_neighbors"):
            for paper in papers:
                doi = normalize_doi(paper.get("doi"))
                if not doi:
                    continue
                for direction in ("citations", "references"):
                    try:
                        neighbours = await s2.get_citation_neighbors(
                            f"DOI:{doi}", limit=max_neighbors, direction=direction,
                        )
                    except Exception:
                        continue
                    for neighbour in neighbours:
                        keep(neighbour)

        # --- 3) OpenAlex（1 积分/次，兜底）---
        if openalex is not None and hasattr(openalex, "get_citing_works"):
            for paper in papers:
                wid = paper.get("openalex_id") or ""
                if not wid:
                    continue
                try:
                    fwd = await openalex.get_citing_works(wid, limit=max_neighbors)
                    for p in fwd.papers:
                        keep(p)
                except Exception:
                    pass
                try:
                    bwd = await openalex.get_referenced_works(wid, limit=max_neighbors)
                    for p in bwd.papers:
                        keep(p)
                except Exception:
                    pass

        # --- Europe PMC 存根批量补全（单次最多 50 条）---
        if epmc is not None and stub_ids:
            unique_stubs = list(dict.fromkeys(stub_ids))[: max_neighbors * 3]
            for start in range(0, len(unique_stubs), 50):
                chunk = unique_stubs[start:start + 50]
                if not chunk:
                    continue
                try:
                    batch = await epmc.hydrate(chunk, limit=50)
                except Exception:
                    continue
                for p in batch.papers:
                    keep(p)

        return list(outgoing.values())

    async def _expand_citation_graph(self, seeds: Sequence[dict],
                                     seed_count: int = 5,
                                     max_neighbors: int = 40,
                                     max_depth: int = 2,
                                     max_papers_per_level: int = 60,
                                     max_expansion_per_level: int = 8,
                                     ) -> list[dict]:
        """从高相关种子出发做**分层 BFS** 引文扩展。

        返回**新增**论文（已与种子去重），由调用方合并进候选池。

        设计借鉴 findpapers（4 年迭代的成熟实现）：
          引文网络是**指数爆炸**的（1 → 50 → 2500 → …），因此必须有两个独立上限：
            - ``max_papers_per_level``：每层**入库**上限（按引用量取 top N）
            - ``max_expansion_per_level``：每层**参与下一轮扩展**的上限（更严格）
          二者分离，才能在加深遍历的同时不失控。
          这也是我们此前"单轮浅扩展"与成熟实现的主要差距。

        实测约束：本领域大量文献无 pmid，因此 Europe PMC 覆盖不全，
        必须由 S2（按 DOI）与 OpenAlex 补位。
        """
        def seed_score(p: dict) -> tuple:
            return (
                1 if p.get("pmid") else 0,
                1 if p.get("doi") else 0,
                1 if p.get("openalex_id") else 0,
                p.get("citation_count") or 0,
            )

        ranked = sorted(
            (p for p in seeds
             if p.get("pmid") or p.get("doi") or p.get("openalex_id")),
            key=seed_score, reverse=True,
        )[:max(1, seed_count)]
        if not ranked:
            return []

        # visited 跨层共享，避免重复抓取与循环
        visited: set[str] = {self._paper_identity(p) for p in seeds}
        collected: dict[str, dict] = {}
        frontier = list(ranked)

        def top_n(items: list[dict], n: int) -> list[dict]:
            return sorted(
                items, key=lambda p: p.get("citation_count") or 0, reverse=True
            )[:max(1, n)]

        for level in range(1, max(0, int(max_depth)) + 1):
            if not frontier:
                break
            try:
                neighbours = await self._fetch_citation_neighbours(
                    frontier, max_neighbors=max_neighbors
                )
            except Exception:
                break

            # 过滤已访问
            fresh: list[dict] = []
            for paper in neighbours:
                identity = self._paper_identity(paper)
                if identity in visited:
                    continue
                visited.add(identity)
                fresh.append(paper)
            if not fresh:
                break

            # 入库：按引用量取 top N
            for paper in top_n(fresh, max_papers_per_level):
                collected.setdefault(self._paper_identity(paper), paper)

            # 扩展：更严格的独立上限，防止下一轮指数增长
            frontier = top_n(fresh, max_expansion_per_level)

        return list(collected.values())


# ============================================================
# Init DB on import
# ============================================================
init_db()

import os
import logging
import re
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
DEEPSEEK_MODEL = "deepseek-chat"
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "")
DEFAULT_LLM_PROVIDER = "glm"
DEFAULT_LLM_MODEL = "glm-4.7-flash"

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

    @property
    def ok(self) -> bool:
        return self.error is None


def _source_error(error: Exception) -> SourceError:
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    http_status_error = getattr(httpx, "HTTPStatusError", ())
    timeout_errors = _httpx_timeout_types()
    if status_code is not None or (http_status_error and isinstance(error, http_status_error)):
        if status_code in {401, 403}:
            category = "authentication"
        elif status_code == 429:
            category = "rate_limit"
        else:
            category = "http"
    elif timeout_errors and isinstance(error, timeout_errors):
        category = "timeout"
    elif isinstance(error, (ValueError, KeyError, TypeError, ET.ParseError)):
        category = "response_parsing"
    else:
        category = "request"
    retryable = category in {"timeout", "rate_limit"} or bool(
        status_code and status_code >= 500
    )
    return SourceError(category, status_code, retryable, str(error)[:240])


def normalize_doi(value: str | None) -> str:
    doi = (value or "").strip().casefold()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi.strip()


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
                     offset: int = 0) -> SourceBatch:
        try:
            r = await self.client.get("/works", params={"search": query, "per_page": limit, "page": page})
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


# ============================================================
# LLM Provider Configurations
# ============================================================
LLM_PROVIDERS = {
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-v4-flash", "deepseek-v4-pro", "deepseek-chat", "deepseek-coder", "deepseek-reasoner"],
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
        "models": [DEFAULT_LLM_MODEL, "glm-4.7"],
        "default_model": DEFAULT_LLM_MODEL,
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
    timeout_errors = _httpx_timeout_types()
    http_status_error = getattr(httpx, "HTTPStatusError", ())
    if timeout_errors and isinstance(error, timeout_errors):
        return "timeout"
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
    def __init__(self, provider: str = DEFAULT_LLM_PROVIDER, api_key: str = "", model: str = ""):
        cfg = LLM_PROVIDERS.get(provider, LLM_PROVIDERS[DEFAULT_LLM_PROVIDER])
        self.provider = provider if provider in LLM_PROVIDERS else DEFAULT_LLM_PROVIDER
        self.model = model or cfg["default_model"]
        self.base_url = cfg["base_url"]
        self.api_format = cfg.get("api_format", "openai")

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
            self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=httpx.Timeout(120.0))
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
                              response_format: dict | None = None) -> str:
        client = self._get_client()
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
                    "max_tokens": 4096,
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
                "max_tokens": 4096,
            }
            if response_format:
                payload["response_format"] = response_format
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except LLMRequestError:
            raise
        except Exception as error:
            raise LLMRequestError(
                self.provider,
                _llm_error_category(error),
                str(error),
            ) from error

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
# Search Engine
# ============================================================
import re as _re
import asyncio as _asyncio

def _contains_chinese(text: str) -> bool:
    return bool(_re.search(r'[\u4e00-\u9fff]', text))


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
                 clients: dict[str, object] | None = None, query_planner=None):
        self.llm_config = llm_config or LLMConfig.default()
        self._clients_injected = clients is not None
        clients = dict(clients or {})
        self.clients = clients
        self._oa = None  # OpenAlex
        self._arx = None # arXiv
        self._ss = None  # Semantic Scholar
        self._cr = None  # Crossref
        self._pm = None  # PubMed
        self._oa = clients.get("openalex")
        self._arx = clients.get("arxiv")
        self._ss = clients.get("semantic_scholar")
        self._cr = clients.get("crossref")
        self._pm = clients.get("pubmed")
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
        self.clients = {
            name: client
            for name, client in {
                "openalex": self._oa,
                "arxiv": self._arx,
                "semantic_scholar": self._ss,
                "crossref": self._cr,
                "pubmed": self._pm,
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

        per_query_limit = max(10, min(100, limit))
        tasks = []
        task_meta = []
        for source_name, client in selected_clients.items():
            for variant_index, variant in enumerate(plan.variants):
                task_meta.append((source_name, variant_index, variant))
                tasks.append(
                    self._fetch_source_query(
                        source_name, client, variant, per_query_limit, limit
                    )
                )

        occurrences = []
        source_calls: dict[str, list[SourceBatch]] = {
            name: [] for name in selected_clients
        }
        try:
            gathered = await _asyncio.gather(*tasks, return_exceptions=True)
            for meta, result in zip(task_meta, gathered):
                source_name, variant_index, _variant = meta
                if isinstance(result, Exception):
                    result = SourceBatch(source_name, (), 0, _source_error(result))
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

    async def _fetch_source_query(self, source_name: str, client, query: str,
                                  request_limit: int, result_limit: int) -> SourceBatch:
        first = await client.search(query, limit=request_limit, page=1, offset=0)
        if not isinstance(first, SourceBatch) or not first.ok:
            return first
        papers = list(first.papers)
        error = None
        if first.total > len(papers) and len(papers) < result_limit:
            second_offset = len(papers) if source_name == "semantic_scholar" else 0
            second = await client.search(
                query,
                limit=request_limit,
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

    def _merge_and_rank(self, plan: QueryPlan, occurrences, sort: str) -> list[dict]:
        by_identity = {}
        for source_name, variant_index, rank, raw in occurrences:
            paper = dict(raw)
            paper["doi"] = normalize_doi(paper.get("doi"))
            paper.setdefault("source", source_name)
            identity = self._paper_identity(paper)
            if identity not in by_identity:
                paper["sources"] = [paper["source"]]
                paper["_fusion_score"] = 0.0
                by_identity[identity] = paper
            else:
                self._merge_paper(by_identity[identity], paper)
            query_weight = 1.25 if variant_index == 0 else 1.0
            by_identity[identity]["_fusion_score"] += query_weight / (60 + rank)

        variants = [self._normalized_title(value) for value in plan.variants]
        query_terms = {
            token
            for variant in variants
            for token in variant.split()
            if len(token) > 1
        }
        original_title = self._normalized_title(plan.original)
        for paper in by_identity.values():
            text = self._normalized_title(
                f"{paper.get('title', '')} {paper.get('abstract', '')}"
            )
            overlap = sum(1 for term in query_terms if term in text)
            paper["_lexical_score"] = overlap / max(1, len(query_terms))
            if self._normalized_title(paper.get("title", "")) == original_title:
                paper["_fusion_score"] += 2.0
            paper["_score"] = paper["_fusion_score"] + 0.25 * paper["_lexical_score"]

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


# ============================================================
# Init DB on import
# ============================================================
init_db()

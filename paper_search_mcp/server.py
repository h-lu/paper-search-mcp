# paper_search_mcp/server.py
"""
MCP Server - 学术论文搜索服务

2025 最佳实践版本：
- 使用工厂函数减少代码重复
- 统一的错误处理
- 日志记录替代 print()
- 清晰的模块化结构

支持的学术平台:
- arXiv: 预印本服务器 (物理、数学、计算机科学)
- PubMed: 生物医学文献
- bioRxiv/medRxiv: 生物/医学预印本
- Semantic Scholar: AI 驱动的学术搜索
- CrossRef: DOI 引用数据库
- IACR: 密码学预印本
- Google Scholar: 学术搜索引擎
- RePEc/IDEAS: 经济学论文库
"""
from typing import List, Dict, Optional, Any
import logging
import os

from mcp.server.fastmcp import FastMCP

from .academic_platforms.arxiv import ArxivSearcher
from .academic_platforms.pubmed import PubMedSearcher
from .academic_platforms.biorxiv import BioRxivSearcher
from .academic_platforms.medrxiv import MedRxivSearcher
from .academic_platforms.google_scholar import GoogleScholarSearcher
from .academic_platforms.iacr import IACRSearcher
from .academic_platforms.semantic import SemanticSearcher
from .academic_platforms.crossref import CrossRefSearcher
from .academic_platforms.repec import RePECSearcher
from .academic_platforms.sci_hub import SciHubFetcher, check_paper_year
from .paper import Paper

# ============================================================
# 配置
# ============================================================
# PDF 下载目录，可通过环境变量 PAPER_DOWNLOAD_PATH 配置
# 默认为用户目录下的 paper_downloads (跨平台兼容)
# - macOS: ~/paper_downloads
# - Linux: ~/paper_downloads  
# - Windows: C:\Users\<username>\paper_downloads
from pathlib import Path

def get_download_path() -> str:
    """获取下载路径，支持跨平台
    
    注意：此函数每次调用时都会重新计算路径，以确保：
    1. 环境变量 PAPER_DOWNLOAD_PATH 的变化能够生效
    2. MCP 在不同环境下运行时能正确获取 HOME 目录
    """
    env_path = os.environ.get("PAPER_DOWNLOAD_PATH")
    if env_path:
        return env_path
    # 使用 Path.home() 获取跨平台的用户主目录
    return str(Path.home() / "paper_downloads")

# ============================================================
# 日志配置
# ============================================================
logger = logging.getLogger(__name__)

# ============================================================
# MCP Server 初始化
# ============================================================
mcp = FastMCP("paper_search_server")

# ============================================================
# 搜索器实例（单例）
# ============================================================
SEARCHERS = {
    'arxiv': ArxivSearcher(),
    'pubmed': PubMedSearcher(),
    'biorxiv': BioRxivSearcher(),
    'medrxiv': MedRxivSearcher(),
    'google_scholar': GoogleScholarSearcher(),
    'iacr': IACRSearcher(),
    'semantic': SemanticSearcher(),
    'crossref': CrossRefSearcher(),
    'repec': RePECSearcher(),
}

# Sci-Hub 实例（单独管理，仅用于下载）
SCIHUB = SciHubFetcher()



# ============================================================
# 通用搜索和操作函数
# ============================================================
async def _search(
    searcher_name: str, 
    query: str, 
    max_results: int = 10,
    **kwargs
) -> List[Dict]:
    """通用搜索函数"""
    searcher = SEARCHERS.get(searcher_name)
    if not searcher:
        logger.error(f"Unknown searcher: {searcher_name}")
        return []
    
    try:
        papers = searcher.search(query, max_results=max_results, **kwargs)
        return [paper.to_dict() for paper in papers]
    except Exception as e:
        logger.error(f"Search failed for {searcher_name}: {e}")
        return []


async def _download(
    searcher_name: str, 
    paper_id: str, 
    save_path: Optional[str] = None
) -> str:
    """通用下载函数"""
    if save_path is None:
        save_path = get_download_path()
    
    searcher = SEARCHERS.get(searcher_name)
    if not searcher:
        return f"Error: Unknown searcher {searcher_name}"
    
    try:
        return searcher.download_pdf(paper_id, save_path)
    except NotImplementedError as e:
        return str(e)
    except Exception as e:
        logger.error(f"Download failed for {searcher_name}: {e}")
        return f"Error downloading: {str(e)}"


async def _read(
    searcher_name: str, 
    paper_id: str, 
    save_path: Optional[str] = None
) -> str:
    """通用阅读函数"""
    if save_path is None:
        save_path = get_download_path()
    
    searcher = SEARCHERS.get(searcher_name)
    if not searcher:
        return f"Error: Unknown searcher {searcher_name}"
    
    try:
        return searcher.read_paper(paper_id, save_path)
    except NotImplementedError as e:
        return str(e)
    except Exception as e:
        logger.error(f"Read failed for {searcher_name}: {e}")
        return f"Error reading paper: {str(e)}"


# ============================================================
# arXiv 工具
# ============================================================
@mcp.tool()
async def search_arxiv(query: str, max_results: int = 10) -> List[Dict]:
    """Search preprints on arXiv - major open-access preprint server.
    
    USE THIS TOOL WHEN:
    - Searching for PREPRINTS (not peer-reviewed yet)
    - You need free, immediate access to full-text PDFs
    - Searching in: Physics, Mathematics, Computer Science, Statistics,
      Quantitative Biology, Quantitative Finance, Electrical Engineering
    
    NOTE: arXiv is a PREPRINT server - papers may not be peer-reviewed.
    For peer-reviewed papers, use search_crossref or search_semantic.
    
    WORKFLOW:
    1. search_arxiv(query) -> get paper_id (e.g., '2106.12345')
    2. download_arxiv(paper_id) -> get PDF (always available)
    3. read_arxiv_paper(paper_id) -> get full text as Markdown

    Args:
        query: Search terms in any supported field.
        max_results: Number of results (default: 10).
    
    Returns:
        List of paper dicts with: paper_id, title, authors, abstract, 
        published_date, pdf_url, categories.
    
    Example:
        search_arxiv("quantum computing error correction", max_results=5)
    """
    return await _search('arxiv', query, max_results)


@mcp.tool()
async def download_arxiv(paper_id: str, save_path: Optional[str] = None) -> str:
    """Download PDF from arXiv (always free and available).
    
    Args:
        paper_id: arXiv ID (e.g., '2106.12345', '2312.00001v2').
        save_path: Directory to save PDF (default: ~/paper_downloads).
    
    Returns:
        Path to downloaded PDF file.
    
    Example:
        download_arxiv("2106.12345")
    """
    return await _download('arxiv', paper_id, save_path)


@mcp.tool()
async def read_arxiv_paper(paper_id: str, save_path: Optional[str] = None) -> str:
    """Download and extract full text from arXiv paper as Markdown.
    
    Args:
        paper_id: arXiv ID (e.g., '2106.12345').
        save_path: Directory to save PDF (default: ~/paper_downloads).
    
    Returns:
        Full paper text in Markdown format.
    
    Example:
        read_arxiv_paper("2106.12345")
    """
    return await _read('arxiv', paper_id, save_path)


# ============================================================
# PubMed 工具
# ============================================================
@mcp.tool()
async def search_pubmed(query: str, max_results: int = 10) -> List[Dict]:
    """Search biomedical literature on PubMed (NCBI database).
    
    USE THIS TOOL WHEN:
    - Searching for medical, clinical, or biomedical research
    - You need peer-reviewed published papers (not preprints)
    - Searching for drug studies, clinical trials, disease research
    
    DOMAIN: Medicine, Biology, Pharmacology, Public Health, 
    Clinical Research, Genetics, Biochemistry.
    
    LIMITATION: PubMed provides metadata/abstracts ONLY, not full PDFs.
    
    WORKFLOW FOR FULL TEXT:
    1. search_pubmed(query) -> get DOI from results
    2. download_scihub(doi) -> download PDF (if published before 2023)
    
    Args:
        query: Medical/scientific terms (e.g., 'cancer immunotherapy', 'COVID-19 vaccine').
        max_results: Number of results (default: 10).
    
    Returns:
        List of paper dicts with: paper_id (PMID), title, authors, 
        abstract, published_date, doi, url.
    
    Example:
        search_pubmed("CRISPR gene therapy", max_results=5)
    """
    return await _search('pubmed', query, max_results)


@mcp.tool()
async def download_pubmed(paper_id: str, save_path: Optional[str] = None) -> str:
    """PubMed does NOT support direct PDF downloads.
    
    PubMed is a metadata database - it does not host PDFs.
    
    INSTEAD (try in order):
    1. download_scihub(doi) - if published before 2023
    2. download_semantic(id) - last resort
    
    Args:
        paper_id: PMID (unused).
        save_path: Unused.
    
    Returns:
        Error message with alternatives.
    """
    return await _download('pubmed', paper_id, save_path)


@mcp.tool()
async def read_pubmed_paper(paper_id: str, save_path: Optional[str] = None) -> str:
    """PubMed does NOT support direct paper reading.
    
    INSTEAD (try in order):
    1. read_scihub_paper(doi) - if published before 2023
    2. read_semantic_paper(id) - last resort
    
    Args:
        paper_id: PMID (unused).
        save_path: Unused.
    
    Returns:
        Error message with alternatives.
    """
    return await _read('pubmed', paper_id, save_path)


# ============================================================
# bioRxiv 工具
# ============================================================
@mcp.tool()
async def search_biorxiv(query: str, max_results: int = 10) -> List[Dict]:
    """Search biology preprints on bioRxiv.
    
    USE THIS TOOL WHEN:
    - Searching for cutting-edge biology research (preprints)
    - You need the latest findings before peer review
    - Searching by CATEGORY, not keyword (see below)
    
    DOMAIN: Molecular Biology, Cell Biology, Genetics, Neuroscience,
    Bioinformatics, Evolutionary Biology, Microbiology, etc.
    
    NOTE: bioRxiv search uses CATEGORY names, not keywords.
    Categories: 'neuroscience', 'cell_biology', 'genetics', 'genomics',
    'bioinformatics', 'cancer_biology', 'immunology', etc.
    
    WORKFLOW:
    1. search_biorxiv(category) -> get DOI
    2. download_biorxiv(doi) or read_biorxiv_paper(doi)

    Args:
        query: Category name (e.g., 'neuroscience', 'cell_biology').
        max_results: Number of results (default: 10).
    
    Returns:
        List of recent preprints in that category.
    
    Example:
        search_biorxiv("neuroscience", max_results=5)
    """
    return await _search('biorxiv', query, max_results)


@mcp.tool()
async def download_biorxiv(paper_id: str, save_path: Optional[str] = None) -> str:
    """Download PDF from bioRxiv (free and open access).
    
    Args:
        paper_id: bioRxiv DOI (e.g., '10.1101/2024.01.01.123456').
        save_path: Directory to save PDF.
    
    Returns:
        Path to downloaded PDF.
    """
    return await _download('biorxiv', paper_id, save_path)


@mcp.tool()
async def read_biorxiv_paper(paper_id: str, save_path: Optional[str] = None) -> str:
    """Download and extract full text from bioRxiv paper.
    
    Args:
        paper_id: bioRxiv DOI.
        save_path: Directory to save PDF.
    
    Returns:
        Full paper text in Markdown format.
    """
    return await _read('biorxiv', paper_id, save_path)


# ============================================================
# medRxiv 工具
# ============================================================
@mcp.tool()
async def search_medrxiv(query: str, max_results: int = 10) -> List[Dict]:
    """Search medical preprints on medRxiv.
    
    USE THIS TOOL WHEN:
    - Searching for clinical/medical research preprints
    - You need latest COVID-19, epidemiology, or clinical studies
    - Searching by CATEGORY, not keyword (see below)
    
    DOMAIN: Epidemiology, Infectious Diseases, Cardiology, Oncology,
    Public Health, Psychiatry, Health Informatics, etc.
    
    NOTE: medRxiv search uses CATEGORY names, not keywords.
    Categories: 'infectious_diseases', 'epidemiology', 'cardiology',
    'oncology', 'health_informatics', 'psychiatry', etc.
    
    WORKFLOW:
    1. search_medrxiv(category) -> get DOI
    2. download_medrxiv(doi) or read_medrxiv_paper(doi)

    Args:
        query: Category name (e.g., 'infectious_diseases', 'epidemiology').
        max_results: Number of results (default: 10).
    
    Returns:
        List of recent preprints in that category.
    
    Example:
        search_medrxiv("infectious_diseases", max_results=5)
    """
    return await _search('medrxiv', query, max_results)


@mcp.tool()
async def download_medrxiv(paper_id: str, save_path: Optional[str] = None) -> str:
    """Download PDF from medRxiv (free and open access).
    
    Args:
        paper_id: medRxiv DOI (e.g., '10.1101/2024.01.01.12345678').
        save_path: Directory to save PDF.
    
    Returns:
        Path to downloaded PDF.
    """
    return await _download('medrxiv', paper_id, save_path)


@mcp.tool()
async def read_medrxiv_paper(paper_id: str, save_path: Optional[str] = None) -> str:
    """Download and extract full text from medRxiv paper.
    
    Args:
        paper_id: medRxiv DOI.
        save_path: Directory to save PDF.
    
    Returns:
        Full paper text in Markdown format.
    """
    return await _read('medrxiv', paper_id, save_path)


# ============================================================
# Google Scholar 工具
# ============================================================
@mcp.tool()
async def search_google_scholar(query: str, max_results: int = 10) -> List[Dict]:
    """Search academic papers on Google Scholar (broad coverage).
    
    USE THIS TOOL WHEN:
    - You need broad academic search across ALL disciplines
    - You want citation counts and "cited by" information
    - Other specialized tools don't cover your topic
    
    COVERAGE: All academic disciplines, books, theses, patents.
    
    LIMITATIONS:
    - Uses web scraping (may be rate-limited)
    - Does NOT support PDF download
    
    FOR FULL TEXT (try in order):
    1. download_arxiv(id) - if arXiv preprint
    2. download_scihub(doi) - if published before 2023
    3. download_semantic(id) - last resort
    
    Args:
        query: Search terms (any academic topic).
        max_results: Number of results (default: 10, keep small to avoid blocks).
    
    Returns:
        List of paper dicts with: title, authors, abstract snippet,
        citations count, url, source.
    
    Example:
        search_google_scholar("climate change economic impact", max_results=5)
    """
    return await _search('google_scholar', query, max_results)


# ============================================================
# IACR ePrint 工具
# ============================================================
@mcp.tool()
async def search_iacr(
    query: str, max_results: int = 10, fetch_details: bool = True
) -> List[Dict]:
    """Search cryptography papers on IACR ePrint Archive.
    
    USE THIS TOOL WHEN:
    - Searching for cryptography or security research
    - You need papers on encryption, blockchain, zero-knowledge proofs
    - Looking for security protocols, hash functions, signatures
    
    DOMAIN: Cryptography ONLY - encryption, signatures, protocols,
    blockchain, secure computation, zero-knowledge, hash functions.
    
    All papers are FREE and open access with PDF download.
    
    WORKFLOW:
    1. search_iacr(query) -> get paper_id (e.g., '2024/123')
    2. download_iacr(paper_id) or read_iacr_paper(paper_id)
    
    Args:
        query: Crypto terms (e.g., 'zero knowledge', 'homomorphic encryption').
        max_results: Number of results (default: 10).
        fetch_details: Get full metadata per paper (default: True).
    
    Returns:
        List of paper dicts with: paper_id, title, authors, abstract,
        published_date, pdf_url.
    
    Example:
        search_iacr("post-quantum cryptography", max_results=5)
    """
    searcher = SEARCHERS['iacr']
    try:
        papers = searcher.search(query, max_results, fetch_details)
        return [paper.to_dict() for paper in papers] if papers else []
    except Exception as e:
        logger.error(f"IACR search failed: {e}")
        return []


@mcp.tool()
async def download_iacr(paper_id: str, save_path: Optional[str] = None) -> str:
    """Download PDF from IACR ePrint (always free).
    
    Args:
        paper_id: IACR ID (e.g., '2024/123', '2009/101').
        save_path: Directory to save PDF.
    
    Returns:
        Path to downloaded PDF.
    
    Example:
        download_iacr("2024/123")
    """
    return await _download('iacr', paper_id, save_path)


@mcp.tool()
async def read_iacr_paper(paper_id: str, save_path: Optional[str] = None) -> str:
    """Download and extract full text from IACR paper.
    
    Args:
        paper_id: IACR ID (e.g., '2024/123').
        save_path: Directory to save PDF.
    
    Returns:
        Full paper text in Markdown format.
    
    Example:
        read_iacr_paper("2024/123")
    """
    return await _read('iacr', paper_id, save_path)


# ============================================================
# Semantic Scholar 工具
# ============================================================
@mcp.tool()
async def search_semantic(
    query: str, year: Optional[str] = None, max_results: int = 10
) -> List[Dict]:
    """Search papers on Semantic Scholar - general-purpose academic search engine.
    
    USE THIS TOOL WHEN:
    - You want to search across ALL academic disciplines
    - You need citation counts and influence metrics
    - You want to filter by publication year
    - You need open-access PDF links when available
    
    COVERAGE: ALL academic fields - sciences, humanities, medicine, etc.
    Indexes 200M+ papers from journals, conferences, and preprints.
    
    WORKFLOW:
    1. search_semantic(query) -> get paper_id or DOI
    2. download_semantic(paper_id) -> get PDF (if open-access)
    3. If no PDF: use download_scihub(doi) for older papers
    
    Args:
        query: Search terms (any topic, any field).
        year: Optional year filter: '2023', '2020-2023', '2020-', '-2019'.
        max_results: Number of results (default: 10).
    
    Returns:
        List of paper dicts with: paper_id, title, authors, abstract,
        published_date, doi, citations, url, pdf_url (if available).
    
    Example:
        search_semantic("climate change impact agriculture", year="2020-", max_results=5)
    """
    kwargs = {'year': year} if year else {}
    return await _search('semantic', query, max_results, **kwargs)


@mcp.tool()
async def download_semantic(paper_id: str, save_path: Optional[str] = None) -> str:
    """Download PDF via Semantic Scholar (open-access only, use as LAST RESORT).
    
    DOWNLOAD PRIORITY (try in order):
    1. If arXiv paper -> use download_arxiv(arxiv_id) (always works)
    2. If published before 2023 -> use download_scihub(doi)
    3. Use this tool as last resort (may not have PDF)
    
    Args:
        paper_id: Semantic Scholar ID, or prefixed: 'DOI:xxx', 'ARXIV:xxx', 'PMID:xxx'
        save_path: Directory to save PDF.
    
    Returns:
        Path to downloaded PDF, or error if not available.
    """
    return await _download('semantic', paper_id, save_path)


@mcp.tool()
async def read_semantic_paper(paper_id: str, save_path: Optional[str] = None) -> str:
    """Read paper via Semantic Scholar (open-access only, use as LAST RESORT).
    
    DOWNLOAD PRIORITY (try in order):
    1. If arXiv paper -> use read_arxiv_paper(arxiv_id)
    2. If published before 2023 -> use read_scihub_paper(doi)
    3. Use this tool as last resort
    
    Args:
        paper_id: Semantic Scholar ID or prefixed ID (DOI:, ARXIV:, PMID:).
        save_path: Directory to save PDF.
    
    Returns:
        Full paper text in Markdown format.
    """
    return await _read('semantic', paper_id, save_path)


# ============================================================
# CrossRef 工具
# ============================================================
@mcp.tool()
async def search_crossref(
    query: str, 
    max_results: int = 10,
    **kwargs
) -> List[Dict]:
    """Search academic papers in CrossRef - the largest DOI citation database.
    
    USE THIS TOOL WHEN:
    - You need to find papers by DOI or citation metadata
    - You want to search across all academic publishers (not just preprints)
    - You need publication metadata like journal, volume, issue, citations
    - You want to verify if a DOI exists or get its metadata
    
    CrossRef indexes 150M+ scholarly works from thousands of publishers.
    Results include DOI, authors, title, abstract, citations, and publisher info.

    Args:
        query: Search terms (e.g., 'machine learning', 'CRISPR gene editing').
        max_results: Number of results (default: 10, max: 1000).
        **kwargs: Optional filters:
            - filter: 'has-full-text:true,from-pub-date:2020'
            - sort: 'relevance' | 'published' | 'cited'
            - order: 'asc' | 'desc'
    
    Returns:
        List of paper metadata dicts with keys: paper_id (DOI), title, 
        authors, abstract, doi, published_date, citations, url.
    
    Example:
        search_crossref("attention mechanism transformer", max_results=5)
    """
    return await _search('crossref', query, max_results, **kwargs)


@mcp.tool()
async def get_crossref_paper_by_doi(doi: str) -> Dict:
    """Get paper metadata from CrossRef using its DOI.
    
    USE THIS TOOL WHEN:
    - You have a DOI and need full metadata (title, authors, journal, etc.)
    - You want to verify a DOI exists
    - You need citation count for a specific paper
    
    Args:
        doi: Digital Object Identifier (e.g., '10.1038/nature12373').
    
    Returns:
        Paper metadata dict, or empty dict {} if DOI not found.
    
    Example:
        get_crossref_paper_by_doi("10.1038/nature12373")
    """
    searcher = SEARCHERS['crossref']
    try:
        paper = searcher.get_paper_by_doi(doi)
        return paper.to_dict() if paper else {}
    except Exception as e:
        logger.error(f"CrossRef DOI lookup failed: {e}")
        return {}


@mcp.tool()
async def download_crossref(paper_id: str, save_path: Optional[str] = None) -> str:
    """CrossRef does NOT support direct PDF downloads.
    
    CrossRef is a metadata/citation database only - it does not host PDFs.
    
    INSTEAD (try in order):
    1. download_arxiv(id) - if arXiv preprint (always works)
    2. download_scihub(doi) - if published before 2023
    3. download_semantic(id) - last resort (may not have PDF)
    
    Args:
        paper_id: DOI (e.g., '10.1038/nature12373').
        save_path: Unused.
    
    Returns:
        Error message explaining alternatives.
    """
    return await _download('crossref', paper_id, save_path)


@mcp.tool()
async def read_crossref_paper(paper_id: str, save_path: Optional[str] = None) -> str:
    """CrossRef does NOT support direct paper reading.
    
    CrossRef provides metadata only, not full-text content.
    
    INSTEAD (try in order):
    1. read_arxiv_paper(id) - if arXiv preprint
    2. read_scihub_paper(doi) - if published before 2023
    3. read_semantic_paper(id) - last resort
    
    Args:
        paper_id: DOI (e.g., '10.1038/nature12373').
        save_path: Unused.
    
    Returns:
        Error message explaining alternatives.
    """
    return await _read('crossref', paper_id, save_path)


# ============================================================
# RePEc/IDEAS 工具
# ============================================================
@mcp.tool()
async def search_repec(
    query: str, 
    max_results: int = 10,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    search_field: str = 'all',
    sort_by: str = 'relevance',
    doc_type: str = 'all',
    series: Optional[str] = None,
) -> List[Dict]:
    """Search economics papers on RePEc/IDEAS - the largest open economics bibliography.
    
    USE THIS TOOL WHEN:
    - Searching for ECONOMICS research (macro, micro, finance, econometrics)
    - You need working papers from NBER, Federal Reserve, World Bank, etc.
    - You want to find papers by JEL classification
    - Searching for economic policy analysis
    
    COVERAGE: 4.5M+ items including:
    - Working Papers: NBER, Fed banks, ECB, IMF, World Bank
    - Journal Articles: AER, JPE, QJE, Econometrica, etc.
    - Books and Book Chapters
    
    SEARCH SYNTAX:
    - Boolean: + for AND, | for OR, ~ for NOT (e.g., 'money ~liquidity')
    - Phrase: use double quotes (e.g., '"monetary policy"')
    - Author(Year): e.g., 'Acemoglu (2019)' or 'Kydland Prescott (1977)'
    - Synonyms: automatic (labor=labour, USA=United States)
    - Word stemming: automatic (find matches finds, finding, findings)
    
    LIMITATION: RePEc provides metadata only, not full PDFs.
    PDFs are hosted at original institutions (often freely available).
    
    Args:
        query: Search terms with optional boolean operators.
        max_results: Number of results (default: 10).
        year_from: Optional start year filter (e.g., 2020).
        year_to: Optional end year filter (e.g., 2025).
        search_field: Where to search, one of:
            - 'all': Whole record (default)
            - 'abstract': Abstract only
            - 'keywords': Keywords only
            - 'title': Title only
            - 'author': Author only
        sort_by: How to sort results, one of:
            - 'relevance': Most relevant (default)
            - 'newest': Most recent first
            - 'oldest': Oldest first
            - 'citations': Most cited first
            - 'recent_relevant': Recent and relevant
            - 'relevant_cited': Relevant and cited
        doc_type: Document type filter, one of:
            - 'all': All types (default)
            - 'articles': Journal articles
            - 'papers': Working papers (NBER, Fed, etc.)
            - 'chapters': Book chapters
            - 'books': Books
            - 'software': Software components
        series: Institution/journal series to search within, one of:
            - Institutions: 'nber', 'imf', 'worldbank', 'ecb', 'bis', 'cepr', 'iza'
            - Federal Reserve: 'fed', 'fed_ny', 'fed_chicago', 'fed_stlouis'
            - Top 5 Journals: 'aer', 'jpe', 'qje', 'econometrica', 'restud'
            - Other journals: 'jfe', 'jme', 'aej_macro', 'aej_micro', 'aej_applied'
    
    Returns:
        List of paper dicts with: paper_id (RePEc handle), title, authors,
        abstract, published_date, url, categories (JEL codes).
    
    Example:
        search_repec('inflation', series='nber')  # Search NBER only
        search_repec('causal inference', series='aer', sort_by='newest')
        search_repec('machine learning', series='fed', year_from=2020)
    """
    kwargs = {
        'year_from': year_from,
        'year_to': year_to,
        'search_field': search_field,
        'sort_by': sort_by,
        'doc_type': doc_type,
        'series': series,
    }
    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return await _search('repec', query, max_results, **kwargs)


@mcp.tool()
async def download_repec(paper_id: str, save_path: Optional[str] = None) -> str:
    """RePEc/IDEAS does NOT support direct PDF downloads.
    
    RePEc is a metadata index - PDFs are hosted at original institutions.
    
    INSTEAD (try in order):
    1. Visit paper URL - many NBER/Fed papers are freely available
    2. download_scihub(doi) - if published before 2023
    
    Args:
        paper_id: RePEc handle (unused).
        save_path: Unused.
    
    Returns:
        Error message with alternatives.
    """
    return await _download('repec', paper_id, save_path)


@mcp.tool()
async def read_repec_paper(paper_id: str, save_path: Optional[str] = None) -> str:
    """RePEc/IDEAS does NOT support direct paper reading.
    
    INSTEAD (try in order):
    1. Visit paper URL - many NBER/Fed papers are freely available
    2. read_scihub_paper(doi) - if published before 2023
    
    Args:
        paper_id: RePEc handle (unused).
        save_path: Unused.
    
    Returns:
        Error message with alternatives.
    """
    return await _read('repec', paper_id, save_path)


@mcp.tool()
async def get_repec_paper(url_or_handle: str) -> Dict:
    """Get detailed paper information from RePEc/IDEAS.
    
    Fetches complete metadata from an IDEAS paper detail page, including
    abstract, authors, keywords, and JEL codes that may be missing from
    search results.
    
    USE THIS WHEN:
    - You have a paper URL/handle from search results and need the abstract
    - You want complete author information for a specific paper
    - You need JEL classification codes or keywords
    
    Args:
        url_or_handle: Paper URL or RePEc handle, e.g.:
            - URL: "https://ideas.repec.org/p/nbr/nberwo/32000.html"
            - Handle: "RePEc:nbr:nberwo:32000"
    
    Returns:
        Paper dict with: paper_id, title, authors, abstract, keywords,
        categories (JEL codes), published_date, url, pdf_url (if available),
        doi (if found), and extra info like journal name.
    
    Example:
        get_repec_paper("https://ideas.repec.org/a/aea/aecrev/v110y2020i1p1-40.html")
    """
    searcher = SEARCHERS['repec']
    paper = searcher.get_paper_details(url_or_handle)
    if paper:
        return paper.to_dict()
    else:
        return {"error": f"Failed to fetch paper details from: {url_or_handle}"}

# ============================================================
# Sci-Hub 工具（仅下载，不搜索）
# ============================================================
@mcp.tool()
async def download_scihub(doi: str, save_path: Optional[str] = None) -> str:
    """Download paper PDF via Sci-Hub using DOI (for older papers only).
    
    USE THIS TOOL WHEN:
    - You have a DOI and need the full PDF
    - The paper was published BEFORE 2023
    - The paper is behind a paywall and not on arXiv
    - You first searched CrossRef and got the DOI
    
    WORKFLOW: search_crossref(query) -> get DOI -> download_scihub(doi)
    
    Args:
        doi: Paper DOI (e.g., '10.1038/nature12373', '10.1126/science.1234567').
        save_path: Directory to save PDF (default: ~/paper_downloads).
    
    Returns:
        Path to downloaded PDF file (e.g., 'downloads/scihub_10.1038_xxx.pdf'),
        or error message if download fails.
    
    Example:
        download_scihub("10.1038/nature12373")  # 2013 Nature paper
    """
    if save_path is None:
        save_path = get_download_path()
    try:
        return SCIHUB.download_pdf(doi, save_path)
    except Exception as e:
        logger.error(f"Sci-Hub download failed: {e}")
        return f"Error: {e}"


@mcp.tool()
async def read_scihub_paper(doi: str, save_path: Optional[str] = None) -> str:
    """Download and extract full text from paper via Sci-Hub (older papers only).
    
    USE THIS TOOL WHEN:
    - You need the complete text content of a paper (not just abstract)
    - The paper was published BEFORE 2023
    - You want to analyze, summarize, or answer questions about a paper
    
    This downloads the PDF and extracts text as clean Markdown format,
    suitable for LLM processing. Includes paper metadata at the start.
    
    WORKFLOW: search_crossref(query) -> get DOI -> read_scihub_paper(doi)
    
    Args:
        doi: Paper DOI (e.g., '10.1038/nature12373').
        save_path: Directory to save PDF (default: ~/paper_downloads).
    
    Returns:
        Full paper text in Markdown format with metadata header,
        or error message if download/extraction fails.
    
    Example:
        read_scihub_paper("10.1038/nature12373")
    """
    if save_path is None:
        save_path = get_download_path()
    try:
        return SCIHUB.read_paper(doi, save_path)
    except Exception as e:
        logger.error(f"Sci-Hub read failed: {e}")
        return f"Error: {e}"


# ============================================================
# 服务器入口
# ============================================================
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行 MCP 服务器
    mcp.run(transport="stdio")

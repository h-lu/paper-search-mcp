# Paper Find MCP Server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server for searching and downloading academic papers from multiple platforms. Designed for LLM tools like Claude Desktop, Cursor, etc.

[![PyPI version](https://badge.fury.io/py/paper-find-mcp.svg)](https://badge.fury.io/py/paper-find-mcp) ![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.10+-blue.svg)

**[中文文档](README_CN.md)**

---

## Supported Platforms

### Core Platforms

| Platform | Search | Download | Read | Description |
|----------|:------:|:--------:|:----:|-------------|
| **arXiv** | ✅ | ✅ | ✅ | Preprints: Physics, Math, CS, Stats, Biology, Finance |
| **Semantic Scholar** | ✅ | ✅ | ✅ | General academic search, 200M+ papers, AI-powered |
| **PubMed** | ✅ | ❌ | ❌ | Biomedical literature |
| **bioRxiv** | ✅ | ✅ | ✅ | Biology preprints |
| **medRxiv** | ✅ | ✅ | ✅ | Medical preprints |
| **CrossRef** | ✅ | ❌ | ❌ | DOI metadata, 150M+ records |
| **IACR** | ✅ | ✅ | ✅ | Cryptography papers |
| **Google Scholar** | ✅ | ❌ | ❌ | All-discipline search (web scraping) |
| **RePEc/IDEAS** | ✅ | ❌ | ❌ | Economics paper library, 4.5M+ items |
| **Sci-Hub** | ❌ | ✅ | ✅ | Download paywalled papers (pre-2023) |

### RePEc/IDEAS Features

RePEc is the largest open economics bibliography, with rich search options:

**Search Fields**: Full text / Abstract / Keywords / Title / Author

**Sort Options**: Relevance / Newest / Oldest / Citations / Recent & Relevant

**Document Types**: Journal Articles / Working Papers / Book Chapters / Books

**Institution/Journal Filters**:
| Category | Options |
|----------|---------|
| Research Institutions | `nber`, `imf`, `worldbank`, `ecb`, `bis`, `cepr`, `iza` |
| Federal Reserve | `fed`, `fed_ny`, `fed_chicago`, `fed_stlouis`, `fed_sf` |
| Top 5 Journals | `aer`, `jpe`, `qje`, `econometrica`, `restud` |
| Other Journals | `jfe`, `jme`, `aej_macro`, `aej_micro`, `aej_applied` |

---

## Quick Start

### Installation

**Install from PyPI (recommended):**

```bash
# Using uv (recommended)
uv pip install paper-find-mcp

# Or using pip
pip install paper-find-mcp
```

**Install from source:**

```bash
# Clone the repository
git clone https://github.com/h-lu/paper-find-mcp.git
cd paper-find-mcp

# Install with uv
uv pip install -e .

# Or use pip
pip install -e .
```

### Configure Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

**Using uvx (recommended, no pre-installation required):**

```json
{
  "mcpServers": {
    "paper_find_server": {
      "command": "uvx",
      "args": ["paper-find-mcp"],
      "env": {
        "SEMANTIC_SCHOLAR_API_KEY": "",
        "CROSSREF_MAILTO": "your_email@example.com",
        "NCBI_API_KEY": "",
        "PAPER_DOWNLOAD_PATH": "~/paper_downloads"
      }
    }
  }
}
```

**Using pip installed package:**

```json
{
  "mcpServers": {
    "paper_find_server": {
      "command": "paper-find-mcp",
      "args": [],
      "env": {
        "SEMANTIC_SCHOLAR_API_KEY": "",
        "CROSSREF_MAILTO": "your_email@example.com",
        "NCBI_API_KEY": "",
        "PAPER_DOWNLOAD_PATH": "~/paper_downloads"
      }
    }
  }
}
```

---

## Usage Guide

### Choose Tools by Discipline

```
General academic search    → search_semantic or search_crossref
CS/Physics preprints       → search_arxiv
Biomedical                 → search_pubmed + download_scihub(doi)
Economics                  → search_repec (supports NBER, IMF, Fed, AER, etc.)
Cryptography               → search_iacr
Download paywalled papers  → download_scihub(doi) [pre-2023]
```

### Typical Workflow

```python
# 1. Search for papers
papers = search_semantic("climate change agriculture", max_results=5)

# 2. Get DOI
doi = papers[0]["doi"]

# 3. Download via Sci-Hub (older papers)
pdf_path = download_scihub(doi)

# 4. Read full text
text = read_scihub_paper(doi)
```

### RePEc Economics Search Examples

```python
# Search NBER working papers
search_repec("inflation expectations", series='nber')

# Search AER journal articles, sorted by newest
search_repec("causal inference", series='aer', sort_by='newest')

# Search Fed papers, with year filter
search_repec("monetary policy", series='fed', year_from=2020)

# Search by author
search_repec("Acemoglu", search_field='author')

# Get paper details (with full abstract)
get_repec_paper("https://ideas.repec.org/p/nbr/nberwo/32000.html")
```

---

## Complete Tool List

### Search Tools

| Tool | Description |
|------|-------------|
| `search_arxiv` | Search arXiv preprints |
| `search_semantic` | Semantic Scholar general search |
| `search_crossref` | CrossRef DOI metadata search |
| `search_pubmed` | PubMed biomedical search |
| `search_biorxiv` | bioRxiv biology preprints |
| `search_medrxiv` | medRxiv medical preprints |
| `search_iacr` | IACR cryptography papers |
| `search_google_scholar` | Google Scholar search |
| `search_repec` | RePEc/IDEAS economics search |

### Download Tools

| Tool | Description |
|------|-------------|
| `download_arxiv` | Download arXiv PDF (free) |
| `download_semantic` | Download open access papers |
| `download_biorxiv` | Download bioRxiv PDF |
| `download_medrxiv` | Download medRxiv PDF |
| `download_iacr` | Download IACR PDF |
| `download_scihub` | Download via Sci-Hub |

### Read Tools (PDF → Markdown)

| Tool | Description |
|------|-------------|
| `read_arxiv_paper` | Read arXiv paper |
| `read_semantic_paper` | Read Semantic Scholar paper |
| `read_biorxiv_paper` | Read bioRxiv paper |
| `read_medrxiv_paper` | Read medRxiv paper |
| `read_iacr_paper` | Read IACR paper |
| `read_scihub_paper` | Read Sci-Hub downloaded paper |

### Helper Tools

| Tool | Description |
|------|-------------|
| `get_repec_paper` | Get RePEc paper details (full abstract) |
| `get_crossref_paper_by_doi` | Get paper metadata by DOI |

---

## Environment Variables

| Variable | Purpose | Recommended |
|----------|---------|:-----------:|
| `SEMANTIC_SCHOLAR_API_KEY` | Increase Semantic Scholar rate limit | ✅ |
| `CROSSREF_MAILTO` | CrossRef polite pool access | ✅ |
| `NCBI_API_KEY` | Increase PubMed rate limit | Optional |
| `SCIHUB_MIRROR` | Custom Sci-Hub mirror | Optional |
| `PAPER_DOWNLOAD_PATH` | PDF download directory (default: `~/paper_downloads`) | Optional |

---

## Development

```bash
# Clone the repository
git clone https://github.com/h-lu/paper-find-mcp.git
cd paper-find-mcp

# Create virtual environment
uv venv && source .venv/bin/activate

# Install dev dependencies
uv pip install -e .

# Run tests
uv run pytest tests/ -v
```

---

## License

MIT License

Original code based on [paper-search-mcp](https://github.com/openags/paper-search-mcp)  
Copyright (c) 2025 OPENAGS

Modifications and enhancements  
Copyright (c) 2025 Haibo Lu

---

🎓 Happy researching!

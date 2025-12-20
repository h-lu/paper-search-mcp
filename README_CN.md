# 论文搜索 MCP 服务器 (Paper Find MCP)

一个用于搜索和下载学术论文的 [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) 服务器，支持多个学术平台。专为 Claude Desktop、Cursor 等 LLM 工具设计。

[![PyPI version](https://badge.fury.io/py/paper-find-mcp.svg)](https://badge.fury.io/py/paper-find-mcp) ![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.10+-blue.svg)

**[English](README.md)**

---

## 支持的平台

### 核心平台

| 平台 | 搜索 | 下载 | 阅读 | 说明 |
|------|:----:|:----:|:----:|------|
| **arXiv** | ✅ | ✅ | ✅ | 预印本: 物理、数学、计算机、统计、生物、金融 |
| **Semantic Scholar** | ✅ | ✅ | ✅ | 通用学术搜索，2亿+论文，AI驱动 |
| **PubMed** | ✅ | ❌ | ❌ | 生物医学文献 |
| **bioRxiv** | ✅ | ✅ | ✅ | 生物学预印本 |
| **medRxiv** | ✅ | ✅ | ✅ | 医学预印本 |
| **CrossRef** | ✅ | ❌ | ❌ | DOI 元数据，1.5亿+记录 |
| **IACR** | ✅ | ✅ | ✅ | 密码学论文 |
| **Google Scholar** | ✅ | ❌ | ❌ | 全学科搜索（网页抓取） |
| **RePEc/IDEAS** | ✅ | ❌ | ❌ | 经济学论文库，450万+条目 |
| **Sci-Hub** | ❌ | ✅ | ✅ | 下载 2023 年前的付费论文 |

### RePEc/IDEAS 特色功能

RePEc 是最大的开放经济学文献库，支持丰富的搜索选项：

**搜索字段**: 全文 / 摘要 / 关键词 / 标题 / 作者

**排序方式**: 相关性 / 最新 / 最早 / 被引次数 / 最新且相关

**文档类型**: 期刊文章 / 工作论文 / 书籍章节 / 书籍

**机构/期刊过滤**:
| 类别 | 可选值 |
|------|--------|
| 研究机构 | `nber`, `imf`, `worldbank`, `ecb`, `bis`, `cepr`, `iza` |
| 美联储 | `fed`, `fed_ny`, `fed_chicago`, `fed_stlouis`, `fed_sf` |
| Top 5 期刊 | `aer`, `jpe`, `qje`, `econometrica`, `restud` |
| 其他期刊 | `jfe`, `jme`, `aej_macro`, `aej_micro`, `aej_applied` |

---

## 快速开始

### 安装

**从 PyPI 安装（推荐）：**

```bash
# 使用 uv（推荐）
uv pip install paper-find-mcp

# 或使用 pip
pip install paper-find-mcp
```

**从源码安装：**

```bash
# 克隆仓库
git clone https://github.com/h-lu/paper-find-mcp.git
cd paper-find-mcp

# 使用 uv 安装
uv pip install -e .

# 或使用 pip
pip install -e .
```

### 配置 Claude Desktop

编辑 `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

**使用 uvx（推荐，无需预先安装）：**

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

**使用 pip 安装后运行：**

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

## 使用指南

### 按学科选择工具

```
通用学术搜索      → search_semantic 或 search_crossref
计算机/物理预印本  → search_arxiv
生物医学          → search_pubmed + download_scihub(doi)
经济学            → search_repec (支持 NBER, IMF, Fed, AER 等)
密码学            → search_iacr
下载付费论文       → download_scihub(doi) [2023年前]
```

### 典型工作流

```python
# 1. 搜索论文
papers = search_semantic("climate change agriculture", max_results=5)

# 2. 获取 DOI
doi = papers[0]["doi"]

# 3. 通过 Sci-Hub 下载 (旧论文)
pdf_path = download_scihub(doi)

# 4. 阅读全文
text = read_scihub_paper(doi)
```

### RePEc 经济学搜索示例

```python
# 搜索 NBER 工作论文
search_repec("inflation expectations", series='nber')

# 搜索 AER 期刊文章，按最新排序
search_repec("causal inference", series='aer', sort_by='newest')

# 搜索美联储论文，限定年份
search_repec("monetary policy", series='fed', year_from=2020)

# 按作者搜索
search_repec("Acemoglu", search_field='author')

# 获取论文详情（包含完整摘要）
get_repec_paper("https://ideas.repec.org/p/nbr/nberwo/32000.html")
```

---

## 完整工具列表

### 搜索工具

| 工具 | 说明 |
|------|------|
| `search_arxiv` | 搜索 arXiv 预印本 |
| `search_semantic` | Semantic Scholar 通用搜索 |
| `search_crossref` | CrossRef DOI 元数据搜索 |
| `search_pubmed` | PubMed 生物医学搜索 |
| `search_biorxiv` | bioRxiv 生物学预印本 |
| `search_medrxiv` | medRxiv 医学预印本 |
| `search_iacr` | IACR 密码学论文 |
| `search_google_scholar` | Google Scholar 搜索 |
| `search_repec` | RePEc/IDEAS 经济学搜索 |

### 下载工具

| 工具 | 说明 |
|------|------|
| `download_arxiv` | 下载 arXiv PDF（免费） |
| `download_semantic` | 下载开放获取论文 |
| `download_biorxiv` | 下载 bioRxiv PDF |
| `download_medrxiv` | 下载 medRxiv PDF |
| `download_iacr` | 下载 IACR PDF |
| `download_scihub` | 通过 Sci-Hub 下载 |

### 阅读工具 (PDF → Markdown)

| 工具 | 说明 |
|------|------|
| `read_arxiv_paper` | 阅读 arXiv 论文 |
| `read_semantic_paper` | 阅读 Semantic Scholar 论文 |
| `read_biorxiv_paper` | 阅读 bioRxiv 论文 |
| `read_medrxiv_paper` | 阅读 medRxiv 论文 |
| `read_iacr_paper` | 阅读 IACR 论文 |
| `read_scihub_paper` | 阅读 Sci-Hub 下载的论文 |

### 辅助工具

| 工具 | 说明 |
|------|------|
| `get_repec_paper` | 获取 RePEc 论文详情（完整摘要） |
| `get_crossref_paper_by_doi` | 通过 DOI 获取论文元数据 |

---

## 环境变量

| 变量 | 用途 | 推荐 |
|------|------|:----:|
| `SEMANTIC_SCHOLAR_API_KEY` | 提高 Semantic Scholar 请求限制 | ✅ |
| `CROSSREF_MAILTO` | CrossRef 礼貌池访问 | ✅ |
| `NCBI_API_KEY` | 提高 PubMed 请求限制 | 可选 |
| `SCIHUB_MIRROR` | 自定义 Sci-Hub 镜像 | 可选 |
| `PAPER_DOWNLOAD_PATH` | PDF 下载目录 (默认: `~/paper_downloads`) | 可选 |

---

## 开发

```bash
# 克隆仓库
git clone https://github.com/h-lu/paper-find-mcp.git
cd paper-find-mcp

# 创建虚拟环境
uv venv && source .venv/bin/activate

# 安装开发依赖
uv pip install -e .

# 运行测试
uv run pytest tests/ -v
```

---

## 许可证

MIT License

原始代码基于 [paper-search-mcp](https://github.com/openags/paper-search-mcp)  
Copyright (c) 2025 OPENAGS

修改和增强  
Copyright (c) 2025 Haibo Lu

---

🎓 祝研究顺利！

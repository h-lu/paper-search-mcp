# paper_search_mcp/academic_platforms/semantic.py
"""
SemanticSearcher - Semantic Scholar 论文搜索

2025 最佳实践版本：
- 支持 API Key（提升速率限制，获取专用配额）
- 只请求必要字段（减少延迟和配额消耗）
- 指数退避重试机制
- 使用 PyMuPDF4LLM 提取 PDF（替代 PyPDF2）
- Session 复用
"""
from typing import List, Optional
from datetime import datetime
import requests
import time
import os
import re
import logging

from ..paper import Paper

import pymupdf4llm
import subprocess
import shutil

logger = logging.getLogger(__name__)


class PaperSource:
    """Abstract base class for paper sources"""
    def search(self, query: str, **kwargs) -> List[Paper]:
        raise NotImplementedError

    def download_pdf(self, paper_id: str, save_path: str) -> str:
        raise NotImplementedError

    def read_paper(self, paper_id: str, save_path: str) -> str:
        raise NotImplementedError


class SemanticSearcher(PaperSource):
    """Semantic Scholar 论文搜索器
    
    使用 Semantic Scholar Academic Graph API 搜索论文。
    
    2025 最佳实践：
    - API Key 提供专用速率限制（1 RPS 起步，可申请提升）
    - 只请求必要字段减少延迟
    - 指数退避处理 429 错误
    - 支持多种论文 ID 格式（DOI, arXiv, PMID 等）
    
    环境变量：
    - SEMANTIC_SCHOLAR_API_KEY: API 密钥（推荐）
    
    获取 API Key: https://www.semanticscholar.org/product/api
    """
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    # 只请求必要字段（2025 最佳实践）
    DEFAULT_FIELDS = [
        "title", "abstract", "year", "citationCount", 
        "authors", "url", "publicationDate", 
        "externalIds", "fieldsOfStudy", "openAccessPdf"
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """初始化 Semantic Scholar 搜索器
        
        Args:
            api_key: API Key（默认从环境变量获取）
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.api_key = api_key or os.environ.get('SEMANTIC_SCHOLAR_API_KEY', '')
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Session 复用
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'paper_search_mcp/1.0',
            'Accept': 'application/json',
        })
        
        # 添加 API Key 到 headers
        if self.api_key:
            self.session.headers['x-api-key'] = self.api_key
            logger.info("Using authenticated access with API key")
        else:
            logger.warning(
                "No SEMANTIC_SCHOLAR_API_KEY set. "
                "Using shared rate limit (5000 req/5min shared with all users)"
            )
        
        # 速率限制追踪
        self._last_request_time = 0.0
        # 有 API Key = 1 RPS，无 API Key = 共享池
        self.min_request_interval = 1.0 if self.api_key else 0.5

    def _rate_limit_wait(self):
        """速率限制等待"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _make_request(
        self, 
        endpoint: str, 
        params: dict,
        retry_count: int = 0
    ) -> Optional[requests.Response]:
        """发送 API 请求，带重试机制
        
        Args:
            endpoint: API 端点路径
            params: 请求参数
            retry_count: 当前重试次数
            
        Returns:
            Response 对象或 None（发生错误时）
        """
        self._rate_limit_wait()
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            
            # 处理 429 速率限制
            if response.status_code == 429:
                if retry_count < self.max_retries:
                    # 指数退避 + 随机抖动
                    wait_time = (2 ** retry_count) + (time.time() % 1)
                    logger.warning(f"Rate limited (429), retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    return self._make_request(endpoint, params, retry_count + 1)
                else:
                    logger.error(f"Rate limited after {self.max_retries} retries")
                    return None
            
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            if retry_count < self.max_retries:
                wait_time = 2 ** retry_count
                logger.warning(f"Request failed, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
                return self._make_request(endpoint, params, retry_count + 1)
            logger.error(f"Request failed after {self.max_retries} retries: {e}")
            return None

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """解析日期字符串"""
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str.strip(), "%Y-%m-%d")
        except ValueError:
            # 尝试只解析年份
            try:
                return datetime.strptime(date_str.strip()[:4], "%Y")
            except ValueError:
                return None

    def _extract_pdf_url(self, open_access_pdf: dict) -> str:
        """从 openAccessPdf 字段提取 PDF URL"""
        if not open_access_pdf:
            return ""
        
        # 直接获取 URL
        if open_access_pdf.get('url'):
            return open_access_pdf['url']
        
        # 从 disclaimer 中提取
        disclaimer = open_access_pdf.get('disclaimer', '')
        if disclaimer:
            # 匹配 URL 模式
            url_pattern = r'https?://[^\s,)"]+'
            matches = re.findall(url_pattern, disclaimer)
            
            if matches:
                # 优先返回 DOI 或 arXiv URL
                for url in matches:
                    if 'doi.org' in url or 'arxiv.org' in url:
                        # 转换 arXiv abs 链接为 PDF 链接
                        if 'arxiv.org/abs/' in url:
                            return url.replace('/abs/', '/pdf/') + '.pdf'
                        return url
                return matches[0]
        
        return ""

    def _parse_paper(self, data: dict) -> Optional[Paper]:
        """解析论文数据
        
        Args:
            data: API 返回的论文数据
            
        Returns:
            Paper 对象或 None
        """
        try:
            paper_id = data.get('paperId', '')
            if not paper_id:
                return None
            
            # 作者
            authors = [
                author.get('name', '') 
                for author in data.get('authors', [])
                if author.get('name')
            ]
            
            # DOI
            external_ids = data.get('externalIds', {}) or {}
            doi = external_ids.get('DOI', '')
            
            # PDF URL
            pdf_url = self._extract_pdf_url(data.get('openAccessPdf'))
            
            return Paper(
                paper_id=paper_id,
                title=data.get('title', 'Untitled'),
                authors=authors,
                abstract=data.get('abstract', ''),
                url=data.get('url', ''),
                pdf_url=pdf_url,
                published_date=self._parse_date(data.get('publicationDate', '')),
                source="semantic",
                categories=data.get('fieldsOfStudy', []) or [],
                doi=doi,
                citations=data.get('citationCount', 0) or 0,
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse paper: {e}")
            return None

    def search(
        self, 
        query: str, 
        year: Optional[str] = None, 
        max_results: int = 10
    ) -> List[Paper]:
        """搜索论文
        
        Args:
            query: 搜索关键词
            year: 年份过滤（支持格式："2019", "2016-2020", "2010-", "-2015"）
            max_results: 最大返回数量
            
        Returns:
            List[Paper]: 论文列表
        """
        params = {
            "query": query,
            "limit": min(max_results, 100),  # API 限制
            "fields": ",".join(self.DEFAULT_FIELDS),
        }
        
        if year:
            params["year"] = year
        
        response = self._make_request("paper/search", params)
        if not response:
            return []
        
        try:
            data = response.json()
            results = data.get('data', [])
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            return []
        
        papers = []
        for item in results[:max_results]:
            paper = self._parse_paper(item)
            if paper:
                papers.append(paper)
        
        logger.info(f"Found {len(papers)} papers for query: {query}")
        return papers

    def get_paper_details(self, paper_id: str) -> Optional[Paper]:
        """获取单篇论文详情
        
        Args:
            paper_id: 论文 ID，支持多种格式：
                - Semantic Scholar ID: "649def34f8be52c8b66281af98ae884c09aef38b"
                - DOI: "DOI:10.18653/v1/N18-3011"
                - arXiv: "ARXIV:2106.15928"
                - PMID: "PMID:19872477"
                - ACL: "ACL:W12-3903"
                - URL: "URL:https://arxiv.org/abs/2106.15928"
                
        Returns:
            Paper 对象或 None
        """
        params = {"fields": ",".join(self.DEFAULT_FIELDS)}
        
        response = self._make_request(f"paper/{paper_id}", params)
        if not response:
            return None
        
        try:
            data = response.json()
            return self._parse_paper(data)
        except Exception as e:
            logger.error(f"Failed to get paper details: {e}")
            return None

    def _download_with_curl(self, url: str, pdf_path: str) -> bool:
        """使用 curl 下载 PDF（更可靠）
        
        Args:
            url: PDF URL
            pdf_path: 保存路径
            
        Returns:
            是否成功
        """
        if not shutil.which('curl'):
            return False
        
        try:
            result = subprocess.run(
                [
                    'curl', '-L',  # 跟随重定向
                    '-o', pdf_path,
                    '--connect-timeout', '30',
                    '--max-time', '300',  # 最大 5 分钟
                    '-f',  # 失败时返回错误码
                    '-s',  # 静默模式
                    '--retry', '3',
                    '--retry-delay', '2',
                    url
                ],
                capture_output=True,
                text=True,
                timeout=360
            )
            
            if result.returncode == 0 and os.path.exists(pdf_path):
                file_size = os.path.getsize(pdf_path)
                if file_size > 1000:  # 至少 1KB
                    logger.info(f"PDF downloaded with curl: {pdf_path} ({file_size} bytes)")
                    return True
                else:
                    os.remove(pdf_path)
                    logger.warning(f"Downloaded file too small: {file_size} bytes")
                    return False
            else:
                logger.warning(f"curl failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.warning("curl download timed out")
            return False
        except Exception as e:
            logger.warning(f"curl error: {e}")
            return False

    def download_pdf(self, paper_id: str, save_path: str) -> str:
        """下载论文 PDF
        
        优先使用 curl（更可靠），失败时回退到 requests。
        
        Args:
            paper_id: 论文 ID（支持多种格式）
            save_path: 保存目录
            
        Returns:
            下载的文件路径或错误信息
        """
        paper = self.get_paper_details(paper_id)
        if not paper:
            return f"Error: Could not find paper {paper_id}"
        
        if not paper.pdf_url:
            return f"Error: No PDF URL available for paper {paper_id}"
        
        # 准备保存路径
        os.makedirs(save_path, exist_ok=True)
        safe_id = paper_id.replace('/', '_').replace(':', '_')
        filename = f"semantic_{safe_id}.pdf"
        pdf_path = os.path.join(save_path, filename)
        
        # 方法1: 优先使用 curl（更可靠）
        if self._download_with_curl(paper.pdf_url, pdf_path):
            return pdf_path
        
        logger.info("curl failed, falling back to requests...")
        
        # 方法2: 回退到 requests
        download_timeout = (30, 180)  # 连接 30s，读取 180s
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    paper.pdf_url, 
                    timeout=download_timeout,
                    stream=True
                )
                response.raise_for_status()
                
                with open(pdf_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                logger.info(f"PDF downloaded with requests: {pdf_path}")
                return pdf_path
                
            except requests.exceptions.Timeout as e:
                logger.warning(f"Download timeout (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Download failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return f"Error downloading PDF: All methods failed for {paper.pdf_url}"

    def read_paper(self, paper_id: str, save_path: str) -> str:
        """下载并提取论文文本
        
        使用 PyMuPDF4LLM 提取 Markdown 格式。
        
        Args:
            paper_id: 论文 ID
            save_path: 保存目录
            
        Returns:
            提取的文本内容或错误信息
        """
        # 先下载 PDF
        pdf_path = self.download_pdf(paper_id, save_path)
        if pdf_path.startswith("Error"):
            return pdf_path
        
        # 获取论文元数据
        paper = self.get_paper_details(paper_id)
        
        try:
            text = pymupdf4llm.to_markdown(pdf_path, show_progress=False)
            logger.info(f"Extracted {len(text)} characters using PyMuPDF4LLM")
            
            if not text.strip():
                return f"PDF downloaded to {pdf_path}, but no text could be extracted."
            
            # 添加元数据
            metadata = ""
            if paper:
                metadata = f"# {paper.title}\n\n"
                metadata += f"**Authors**: {', '.join(paper.authors)}\n"
                metadata += f"**Published**: {paper.published_date}\n"
                metadata += f"**URL**: {paper.url}\n"
                metadata += f"**PDF**: {pdf_path}\n\n"
                metadata += "---\n\n"
            
            return metadata + text
            
        except Exception as e:
            logger.error(f"Failed to extract text: {e}")
            return f"Error extracting text: {e}"


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    searcher = SemanticSearcher()
    
    # 配置信息
    print("=" * 60)
    print("SemanticSearcher Configuration")
    print("=" * 60)
    print(f"API Key: {'Configured' if searcher.api_key else 'Not set (shared rate limit)'}")
    print(f"PDF Extraction: PyMuPDF4LLM")
    
    # 测试搜索
    print("\n" + "=" * 60)
    print("1. Testing search...")
    print("=" * 60)
    
    query = "machine learning"
    papers = searcher.search(query, max_results=3)
    
    print(f"Found {len(papers)} papers:")
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper.title[:60]}...")
        print(f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
        print(f"   DOI: {paper.doi or 'N/A'}")
        print(f"   Citations: {paper.citations}")
        print(f"   PDF: {'Available' if paper.pdf_url else 'Not available'}")
    
    # 测试获取详情
    print("\n" + "=" * 60)
    print("2. Testing get_paper_details...")
    print("=" * 60)
    
    if papers:
        paper_id = papers[0].paper_id
        details = searcher.get_paper_details(paper_id)
        if details:
            print(f"Title: {details.title}")
            print(f"Abstract: {details.abstract[:200]}..." if details.abstract else "No abstract")
    
    print("\n✅ All tests completed!")
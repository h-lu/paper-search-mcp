# paper_search_mcp/academic_platforms/sci_hub.py
"""
SciHubFetcher - Sci-Hub PDF 下载器

通过 Sci-Hub 下载学术论文 PDF（仅限 2023 年之前发表的论文）。

注意：Sci-Hub 的使用可能在某些地区受到法律限制。
请确保您在使用前了解当地法律法规。
"""
from pathlib import Path
import re
import hashlib
import logging
import os
from typing import Optional
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import pymupdf4llm

logger = logging.getLogger(__name__)


# Sci-Hub 可用镜像列表
SCIHUB_MIRRORS = [
    "https://sci-hub.se",
    "https://sci-hub.st", 
    "https://sci-hub.ru",
]


class SciHubFetcher:
    """Sci-Hub PDF 下载器
    
    通过 DOI 从 Sci-Hub 下载论文 PDF。
    
    限制：
    - 仅支持 2023 年之前发表的论文
    - 需要有效的 DOI
    
    环境变量：
    - SCIHUB_MIRROR: 自定义 Sci-Hub 镜像地址
    """

    def __init__(
        self, 
        base_url: Optional[str] = None, 
        timeout: int = 30
    ):
        """初始化 Sci-Hub 下载器
        
        Args:
            base_url: Sci-Hub 镜像地址（默认从环境变量或使用默认镜像）
            timeout: 请求超时时间
        """
        self.base_url = (
            base_url or 
            os.environ.get('SCIHUB_MIRROR') or 
            SCIHUB_MIRRORS[0]
        ).rstrip("/")
        self.timeout = timeout
        
        self.session = requests.Session()
        self.session.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        logger.info(f"SciHub initialized with mirror: {self.base_url}")

    def download_pdf(self, doi: str, save_path: Optional[str] = None) -> str:
        """通过 DOI 下载论文 PDF
        
        Args:
            doi: 论文 DOI（如 "10.1038/nature12373"）
            save_path: 保存目录（默认 ~/paper_downloads）
        
        Returns:
            下载的文件路径或错误信息
        """
        if not doi or not doi.strip():
            return "Error: DOI is empty"
        
        doi = doi.strip()
        # 如果未指定路径，使用用户主目录下的 paper_downloads
        output_dir = Path(save_path) if save_path else Path.home() / "paper_downloads"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 获取 PDF URL
            pdf_url = self._get_pdf_url(doi)
            if not pdf_url:
                return f"Error: Could not find PDF for DOI {doi} on Sci-Hub"
            
            # 下载 PDF
            response = self.session.get(pdf_url, verify=False, timeout=self.timeout)
            
            if response.status_code != 200:
                return f"Error: Download failed with status {response.status_code}"
            
            # 验证是 PDF
            content_type = response.headers.get('Content-Type', '')
            if 'pdf' not in content_type.lower() and not response.content[:4] == b'%PDF':
                return f"Error: Response is not a PDF (Content-Type: {content_type})"
            
            # 保存文件
            filename = self._generate_filename(doi, response)
            file_path = output_dir / filename
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"PDF downloaded: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Download failed for {doi}: {e}")
            return f"Error downloading PDF: {e}"

    def read_paper(self, doi: str, save_path: Optional[str] = None) -> str:
        """下载并提取论文文本
        
        Args:
            doi: 论文 DOI
            save_path: 保存目录
            
        Returns:
            提取的 Markdown 文本或错误信息
        """
        # 先下载 PDF
        result = self.download_pdf(doi, save_path)
        if result.startswith("Error"):
            return result
        
        pdf_path = result
        
        try:
            text = pymupdf4llm.to_markdown(pdf_path, show_progress=False)
            logger.info(f"Extracted {len(text)} characters from {pdf_path}")
            
            if not text.strip():
                return f"PDF downloaded to {pdf_path}, but no text could be extracted."
            
            # 添加元数据
            metadata = f"# Paper: {doi}\n\n"
            metadata += f"**DOI**: https://doi.org/{doi}\n"
            metadata += f"**PDF**: {pdf_path}\n"
            metadata += f"**Source**: Sci-Hub\n\n"
            metadata += "---\n\n"
            
            return metadata + text
            
        except Exception as e:
            logger.error(f"Failed to extract text: {e}")
            return f"Error extracting text: {e}"

    def _get_pdf_url(self, doi: str) -> Optional[str]:
        """从 Sci-Hub 获取 PDF 直链"""
        try:
            search_url = f"{self.base_url}/{doi}"
            response = self.session.get(search_url, verify=False, timeout=self.timeout)
            
            if response.status_code != 200:
                logger.warning(f"Sci-Hub returned status {response.status_code}")
                return None
            
            # 检查是否找到文章
            if "article not found" in response.text.lower():
                logger.warning(f"Article not found on Sci-Hub: {doi}")
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 方法 1: embed 标签
            embed = soup.find('embed', {'type': 'application/pdf'})
            if embed and embed.get('src'):
                return self._normalize_url(embed['src'])
            
            # 方法 2: iframe
            iframe = soup.find('iframe')
            if iframe and iframe.get('src'):
                return self._normalize_url(iframe['src'])
            
            # 方法 3: 下载按钮
            for button in soup.find_all('button'):
                onclick = button.get('onclick', '')
                if 'pdf' in onclick.lower():
                    match = re.search(r"location\.href='([^']+)'", onclick)
                    if match:
                        return self._normalize_url(match.group(1))
            
            # 方法 4: 直接链接
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'pdf' in href.lower() or href.endswith('.pdf'):
                    return self._normalize_url(href)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting PDF URL: {e}")
            return None

    def _normalize_url(self, url: str) -> str:
        """规范化 URL"""
        if url.startswith('//'):
            return 'https:' + url
        elif url.startswith('/'):
            return self.base_url + url
        return url

    def _generate_filename(self, doi: str, response: requests.Response) -> str:
        """生成文件名"""
        # 清理 DOI 作为文件名
        clean_doi = re.sub(r'[^\w\-_.]', '_', doi)
        # 添加短哈希以避免冲突
        content_hash = hashlib.md5(response.content).hexdigest()[:6]
        return f"scihub_{clean_doi}_{content_hash}.pdf"


def check_paper_year(published_date: Optional[datetime], cutoff_year: int = 2023) -> bool:
    """检查论文是否在截止年份之前发表
    
    Args:
        published_date: 发表日期
        cutoff_year: 截止年份（默认 2023）
        
    Returns:
        True 如果论文在截止年份之前发表
    """
    if not published_date:
        return False
    return published_date.year < cutoff_year


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 禁用 SSL 警告（仅测试用）
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    fetcher = SciHubFetcher()
    
    print("=" * 60)
    print("Testing SciHubFetcher...")
    print("=" * 60)
    
    # 测试一个已知的老论文 DOI
    test_doi = "10.1038/nature12373"  # 2013 年的论文
    
    print(f"\nDownloading: {test_doi}")
    result = fetcher.download_pdf(test_doi)
    print(f"Result: {result}")
    
    if not result.startswith("Error"):
        print("\n✅ Download successful!")
    else:
        print("\n❌ Download failed (this may be due to Sci-Hub availability)")
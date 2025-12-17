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
import subprocess
import shutil
from typing import Optional
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import pymupdf4llm

logger = logging.getLogger(__name__)


# Sci-Hub 可用镜像列表（按可用性排序，2024/2025 更新）
SCIHUB_MIRRORS = [
    "https://sci-hub.ru",
    "https://sci-hub.wf",
    "https://sci-hub.ren",
    "https://sci-hub.se",
    "https://sci-hub.st"
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
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        logger.info(f"SciHub initialized with mirror: {self.base_url}")

    def _download_with_curl(self, url: str, file_path: str) -> bool:
        """使用 curl 下载 PDF（更可靠）
        
        Args:
            url: PDF URL
            file_path: 保存路径
            
        Returns:
            是否成功
        """
        if not shutil.which('curl'):
            return False
        
        try:
            result = subprocess.run(
                [
                    'curl', '-L',  # 跟随重定向
                    '-o', file_path,
                    '--connect-timeout', '30',
                    '--max-time', '300',  # 最大 5 分钟
                    '-k',  # 允许不安全的 SSL（Sci-Hub 证书问题）
                    '-f',  # 失败时返回错误码
                    '-s',  # 静默模式
                    '--retry', '3',
                    '--retry-delay', '2',
                    '-A', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    url
                ],
                capture_output=True,
                text=True,
                timeout=360
            )
            
            if result.returncode == 0 and os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                # 检查是否是有效的 PDF（至少 10KB，且以 %PDF 开头）
                if file_size > 10000:
                    with open(file_path, 'rb') as f:
                        header = f.read(4)
                    if header == b'%PDF':
                        logger.info(f"PDF downloaded with curl: {file_path} ({file_size} bytes)")
                        return True
                    else:
                        logger.warning(f"Downloaded file is not a PDF")
                        os.remove(file_path)
                        return False
                else:
                    logger.warning(f"Downloaded file too small: {file_size} bytes")
                    os.remove(file_path)
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

    def download_pdf(self, doi: str, save_path: Optional[str] = None) -> str:
        """通过 DOI 下载论文 PDF
        
        优先使用 curl（更可靠），失败时回退到 requests。
        
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
            # 获取 PDF URL（必须用 requests 解析 HTML）
            pdf_url = self._get_pdf_url(doi)
            if not pdf_url:
                return f"Error: Could not find PDF for DOI {doi} on Sci-Hub"
            
            # 生成文件路径
            clean_doi = re.sub(r'[^\w\-_.]', '_', doi)
            file_path = output_dir / f"scihub_{clean_doi}.pdf"
            
            # 方法1: 优先使用 curl（更可靠）
            if self._download_with_curl(pdf_url, str(file_path)):
                return str(file_path)
            
            logger.info("curl failed, falling back to requests...")
            
            # 方法2: 回退到 requests（带重试）
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.session.get(
                        pdf_url, 
                        verify=False, 
                        timeout=(30, 180),  # 连接 30s，读取 180s
                        stream=True
                    )
                    
                    if response.status_code != 200:
                        logger.warning(f"Download failed with status {response.status_code}")
                        continue
                    
                    # 流式写入
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    # 验证是 PDF
                    with open(file_path, 'rb') as f:
                        header = f.read(4)
                    
                    if header != b'%PDF':
                        logger.warning("Downloaded file is not a PDF")
                        os.remove(file_path)
                        continue
                    
                    logger.info(f"PDF downloaded with requests: {file_path}")
                    return str(file_path)
                    
                except requests.exceptions.Timeout:
                    logger.warning(f"Timeout (attempt {attempt + 1}/{max_retries})")
                except Exception as e:
                    logger.warning(f"Download error (attempt {attempt + 1}/{max_retries}): {e}")
            
            return f"Error: Could not download PDF for DOI {doi}"
            
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
            # 如果已经是直接的 PDF URL，直接返回
            if doi.endswith('.pdf'):
                return doi
            
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
            
            # 方法 1: embed 标签（现代 Sci-Hub 最常用）
            embed = soup.find('embed', {'type': 'application/pdf'})
            logger.debug(f"Found embed tag: {embed}")
            if embed:
                src = embed.get('src') if hasattr(embed, 'get') else None
                logger.debug(f"Embed src: {src}")
                if src and isinstance(src, str):
                    pdf_url = self._normalize_url(src)
                    logger.debug(f"Returning PDF URL from embed: {pdf_url}")
                    return pdf_url
            
            # 方法 2: iframe（回退方案）
            iframe = soup.find('iframe')
            if iframe:
                src = iframe.get('src') if hasattr(iframe, 'get') else None
                if src and isinstance(src, str):
                    pdf_url = self._normalize_url(src)
                    logger.debug(f"Returning PDF URL from iframe: {pdf_url}")
                    return pdf_url
            
            # 方法 3: 下载按钮的 onclick
            for button in soup.find_all('button'):
                onclick = button.get('onclick', '') if hasattr(button, 'get') else ''
                if isinstance(onclick, str) and 'pdf' in onclick.lower():
                    match = re.search(r"location\.href='([^']+)'", onclick)
                    if match:
                        pdf_url = self._normalize_url(match.group(1))
                        logger.debug(f"Returning PDF URL from button: {pdf_url}")
                        return pdf_url
            
            # 方法 4: 直接下载链接
            for link in soup.find_all('a'):
                href = link.get('href', '') if hasattr(link, 'get') else ''
                if isinstance(href, str) and href and ('pdf' in href.lower() or href.endswith('.pdf')):
                    if href.startswith('http'):
                        logger.debug(f"Returning PDF URL from link: {href}")
                        return href
                    else:
                        pdf_url = self._normalize_url(href)
                        logger.debug(f"Returning PDF URL from link: {pdf_url}")
                        return pdf_url
            
            logger.warning(f"No PDF URL found in Sci-Hub page for: {doi}")
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
    test_doi = "10.1080/13504851.2021.1890325"  # 2021 年的论文
    
    print(f"\nDownloading: {test_doi}")
    result = fetcher.download_pdf(test_doi)
    print(f"Result: {result}")
    
    if not result.startswith("Error"):
        print("\n✅ Download successful!")
    else:
        print("\n❌ Download failed (this may be due to Sci-Hub availability)")
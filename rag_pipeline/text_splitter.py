from __future__ import annotations
import re
from typing import List, Tuple, Dict, Any, Optional
from langchain.schema import Document


class MarkdownHeaderTextSplitter:
    """
    Markdown 헤더를 기준으로 텍스트를 분할하는 클래스
    LangChain의 MarkdownHeaderTextSplitter와 유사한 기능을 제공
    """
    
    def __init__(
        self,
        headers_to_split: List[Tuple[str, str]],
        strip_headers: bool = True,
        return_each_line: bool = False
    ):
        """
        Args:
            headers_to_split: 분할할 헤더들의 리스트 [("##", "Header2"), ("#", "Header1")]
            strip_headers: 결과에서 헤더를 제거할지 여부
            return_each_line: 각 라인을 개별 문서로 반환할지 여부
        """
        self.headers_to_split = sorted(headers_to_split, key=lambda x: len(x[0]), reverse=True)
        self.strip_headers = strip_headers
        self.return_each_line = return_each_line
        
    def split_text(self, text: str) -> List[Document]:
        """
        텍스트를 Markdown 헤더 기준으로 분할
        
        Args:
            text: 분할할 마크다운 텍스트
            
        Returns:
            분할된 Document 객체들의 리스트
        """
        if not text.strip():
            return []
            
        lines = text.split('\n')
        documents = []
        current_content = []
        current_metadata = {}
        
        for line in lines:
            stripped_line = line.strip()
            
            # 헤더인지 확인
            header_info = self._get_header_info(stripped_line)
            
            if header_info:
                # 이전 섹션이 있다면 문서로 저장
                if current_content or current_metadata:
                    content = self._build_content(current_content)
                    if content.strip():
                        documents.append(Document(
                            page_content=content,
                            metadata=current_metadata.copy()
                        ))
                
                # 새 섹션 시작
                header_level, header_name, header_text = header_info
                current_content = []
                
                # 메타데이터 업데이트 (현재 레벨과 상위 레벨 유지)
                current_metadata = self._update_metadata(current_metadata, header_level, header_name, header_text)
                
                # 헤더를 콘텐츠에 포함할지 결정
                if not self.strip_headers:
                    current_content.append(line)
            else:
                # 일반 텍스트 라인
                current_content.append(line)
        
        # 마지막 섹션 처리
        if current_content or current_metadata:
            content = self._build_content(current_content)
            if content.strip():
                documents.append(Document(
                    page_content=content,
                    metadata=current_metadata.copy()
                ))
        
        return documents
    
    def _get_header_info(self, line: str) -> Optional[Tuple[int, str, str]]:
        """
        라인이 헤더인지 확인하고 헤더 정보 반환
        
        Returns:
            (header_level, header_name, header_text) 또는 None
        """
        for header_marker, header_name in self.headers_to_split:
            if line.startswith(header_marker + " "):
                header_level = len(header_marker)
                header_text = line[len(header_marker):].strip()
                return (header_level, header_name, header_text)
        return None
    
    def _update_metadata(
        self, 
        current_metadata: Dict[str, Any], 
        new_header_level: int, 
        new_header_name: str, 
        new_header_text: str
    ) -> Dict[str, Any]:
        """
        메타데이터를 업데이트 (헤더 계층 구조 유지)
        """
        new_metadata = {}
        
        # 현재 레벨보다 상위 레벨의 헤더들 유지
        for key, value in current_metadata.items():
            if key.startswith("Header"):
                try:
                    existing_level = int(key.replace("Header", ""))
                    if existing_level < new_header_level:
                        new_metadata[key] = value
                except ValueError:
                    new_metadata[key] = value
        
        # 새 헤더 추가
        new_metadata[new_header_name] = new_header_text
        
        return new_metadata
    
    def _build_content(self, content_lines: List[str]) -> str:
        """
        콘텐츠 라인들을 하나의 문자열로 결합
        """
        if self.return_each_line:
            return '\n'.join(line for line in content_lines if line.strip())
        else:
            return '\n'.join(content_lines)


class RecursiveCharacterTextSplitter:
    """
    문자 수 기반으로 텍스트를 재귀적으로 분할하는 클래스
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        keep_separator: bool = False
    ):
        """
        Args:
            chunk_size: 청크의 최대 크기
            chunk_overlap: 청크 간 겹침 크기
            separators: 분할에 사용할 구분자들
            keep_separator: 구분자를 결과에 포함할지 여부
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.keep_separator = keep_separator
        
        if separators is None:
            self.separators = ["\n\n", "\n", " ", ""]
        else:
            self.separators = separators
    
    def split_text(self, text: str) -> List[str]:
        """
        텍스트를 청크로 분할
        
        Args:
            text: 분할할 텍스트
            
        Returns:
            분할된 텍스트 청크들의 리스트
        """
        return self._split_text_recursive(text, self.separators)
    
    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """
        재귀적으로 텍스트 분할
        """
        final_chunks = []
        
        # 현재 구분자
        separator = separators[0] if separators else ""
        new_separators = separators[1:] if len(separators) > 1 else []
        
        # 구분자로 텍스트 분할
        if separator:
            splits = text.split(separator)
        else:
            splits = [text]
        
        # 각 분할에 대해 처리
        good_splits = []
        for split in splits:
            if len(split) <= self.chunk_size:
                good_splits.append(split)
            else:
                if good_splits:
                    # 현재까지의 좋은 분할들을 병합
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                
                # 큰 분할은 다음 구분자로 재귀 처리
                if new_separators:
                    other_chunks = self._split_text_recursive(split, new_separators)
                    final_chunks.extend(other_chunks)
                else:
                    # 더 이상 구분자가 없으면 강제로 자름
                    final_chunks.append(split[:self.chunk_size])
        
        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)
        
        return final_chunks
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """
        분할된 텍스트들을 청크 크기에 맞게 병합
        """
        chunks = []
        current_chunk = ""
        
        for split in splits:
            if not split.strip():
                continue
                
            # 현재 청크에 추가했을 때의 크기 계산
            if current_chunk:
                potential_chunk = current_chunk + separator + split
            else:
                potential_chunk = split
            
            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # 현재 청크가 있으면 저장
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = split
        
        # 마지막 청크 저장
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def create_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[Document]:
        """
        텍스트 리스트로부터 Document 객체들 생성
        """
        documents = []
        metadatas = metadatas or [{}] * len(texts)
        
        for i, text in enumerate(texts):
            chunks = self.split_text(text)
            for chunk in chunks:
                if chunk.strip():
                    documents.append(Document(
                        page_content=chunk,
                        metadata=metadatas[i] if i < len(metadatas) else {}
                    ))
        
        return documents


class TokenTextSplitter:
    """
    토큰 수 기반으로 텍스트를 분할하는 클래스
    """
    
    def __init__(
        self,
        encoding_name: str = "gpt2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Args:
            encoding_name: 사용할 인코딩 이름
            chunk_size: 청크의 최대 토큰 수
            chunk_overlap: 청크 간 겹침 토큰 수
        """
        try:
            import tiktoken
            self.encoding = tiktoken.get_encoding(encoding_name)
        except ImportError:
            print("Warning: tiktoken not installed. Using character-based approximation.")
            self.encoding = None
            
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """
        텍스트를 토큰 수 기준으로 분할
        """
        if self.encoding:
            return self._split_text_with_tiktoken(text)
        else:
            # tiktoken이 없으면 문자 수로 근사
            return self._split_text_approximate(text)
    
    def _split_text_with_tiktoken(self, text: str) -> List[str]:
        """
        tiktoken을 사용한 정확한 토큰 기반 분할
        """
        tokens = self.encoding.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # 겹침을 고려하여 다음 시작점 설정
            start = end - self.chunk_overlap
            if start >= end:
                break
        
        return chunks
    
    def _split_text_approximate(self, text: str) -> List[str]:
        """
        문자 수 기반 근사 분할 (토큰 ≈ 문자수/4)
        """
        approx_char_size = self.chunk_size * 4
        approx_char_overlap = self.chunk_overlap * 4
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + approx_char_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            
            start = end - approx_char_overlap
            if start >= end:
                break
        
        return chunks


# 편의를 위한 팩토리 함수들
def create_markdown_splitter(
    headers_to_split: List[Tuple[str, str]] = None,
    strip_headers: bool = True
) -> MarkdownHeaderTextSplitter:
    """
    기본 설정의 Markdown 헤더 분할기 생성
    """
    if headers_to_split is None:
        headers_to_split = [
            ("#", "Header1"),
            ("##", "Header2"), 
            ("###", "Header3"),
            ("####", "Header4")
        ]
    
    return MarkdownHeaderTextSplitter(
        headers_to_split=headers_to_split,
        strip_headers=strip_headers
    )


def create_recursive_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> RecursiveCharacterTextSplitter:
    """
    기본 설정의 재귀적 문자 분할기 생성
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

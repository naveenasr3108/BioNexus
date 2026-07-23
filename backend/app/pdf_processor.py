import fitz  # PyMuPDF
import re
import os

class PDFProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_and_metadata(self, pdf_path: str) -> dict:
        """Opens the PDF, pulls metadata + per-page text."""
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        full_text = ""
        pages = []

        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            pages.append({"page_number": page_num + 1, "text": text})
            full_text += text + "\n"

        doc.close()

        return {
            "title": metadata.get("title") or os.path.basename(pdf_path),
            "author": metadata.get("author", "Unknown"),
            "num_pages": len(pages),
            "full_text": self.clean_text(full_text),
            "pages": pages
        }

    def clean_text(self, text: str) -> str:
        """Removes PDF extraction artifacts before the text hits SciBERT/BGE-M3."""
        text = re.sub(r'-\n', '', text)             # rejoin hyphenated words split across lines
        text = re.sub(r'\n+', ' ', text)             # collapse newlines
        text = re.sub(r'\s+', ' ', text)             # collapse multiple spaces/tabs
        text = re.sub(r'\[\d+(,\s*\d+)*\]', '', text)  # strip citation markers like [12, 13]
        text = re.sub(r'Page \d+ of \d+', '', text)  # strip page-number footers, if present
        return text.strip()

    def chunk_text(self, text: str) -> list[str]:
        """
        Sliding-window word chunking. This is what feeds both:
        - SciBERT classification (needs the full text, truncated internally)
        - BGE-M3 embeddings (needs per-chunk vectors, this is where 'num_chunks' comes from)
        """
        words = text.split()
        chunks = []
        i = 0
        step = self.chunk_size - self.chunk_overlap
        while i < len(words):
            chunk = words[i:i + self.chunk_size]
            if len(chunk) < 20:  # drop tiny trailing fragments
                break
            chunks.append(" ".join(chunk))
            i += step
        return chunks

    def extract_sections(self, text: str) -> dict:
        """
        Optional: splits into Abstract/Methods/Results/etc. Not required for Review 1's
        demo output, but useful later (e.g. classifying off the Abstract alone tends to
        be cleaner than the full paper, and Review 3's NLI works best on Results/Conclusion text).
        """
        sections = {}
        pattern = r'(Abstract|Introduction|Methods?|Results?|Discussion|Conclusion)s?\s*[:\n]'
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        for idx, m in enumerate(matches):
            start = m.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            sections[m.group(1).lower()] = text[start:end].strip()
        return sections
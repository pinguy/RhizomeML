import os
import json
from pathlib import Path
from pdfminer.high_level import extract_text
from pdfminer import layout
import fitz  # PyMuPDF as alternative
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, pdf_dir="./PDFs", use_pymupdf=True, max_workers=4):
        self.pdf_dir = Path(pdf_dir)
        self.use_pymupdf = use_pymupdf
        self.max_workers = max_workers
        
    def extract_text_pymupdf(self, pdf_path):
        """Extract text using PyMuPDF (faster and more reliable)"""
        try:
            doc = fitz.open(pdf_path)
            text_blocks = []
            
            for page_num in range(len(doc)):  # Use len(doc) instead of doc.page_count
                page = doc[page_num]  # Use doc[page_num] instead of doc.page(page_num)
                text = page.get_text()
                if text.strip():
                    text_blocks.append({
                        "page": page_num + 1,
                        "text": text.strip()
                    })
            
            doc.close()
            return text_blocks
            
        except Exception as e:
            logger.error(f"PyMuPDF failed for {pdf_path}: {e}")
            return None
    
    def extract_text_pdfminer(self, pdf_path):
        """Extract text using pdfminer.six (as a fallback)"""
        try:
            # Using StringIO to capture output
            text_content = extract_text(pdf_path)
            
            if text_content.strip():
                return [{"page": 1, "text": text_content.strip()}] # pdfminer often gives whole text
            return None
        except Exception as e:
            logger.error(f"PDFMiner failed for {pdf_path}: {e}")
            return None

    def process_pdf(self, pdf_path):
        """Process a single PDF file, trying PyMuPDF first, then pdfminer.six"""
        filename = pdf_path.name
        logger.info(f"⚙️ Processing {filename}...")
        
        text_blocks = None
        if self.use_pymupdf:
            text_blocks = self.extract_text_pymupdf(pdf_path)
        
        if not text_blocks:
            logger.warning(f"PyMuPDF failed or not used for {filename}, trying PDFMiner...")
            text_blocks = self.extract_text_pdfminer(pdf_path)
            
        if text_blocks:
            total_text = "\n\n".join([block["text"] for block in text_blocks])
            logger.info(f"✅ Extracted {len(text_blocks)} blocks from {filename}")
            return {
                "filename": filename,
                "total_text": total_text,
                "pages": text_blocks
            }
        else:
            logger.error(f"❌ Failed to extract text from {filename} using all methods.")
            return {
                "filename": filename,
                "total_text": "",
                "pages": [],
                "status": "failed"
            }

    def process_all_pdfs(self):
        """Process all PDF files in the specified directory using a thread pool"""
        if not self.pdf_dir.exists():
            logger.warning(f"PDF directory not found: {self.pdf_dir}")
            return []
            
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        if not pdf_files:
            logger.info(f"No PDF files found in {self.pdf_dir}")
            return []

        logger.info(f"Found {len(pdf_files)} PDFs in {self.pdf_dir}. Starting processing...")
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_pdf = {executor.submit(self.process_pdf, pdf_file): pdf_file for pdf_file in pdf_files}
            for future in as_completed(future_to_pdf):
                pdf_file = future_to_pdf[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as exc:
                    logger.error(f'{pdf_file} generated an exception: {exc}')
        
        return results
    
    def save_results(self, results, output_file="pdf_texts.json"):
        """Save results to JSON file"""
        output_path = Path(output_file)
        
        # Create detailed and simple versions
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Create a simple version with just filename and full text
        simple_results = [
            {"filename": item["filename"], "text": item["total_text"]}
            for item in results
        ]
        
        simple_path = output_path.with_stem(output_path.stem + "_simple")
        with open(simple_path, "w", encoding="utf-8") as f:
            json.dump(simple_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Saved {len(results)} processed PDFs to {output_path}")
        logger.info(f"✅ Saved simple version to {simple_path}")

def main():
    processor = PDFProcessor()
    results = processor.process_all_pdfs()
    processor.save_results(results)

if __name__ == "__main__":
    main()

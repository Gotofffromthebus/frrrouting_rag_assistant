#!/usr/bin/env python3
"""Split Markdown documents into chunks for RAG."""

import os
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

def chunk_by_headers(markdown_text: str, metadata: dict, 
                     min_chunk_size: int = 200,
                     max_chunk_size: int = 2000,
                     chunk_overlap: int = 100) -> List[Dict]:
    """Split Markdown into chunks by headers."""
    chunks = []
    lines = markdown_text.split('\n')
    
    current_chunk = []
    current_header = metadata.get('title', 'Introduction')
    current_header_path = [current_header]
    chunk_id = 0
    
    def create_chunk(text: str, header_path: List[str], chunk_idx: int) -> Dict:
        """Create chunk dictionary with metadata."""
        return {
            'text': text.strip(),
            'metadata': {
                **metadata,
                'section': ' > '.join(header_path),
                'section_title': header_path[-1] if header_path else current_header,
                'chunk_index': chunk_idx,
                'chunk_size': len(text)
            }
        }
    
    def split_large_chunk(chunk_lines: List[str], header_path: List[str], 
                         chunk_idx: int) -> List[Dict]:
        """Split large chunk into smaller ones by paragraphs."""
        chunk_text = '\n'.join(chunk_lines)
        if len(chunk_text) <= max_chunk_size:
            return [create_chunk(chunk_text, header_path, chunk_idx)]
        
        paragraphs = chunk_text.split('\n\n')
        sub_chunks = []
        temp_chunk = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            test_chunk = '\n\n'.join(temp_chunk + [para])
            if len(test_chunk) > max_chunk_size and temp_chunk:
                sub_chunk_text = '\n\n'.join(temp_chunk)
                if len(sub_chunk_text) >= min_chunk_size:
                    sub_chunks.append(create_chunk(
                        sub_chunk_text, 
                        header_path, 
                        chunk_idx + len(sub_chunks)
                    ))
                temp_chunk = [para]
            else:
                temp_chunk.append(para)
        
        if temp_chunk:
            sub_chunk_text = '\n\n'.join(temp_chunk)
            if len(sub_chunk_text) >= min_chunk_size:
                sub_chunks.append(create_chunk(
                    sub_chunk_text,
                    header_path,
                    chunk_idx + len(sub_chunks)
                ))
        
        return sub_chunks if sub_chunks else [create_chunk(chunk_text, header_path, chunk_idx)]
    
    for line in lines:
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        
        if header_match:
            if current_chunk:
                chunk_text = '\n'.join(current_chunk).strip()
                if len(chunk_text) >= min_chunk_size:
                    if len(chunk_text) > max_chunk_size:
                        sub_chunks = split_large_chunk(
                            current_chunk,
                            current_header_path.copy(),
                            chunk_id
                        )
                        chunks.extend(sub_chunks)
                        chunk_id += len(sub_chunks)
                    else:
                        chunks.append(create_chunk(
                            chunk_text,
                            current_header_path.copy(),
                            chunk_id
                        ))
                        chunk_id += 1
            
            header_level = len(header_match.group(1))
            header_text = header_match.group(2).strip()
            
            current_header_path = current_header_path[:header_level-1]
            current_header_path.append(header_text)
            current_chunk = [line]
        else:
            current_chunk.append(line)
    
    if current_chunk:
        chunk_text = '\n'.join(current_chunk).strip()
        if len(chunk_text) >= min_chunk_size:
            if len(chunk_text) > max_chunk_size:
                sub_chunks = split_large_chunk(
                    current_chunk,
                    current_header_path.copy(),
                    chunk_id
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(create_chunk(
                    chunk_text,
                    current_header_path.copy(),
                    chunk_id
                ))
    
    if chunk_overlap > 0 and len(chunks) > 1:
        for i in range(len(chunks) - 1):
            current_text = chunks[i]['text']
            next_text = chunks[i + 1]['text']
            overlap_text = current_text[-chunk_overlap:] if len(current_text) > chunk_overlap else current_text
            chunks[i + 1]['text'] = overlap_text + '\n\n' + next_text
    
    return chunks

def process_all_docs(docs_dir: str, min_chunk_size: int = 200, 
                    max_chunk_size: int = 2000, chunk_overlap: int = 100) -> List[Dict]:
    """Process all Markdown files and split into chunks."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"Ошибка: директория {docs_dir} не найдена")
        return []
    
    all_chunks = []
    md_files = list(docs_path.rglob("*.md"))
    
    print(f"Found {len(md_files)} Markdown files")
    
    for md_file in tqdm(md_files, desc="Processing"):
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                markdown_text = f.read()
            
            if not markdown_text.strip():
                continue
            
            metadata_file = md_file.with_suffix('.metadata.json')
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            else:
                metadata = {
                    'url': str(md_file.relative_to(docs_path)),
                    'title': md_file.stem,
                    'source': 'fr-docs-scraper',
                    'file_path': str(md_file.relative_to(docs_path))
                }
            
            metadata['file_path'] = str(md_file.relative_to(docs_path))
            
            chunks = chunk_by_headers(
                markdown_text, 
                metadata,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            all_chunks.extend(chunks)
            
        except Exception as e:
            print(f"Error processing {md_file}: {e}")
            continue
    
    return all_chunks

def main():
    parser = argparse.ArgumentParser(description="Split Markdown documents into chunks")
    parser.add_argument("--input", "-i", default="fr_docs", help="Input directory with Markdown files")
    parser.add_argument("--output", "-o", default="chunks.json", help="Output JSON file")
    parser.add_argument("--min-size", type=int, default=200, help="Min chunk size in characters")
    parser.add_argument("--max-size", type=int, default=2000, help="Max chunk size in characters")
    parser.add_argument("--overlap", type=int, default=100, help="Chunk overlap in characters")
    args = parser.parse_args()
    
    print(f"Processing documents from: {args.input}")
    print(f"Chunking params: min={args.min_size}, max={args.max_size}, overlap={args.overlap}")
    
    chunks = process_all_docs(
        args.input,
        min_chunk_size=args.min_size,
        max_chunk_size=args.max_size,
        chunk_overlap=args.overlap
    )
    
    print(f"\nCreated {len(chunks)} chunks")
    
    if chunks:
        sizes = [len(chunk['text']) for chunk in chunks]
        print(f"Chunk sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)}")
    
    print(f"Saving to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    main()


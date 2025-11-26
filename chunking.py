#!/usr/bin/env python3
"""
Скрипт для разбиения Markdown документов на чанки для RAG.

Использование:
    python chunking.py --input fr_docs --output chunks.json --min-size 200 --max-size 2000
"""

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
    """
    Разбивает Markdown на чанки по заголовкам с сохранением контекста.
    
    Args:
        markdown_text: Текст Markdown документа
        metadata: Метаданные документа
        min_chunk_size: Минимальный размер чанка в символах
        max_chunk_size: Максимальный размер чанка в символах
        chunk_overlap: Перекрытие между чанками в символах
    
    Returns:
        Список словарей с текстом и метаданными для каждого чанка
    """
    chunks = []
    lines = markdown_text.split('\n')
    
    current_chunk = []
    current_header = metadata.get('title', 'Introduction')
    current_header_path = [current_header]  # Путь заголовков для контекста
    chunk_id = 0
    
    def create_chunk(text: str, header_path: List[str], chunk_idx: int) -> Dict:
        """Создает словарь чанка с метаданными."""
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
        """Разбивает слишком большой чанк на меньшие по параграфам."""
        chunk_text = '\n'.join(chunk_lines)
        if len(chunk_text) <= max_chunk_size:
            return [create_chunk(chunk_text, header_path, chunk_idx)]
        
        # Разбиваем по двойным переносам строк (параграфы)
        paragraphs = chunk_text.split('\n\n')
        sub_chunks = []
        temp_chunk = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # Проверяем, поместится ли параграф
            test_chunk = '\n\n'.join(temp_chunk + [para])
            if len(test_chunk) > max_chunk_size and temp_chunk:
                # Сохраняем текущий чанк
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
        
        # Последний чанк
        if temp_chunk:
            sub_chunk_text = '\n\n'.join(temp_chunk)
            if len(sub_chunk_text) >= min_chunk_size:
                sub_chunks.append(create_chunk(
                    sub_chunk_text,
                    header_path,
                    chunk_idx + len(sub_chunks)
                ))
        
        return sub_chunks if sub_chunks else [create_chunk(chunk_text, header_path, chunk_idx)]
    
    # Обрабатываем строки
    for line in lines:
        # Проверяем, является ли строка заголовком
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        
        if header_match:
            # Сохраняем текущий чанк, если он достаточно большой
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
            
            # Обновляем путь заголовков
            header_level = len(header_match.group(1))
            header_text = header_match.group(2).strip()
            
            # Обрезаем путь до нужного уровня
            current_header_path = current_header_path[:header_level-1]
            current_header_path.append(header_text)
            
            # Начинаем новый чанк с заголовком
            current_chunk = [line]
        else:
            current_chunk.append(line)
    
    # Последний чанк
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
    
    # Добавляем перекрытие между соседними чанками
    if chunk_overlap > 0 and len(chunks) > 1:
        for i in range(len(chunks) - 1):
            current_text = chunks[i]['text']
            next_text = chunks[i + 1]['text']
            
            # Берем последние N символов текущего чанка
            overlap_text = current_text[-chunk_overlap:] if len(current_text) > chunk_overlap else current_text
            
            # Добавляем в начало следующего чанка
            chunks[i + 1]['text'] = overlap_text + '\n\n' + next_text
    
    return chunks

def process_all_docs(docs_dir: str, min_chunk_size: int = 200, 
                    max_chunk_size: int = 2000, chunk_overlap: int = 100) -> List[Dict]:
    """
    Обрабатывает все Markdown файлы в директории и разбивает их на чанки.
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"Ошибка: директория {docs_dir} не найдена")
        return []
    
    all_chunks = []
    
    # Находим все .md файлы
    md_files = list(docs_path.rglob("*.md"))
    
    print(f"Найдено {len(md_files)} Markdown файлов")
    
    for md_file in tqdm(md_files, desc="Обработка документов"):
        try:
            # Читаем Markdown
            with open(md_file, 'r', encoding='utf-8') as f:
                markdown_text = f.read()
            
            # Пропускаем пустые файлы
            if not markdown_text.strip():
                continue
            
            # Читаем метаданные
            metadata_file = md_file.with_suffix('.metadata.json')
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            else:
                # Создаем базовые метаданные
                metadata = {
                    'url': str(md_file.relative_to(docs_path)),
                    'title': md_file.stem,
                    'source': 'fr-docs-scraper',
                    'file_path': str(md_file.relative_to(docs_path))
                }
            
            # Добавляем путь к файлу в метаданные
            metadata['file_path'] = str(md_file.relative_to(docs_path))
            
            # Разбиваем на чанки
            chunks = chunk_by_headers(
                markdown_text, 
                metadata,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            all_chunks.extend(chunks)
            
        except Exception as e:
            print(f"Ошибка при обработке {md_file}: {e}")
            continue
    
    return all_chunks

def main():
    parser = argparse.ArgumentParser(description="Разбиение Markdown документов на чанки для RAG")
    parser.add_argument("--input", "-i", default="fr_docs", help="Директория с Markdown файлами")
    parser.add_argument("--output", "-o", default="chunks.json", help="Выходной JSON файл с чанками")
    parser.add_argument("--min-size", type=int, default=200, help="Минимальный размер чанка в символах")
    parser.add_argument("--max-size", type=int, default=2000, help="Максимальный размер чанка в символах")
    parser.add_argument("--overlap", type=int, default=100, help="Перекрытие между чанками в символах")
    args = parser.parse_args()
    
    print(f"Обработка документов из: {args.input}")
    print(f"Параметры chunking: min={args.min_size}, max={args.max_size}, overlap={args.overlap}")
    
    # Обрабатываем все документы
    chunks = process_all_docs(
        args.input,
        min_chunk_size=args.min_size,
        max_chunk_size=args.max_size,
        chunk_overlap=args.overlap
    )
    
    print(f"\nСоздано {len(chunks)} чанков")
    
    # Статистика
    if chunks:
        sizes = [len(chunk['text']) for chunk in chunks]
        print(f"Размер чанков: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)}")
    
    # Сохраняем чанки
    print(f"Сохранение в {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    print("Готово!")

if __name__ == "__main__":
    main()


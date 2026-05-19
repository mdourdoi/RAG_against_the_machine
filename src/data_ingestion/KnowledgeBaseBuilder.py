import json
import bm25s
from tqdm import tqdm
from .ChunksCreator import ChunksCreator
import Stemmer
import re


class KnowledgeBaseBuilder:

    def __init__(
            self,
            chunker: ChunksCreator,
            save_dir="data/processed/") -> None:
        self.chunker = chunker
        self.save_dir = save_dir
        self.index_dir = f"{save_dir}/bm25_index"
        self.chunks_file = f"{save_dir}/chunks_metadata.json"
        self.stemmer = Stemmer.Stemmer('english')

    def _save_chunks_metadata(self, chunks: list[dict]) -> bool:
        with open(self.chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=4)

    def build_from_files(self, files_data: list[tuple[str, str]]) -> None:
        chunks = []
        for file_path, content in tqdm(
                files_data, 'Chunking files', unit='files', leave=True):
            temp_chunks = self.chunker.process_file(file_path, content)
            chunks.extend(temp_chunks)

        if len(chunks) == 0:
            print('No data to ingest, aborting...')
            return

        text_content = []
        for chunk in chunks:
            clean_path = chunk['file_path'].replace('_', ' ').replace('/', ' ')
            clean_path = clean_path.replace('.', ' ')
            clean_path = clean_path * 10

            clean_content = chunk['content'].replace('_', ' ')
            clean_content = re.sub(
                r'([a-z])([A-Z])', r'\1 \2', clean_content)

            enriched = f"{clean_path} or {chunk['file_path']}"
            enriched += f"\n\n{clean_content}"
            enriched += f"\n\n{chunk['content']}"

            text_content.append(enriched)
        text_tokens = bm25s.tokenize(
            text_content,
            show_progress=True,
            leave=True,
            stemmer=self.stemmer,
            stopwords="en")
        retriever = bm25s.BM25(k1=2.5, b=0.9)
        retriever.index(text_tokens, show_progress=True, leave_progress=True)
        retriever.save(self.index_dir)
        self._save_chunks_metadata(chunks)

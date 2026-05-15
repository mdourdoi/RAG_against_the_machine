import json
import bm25s
from tqdm import tqdm
from .ChunksCreator import ChunksCreator


class KnowledgeBaseBuilder:

    def __init__(
            self,
            chunker: ChunksCreator,
            save_dir="data/processed/") -> None:
        self.chunker = chunker
        self.save_dir = save_dir
        self.index_dir = f"{save_dir}/bm25_index"
        self.chunks_file = f"{save_dir}/chunks_metadata.json"

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

        text_content = [chunk['content'] for chunk in chunks]
        text_tokens = bm25s.tokenize(
            text_content, show_progress=True, leave=True)
        retriever = bm25s.BM25()
        retriever.index(text_tokens, show_progress=True, leave_progress=True)
        retriever.save(self.index_dir)
        self._save_chunks_metadata(chunks)

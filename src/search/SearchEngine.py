import json
import bm25s
import Stemmer


class SearchEngine:

    def __init__(
            self,
            save_dir: str = "data/processed/") -> None:
        self.save_dir = save_dir
        self.index_dir = f"{save_dir}/bm25_index"
        self.chunks_file = f"{save_dir}/chunks_metadata.json"
        self.load_assets()
        self.stemmer = Stemmer.Stemmer('english')

    def load_assets(self) -> None:
        try:
            self.retriever = bm25s.BM25(k1=2.5, b=0.9).load(
                self.index_dir, load_corpus=False)
        except Exception as e:
            raise Exception(f'Impossible to open the index: {e}')
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            self.chunks_metadata = json.load(f)

    def search_question(self, user_input: str |
                        list[str], k: int = 5) -> list[list[dict]]:
        if isinstance(user_input, str):
            input = [user_input]
        else:
            input = user_input
        query_tokens = bm25s.tokenize(
            input,
            stemmer=self.stemmer,
            stopwords="en"
        )
        results, scores = self.retriever.retrieve(query_tokens, k=k)
        all_answers = []
        for i, line in enumerate(results):
            answer = []
            for j, doc_id in enumerate(line):
                if scores[i][j] <= 0:
                    continue
                chunk = self.chunks_metadata[doc_id]
                source = {
                    "file_path": chunk["file_path"],
                    "first_character_index": chunk["first_character_index"],
                    "last_character_index": chunk["last_character_index"],
                    "content": chunk["content"]}
                answer.append(source)
            all_answers.append(answer)

        return all_answers

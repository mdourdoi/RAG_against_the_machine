from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


class ChunksCreator:
    def __init__(self, max_chunk_size: int):
        """Initialize the chunk splitter configuration.

        Args:
            max_chunk_size: Target maximum chunk size for splitting.
        """
        if max_chunk_size < 800:
            print()
            print('/!\\ WARNING /!\\')
            print('The min chunk size is 800, continuing with this value')
            print()
            self.max_chunk_size = 800
        else:
            self.max_chunk_size = max_chunk_size
        self.chunk_overlap = (self.max_chunk_size * 2) // 10

    def _get_splitter(self, file_path: str) -> RecursiveCharacterTextSplitter:
        """Return a splitter configured for the file type.

        Args:
            file_path: Path to the file being chunked.

        Returns:
            A LangChain splitter adapted to the file extension.
        """
        if file_path.endswith('.py'):
            return RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=self.max_chunk_size,
                add_start_index=True,
                chunk_overlap=self.chunk_overlap,
                strip_whitespace=False)
        if file_path.endswith('.md'):
            return RecursiveCharacterTextSplitter.from_language(
                language=Language.MARKDOWN,
                chunk_size=self.max_chunk_size,
                add_start_index=True,
                chunk_overlap=self.chunk_overlap,
                strip_whitespace=False)
        else:
            return RecursiveCharacterTextSplitter(
                chunk_size=self.max_chunk_size,
                add_start_index=True,
                chunk_overlap=self.chunk_overlap,
                strip_whitespace=False)

    def process_file(self, file_path: str, content: str) -> list[dict]:
        """Split a file into chunks with positional metadata.

        Args:
            file_path: Path to the file being chunked.
            content: File content to split.

        Returns:
            A list of chunk dictionaries containing path, offsets and text.
        """
        splitter = self._get_splitter(file_path)
        documents = splitter.create_documents([content])
        res = []
        for doc in documents:
            chunk_text = doc.page_content
            first_index = doc.metadata['start_index']
            last_index = first_index + len(chunk_text)
            chunk = {
                'file_path': file_path,
                'first_character_index': first_index,
                'last_character_index': last_index,
                'content': chunk_text}
            res.append(chunk)
        return res

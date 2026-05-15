from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


class ChunksCreator:
    def __init__(self, max_chunk_size: int):
        """
        Initialization for the data retriever. Set the max chunk size to
        2000 if the input is higher.
        """
        if max_chunk_size > 2000:
            print('The maximum chunk size is 2000, continuing with this value')
            self.max_chunk_size = 2000
        else:
            self.max_chunk_size = max_chunk_size

    def _get_splitter(self, file_path: str) -> RecursiveCharacterTextSplitter:
        """
        Creates the correct splitter for the instance.

        Args:
            - file_path => the path to the file which will be ingested
        Returns:
            - a splitter from the langchain library adapted to the type of file
        """
        if file_path.endswith('.py'):
            return RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=self.max_chunk_size,
                add_start_index=True)
        if file_path.endswith('.md'):
            return RecursiveCharacterTextSplitter.from_language(
                language=Language.MARKDOWN,
                chunk_size=self.max_chunk_size,
                add_start_index=True)
        else:
            return RecursiveCharacterTextSplitter(
                chunk_size=self.max_chunk_size,
                add_start_index=True)

    def process_file(self, file_path: str, content: str) -> list[dict]:
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

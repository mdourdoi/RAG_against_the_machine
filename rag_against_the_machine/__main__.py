from data_ingestion import ChunksCreator, FilesGetter, KnowledgeBaseBuilder


def main():
    # 1. L'utilisateur te donne un chemin (dossier ou archive)
    source = "vllm-0.10.1"

    # 2. Tu extrais les données en mémoire
    file_getter = FilesGetter()
    extracted_data = file_getter.extract_files_data(source)
    if not extracted_data:
        return

    # 3. Tu lances la machine que l'on a construite précédemment
    chunker = ChunksCreator(max_chunk_size=2000)
    builder = KnowledgeBaseBuilder(chunker=chunker)

    builder.build_from_files(extracted_data)


if __name__ == "__main__":
    main()

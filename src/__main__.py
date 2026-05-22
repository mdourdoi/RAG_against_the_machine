import json
from .data_ingestion import ChunksCreator, FilesGetter, KnowledgeBaseBuilder
from .search import SearchEngine, Searcher
from .llm import Generator
import fire


def check_5_percent_overlap(
        retrieved_source: dict,
        expected_source: dict) -> bool:
    chemin_trouve = retrieved_source.file_path.replace('\\', '/')
    chemin_attendu = expected_source["file_path"].replace('\\', '/')

    if chemin_trouve not in chemin_attendu and chemin_attendu not in chemin_trouve:
        return False

    start_inter = max(retrieved_source.first_character_index,
                      expected_source["first_character_index"])
    end_inter = min(retrieved_source.last_character_index,
                    expected_source["last_character_index"])

    overlap_len = max(0, end_inter - start_inter)

    # 3. Calcul du pourcentage par rapport à la taille de la source attendue
    expected_len = expected_source["last_character_index"] - \
        expected_source["first_character_index"]
    if expected_len == 0:
        return False

    overlap_ratio = overlap_len / expected_len
    return overlap_ratio >= 0.05


def calculate_recall_at_k(predictions, ground_truth_path: str) -> float:
    """
    Calcule le Recall global en comparant les prédictions avec le fichier de réponses.
    """
    # Chargement du fichier contenant les vraies réponses
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    # Création d'un dictionnaire {question_id: [liste_des_sources_attendues]}
    # pour une recherche rapide
    gt_dict = {q["question_id"]: q.get("sources", [])
               for q in gt_data["rag_questions"]}

    total_score = 0
    num_questions = len(predictions.search_results)

    for result in predictions.search_results:
        expected_sources = gt_dict.get(result.question_id, [])
        if not expected_sources:
            continue  # Sécurité si la question n'a pas de sources attendues

        found_count = 0

        # Pour chaque source que l'algorithme était censé trouver
        for expected in expected_sources:
            is_found = any(check_5_percent_overlap(ret_source, expected)
                           for ret_source in result.retrieved_sources)
            if is_found:
                found_count += 1

        # Calcul du score pour cette question spécifique
        question_score = found_count / len(expected_sources)
        total_score += question_score

    # Moyenne globale
    return (total_score / num_questions) if num_questions > 0 else 0.0


def main(
        source: str = "data/raw/vllm-0.10.1",
        dataset_path: str = "private/AnsweredQuestions/dataset_code_private.json",
        max_chunk_size: int = 1400,
        k: int = 5):
    # --- 1. INGESTION ---
    file_getter = FilesGetter()
    extracted_data = file_getter.extract_files_data(source)
    if not extracted_data:
        print("Extract error.")
        return

    chunker = ChunksCreator(max_chunk_size)
    builder = KnowledgeBaseBuilder(chunker=chunker)
    builder.build_from_files(extracted_data)

    # --- 2. RECHERCHE ---
    print("\nChargement du moteur de recherche...")
    engine = SearchEngine(save_dir="data/processed/")
    searcher = Searcher(engine)

    # chemin vers le fichier de test
    ground_truth_path = dataset_path

    print(f"Exécution de la recherche sur le dataset (k={k})...")
    # Note : On peut très bien lancer la recherche sur le fichier 'Answered'
    results = searcher.process_dataset(ground_truth_path, k)

    with open('test.json', 'w') as f:
        f.write(results.model_dump_json(indent=4))

    # --- 3. GÉNÉRATION ---
    print("\nLauching answering phase LLM...")
    generator = Generator()
    
    # LA ligne magique
    final_results = generator.process_dataset(results)

    # --- 4. SAUVEGARDE ---
    with open('output_final.json', 'w', encoding='utf-8') as f:
        f.write(final_results.model_dump_json(indent=4))


if __name__ == "__main__":
    fire.Fire(main)

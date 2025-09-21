"""
Script per la preparazione dei dataset per l'addestramento del modello BiGRU.

Converte il dataset raw da formato wide a long, gestisce deduplicazione
e crea train/test split per l'addestramento.

File richiesti: datasets/raw_datasets.csv
File generati: datasets/train.csv, datasets/test.csv

Esempio di utilizzo:
    python prepare_datasets.py
"""

import os
import csv
import numpy as np
from src.text_processing import normalize_text_for_model, signature_for_dedupe


def prepare_train_and_validation_datasets(
    neutral_label_name: str = "Neutral",
):
    """
    Converte il CSV di tone-adjustment in formato wide (colonne: Original, Polite, Professional, Casual)
    in un CSV in formato long con colonne: text,label,label_name.

    Regole:
    - Mappatura etichette: Original->0, Polite->1, Professional->2, Casual->3
    - Usa label_name dal nome colonna, tranne 'Original' che diventa 'Neutral'
    - Produce 4 righe di output per ogni riga di input (una per stile)
    - Le celle sotto 'Original' possono contenere una etichetta sorgente finale separata da punto e virgola (es. "...;sadness").
      Solo il testo prima del primo punto e virgola viene utilizzato.

    Argomenti:
        neutral_label_name: label_name sostitutiva per la colonna 'Original'.
    """

    # Colonne attese (fisse e nell'ordine esatto)
    expected_order = ["Original", "Polite", "Professional", "Casual"]

    # Mappatura id etichetta (secondo specifica)
    label_id_for_column = {
        "original": 0,
        "polite": 1,
        "professional": 2,
        "casual": 3,
    }

    # Mappatura label_name per visualizzazione
    label_name_for_column = {
        "original": neutral_label_name,
        "polite": "Polite",
        "professional": "Professional",
        "casual": "Casual",
    }

    # Percorsi fissi: input datasets/raw_datasets.csv, output datasets/datasets.csv
    project_root = os.path.dirname(os.path.abspath(__file__))
    input_csv_path = os.path.join(project_root, "datasets", "raw_datasets.csv")
    output_csv_path = os.path.join(project_root, "datasets", "datasets.csv")

    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input CSV non trovato: {input_csv_path}")

    # Legge l'input e scrive l'output in formato long
    # Assicura che la cartella di output esista
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Legge tutte le righe con fallback di encoding
    def _read_csv_rows_with_fallback(path: str):
        encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
        last_err = None
        for enc in encodings_to_try:
            try:
                with open(path, "r", encoding=enc, newline="") as f:
                    return list(csv.reader(f))
            except UnicodeDecodeError as e:
                last_err = e
                continue
        raise ValueError(
            f"Impossibile decodificare il file CSV con encoding standard (ultimo errore: {last_err})"
        )

    rows = _read_csv_rows_with_fallback(input_csv_path)

    if not rows:
        return

    # Verifica header esatto e costruisce matrice (n_righe, 4)
    raw_header = rows[0]
    header = [h.strip() for h in raw_header[:4]]
    if header != expected_order:
        raise ValueError(
            f"Header non valido. Atteso {expected_order}, trovato {raw_header}"
        )

    selected_rows = []
    for row in rows[1:]:
        # Prende le prime 4 colonne nell'ordine fisso
        sel = [row[0] if 0 < len(row) else "",
               row[1] if 1 < len(row) else "",
               row[2] if 2 < len(row) else "",
               row[3] if 3 < len(row) else ""]
        selected_rows.append(sel)

    if len(selected_rows) == 0:
        return

    texts_matrix = np.array(selected_rows, dtype=str)  # shape: (N, 4)

    # Per 'Original', rimuove eventuale parte dopo il primo ';'
    # e rimuove spazi superflui per tutte le colonne
    original_col = texts_matrix[:, 0]
    # partition restituisce (prima, sep, dopo) per ogni elemento
    parted = np.char.partition(original_col, ";")
    texts_matrix[:, 0] = parted[:, 0]
    texts_matrix = np.char.strip(texts_matrix)
    # Normalize all cells for consistent training
    vector_norm = np.vectorize(normalize_text_for_model)
    texts_matrix = vector_norm(texts_matrix)

    # Dedupe at the original-row level using a signature of the 'original' column
    originals = texts_matrix[:, 0]
    sigs = [signature_for_dedupe(t) for t in originals]
    seen = set()
    keep_indices = []
    for i, sig in enumerate(sigs):
        if sig in seen:
            continue
        seen.add(sig)
        keep_indices.append(i)
    if keep_indices:
        texts_matrix = texts_matrix[np.array(keep_indices, dtype=int), :]

    # Prepara label ids e nomi, ripetuti per tutte le righe
    base_label_ids = np.array([
        label_id_for_column["original"],
        label_id_for_column["polite"],
        label_id_for_column["professional"],
        label_id_for_column["casual"],
    ], dtype=int)
    base_label_names = np.array([
        label_name_for_column["original"],
        label_name_for_column["polite"],
        label_name_for_column["professional"],
        label_name_for_column["casual"],
    ], dtype=str)

    N = texts_matrix.shape[0]

    # Split 90/10 a livello di riga originale (prima dell'espansione a 4 stili)
    rng = np.random.RandomState(42)
    row_indices = np.arange(N)
    rng.shuffle(row_indices)
    split_at_rows = int(N * 0.9)
    train_rows = row_indices[:split_at_rows]
    test_rows = row_indices[split_at_rows:]

    # Espansione per il train
    train_matrix = texts_matrix[train_rows, :]  # shape: (Nt, 4)
    Nt = train_matrix.shape[0]
    texts_train = train_matrix.reshape(-1, order="C")
    labels_train = np.tile(base_label_ids, Nt)
    names_train = np.tile(base_label_names, Nt)

    # Espansione per il test
    test_matrix = texts_matrix[test_rows, :]  # shape: (Nte, 4)
    Nte = test_matrix.shape[0]
    texts_test = test_matrix.reshape(-1, order="C")
    labels_test = np.tile(base_label_ids, Nte)
    names_test = np.tile(base_label_names, Nte)

    output_dir = os.path.join(project_root, "datasets")
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    # Scrive train.csv
    with open(train_path, "w", encoding="utf-8", newline="") as ftr:
        wtr = csv.writer(ftr)
        wtr.writerow(["text", "label", "label_name"])
        wtr.writerows(
            zip(texts_train.tolist(), labels_train.astype(str).tolist(), names_train.tolist())
        )

    # Scrive test.csv
    with open(test_path, "w", encoding="utf-8", newline="") as fte:
        wte = csv.writer(fte)
        wte.writerow(["text", "label", "label_name"])
        wte.writerows(
            zip(texts_test.tolist(), labels_test.astype(str).tolist(), names_test.tolist())
        )
 
def main():
    prepare_train_and_validation_datasets()
    project_root = os.path.dirname(os.path.abspath(__file__))
    print(f"Dataset preparati in: {os.path.join(project_root, 'datasets')} (train.csv, test.csv, raw_datasets.csv)")
 
 
if __name__ == "__main__":
    main()

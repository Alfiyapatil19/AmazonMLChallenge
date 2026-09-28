import argparse
import os
import pickle
import sqlite3
import tempfile
import time
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from src.features import FEATURE_NAMES, extract_pair_features
from src.normalization import normalize_country, normalize_text, standardize_address_text
from src.tokenization import extract_address_tokens, extract_name_ngrams, extract_name_tokens


def build_disk_index(database_path: str, target_paths: List[str]) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-65536")
    connection.executescript(
        "CREATE TABLE IF NOT EXISTS records (entity_id TEXT PRIMARY KEY, name TEXT, address TEXT, country TEXT);"
        "CREATE TABLE IF NOT EXISTS postings (token TEXT NOT NULL, entity_id TEXT NOT NULL);"
    )

    record_batch: List[Tuple[str, str, str, str]] = []
    posting_batch: List[Tuple[str, str]] = []
    total_records = connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    records_to_skip = total_records

    def flush() -> None:
        nonlocal total_records
        if not record_batch:
            return
        connection.executemany(
            "INSERT INTO records VALUES (?, ?, ?, ?)", record_batch
        )
        connection.executemany(
            "INSERT INTO postings VALUES (?, ?)", posting_batch
        )
        total_records += len(record_batch)
        record_batch.clear()
        posting_batch.clear()
        connection.commit()
        if total_records % 250000 < 5000:
            print(f"Indexed {total_records:,} test target records", flush=True)

    for path in target_paths:
        chunks = pd.read_csv(
            path,
            sep="\t",
            usecols=["entity_id", "business_name", "business_address", "country"],
            dtype=str,
            keep_default_na=False,
            chunksize=5000,
            encoding="utf-8",
        )
        for chunk in chunks:
            if records_to_skip >= len(chunk):
                records_to_skip -= len(chunk)
                continue
            if records_to_skip:
                chunk = chunk.iloc[records_to_skip:]
                records_to_skip = 0
            for entity_id, business_name, business_address, raw_country in zip(
                chunk["entity_id"],
                chunk["business_name"],
                chunk["business_address"],
                chunk["country"],
            ):
                name = normalize_text(business_name)
                address = standardize_address_text(business_address)
                country = normalize_country(raw_country)
                record_batch.append((entity_id, name, address, country))

                tokens: Set[str] = set(extract_name_tokens(name))
                tokens.update(extract_address_tokens(address))
                tokens.update(extract_name_ngrams(name, n=4))
                posting_batch.extend((token, entity_id) for token in tokens)
                if len(record_batch) >= 5000:
                    flush()
    if records_to_skip:
        connection.close()
        raise ValueError(
            f"Existing disk index has {total_records:,} records, more than the "
            "provided target inputs."
        )
    flush()

    print("Building disk-backed token index...", flush=True)
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS postings_token_entity "
        "ON postings(token, entity_id)"
    )
    if connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='token_counts'"
    ).fetchone() is None:
        connection.execute(
            "CREATE TABLE token_counts AS "
            "SELECT token, COUNT(*) AS post_count FROM postings GROUP BY token"
        )
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS token_counts_token "
        "ON token_counts(token)"
    )
    connection.executescript(
        "CREATE TEMP TABLE IF NOT EXISTS query_tokens (token TEXT PRIMARY KEY, max_count INTEGER);"
        "CREATE TEMP TABLE IF NOT EXISTS rank_tokens (token TEXT PRIMARY KEY, weight INTEGER);"
        "CREATE TEMP TABLE IF NOT EXISTS candidate_ids (entity_id TEXT PRIMARY KEY) WITHOUT ROWID;"
    )
    connection.commit()
    return connection


def generate_ranked_candidates(
    connection: sqlite3.Connection,
    name: str,
    address: str,
    name_tokens: Set[str],
    address_tokens: Set[str],
) -> List[str]:
    all_name_tokens = set(extract_name_tokens(name))
    all_address_tokens = set(extract_address_tokens(address))
    name_ngrams = set(extract_name_ngrams(name, n=4))

    caps = {token: 2000 for token in all_name_tokens}
    caps.update(
        {
            token: 4000 if token.startswith(("sh_", "num_")) else 1500
            for token in all_address_tokens
        }
    )
    caps.update({token: 200 for token in name_ngrams})

    connection.execute("DELETE FROM query_tokens")
    connection.executemany(
        "INSERT INTO query_tokens VALUES (?, ?)", caps.items()
    )
    connection.execute("DELETE FROM candidate_ids")
    connection.execute(
        "INSERT OR IGNORE INTO candidate_ids "
        "SELECT p.entity_id FROM postings p "
        "JOIN query_tokens q ON q.token = p.token "
        "JOIN token_counts c ON c.token = p.token "
        "WHERE c.post_count <= q.max_count"
    )

    if connection.execute("SELECT 1 FROM candidate_ids LIMIT 1").fetchone() is None:
        rarest = connection.execute(
            "SELECT q.token FROM query_tokens q "
            "JOIN token_counts c ON c.token = q.token "
            "ORDER BY c.post_count, q.token LIMIT 1"
        ).fetchone()
        if rarest is not None:
            connection.execute(
                "INSERT OR IGNORE INTO candidate_ids "
                "SELECT entity_id FROM postings WHERE token = ?",
                rarest,
            )

    candidates = [
        row[0] for row in connection.execute("SELECT entity_id FROM candidate_ids")
    ]
    if not candidates:
        return []

    weights = {token: 2 for token in name_tokens}
    weights.update({token: 1 for token in address_tokens})
    connection.execute("DELETE FROM rank_tokens")
    connection.executemany(
        "INSERT INTO rank_tokens VALUES (?, ?)", weights.items()
    )
    overlap = dict(
        connection.execute(
            "SELECT p.entity_id, SUM(r.weight) "
            "FROM postings p "
            "JOIN rank_tokens r ON r.token = p.token "
            "JOIN candidate_ids c ON c.entity_id = p.entity_id "
            "GROUP BY p.entity_id"
        )
    )
    ranked = [(overlap.get(entity_id, 0), entity_id) for entity_id in candidates]
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [entity_id for _, entity_id in ranked[:60]]


def apply_decision_rule(
    candidate_ids: List[str], probabilities: np.ndarray
) -> List[str]:
    if not candidate_ids:
        return []
    max_probability = float(np.max(probabilities))
    if max_probability < 0.985:
        return []
    accepted = [
        (entity_id, float(probability))
        for entity_id, probability in zip(candidate_ids, probabilities)
        if probability >= 0.985 and probability >= max_probability - 0.005
    ]
    accepted.sort(key=lambda item: item[1], reverse=True)
    return sorted(entity_id for entity_id, _ in accepted[:7])


def verify_matching_output(
    s1_test_path: str,
    matching_path: str,
    output_id_column: str = "source1_entity_id",
    chunksize: int = 100000,
) -> Tuple[int, int, int, int, int]:
    expected_rows = 0
    output_rows = 0

    with tempfile.TemporaryDirectory(prefix="entity_resolution_verify_") as temp_dir:
        connection = sqlite3.connect(os.path.join(temp_dir, "coverage.sqlite3"))
        connection.executescript(
            "CREATE TABLE expected (entity_id TEXT PRIMARY KEY);"
            "CREATE TABLE output (entity_id TEXT NOT NULL);"
        )

        for chunk in pd.read_csv(
            s1_test_path,
            sep="\t",
            usecols=["entity_id"],
            dtype=str,
            keep_default_na=False,
            chunksize=chunksize,
            encoding="utf-8",
        ):
            ids = [(entity_id,) for entity_id in chunk["entity_id"]]
            expected_rows += len(ids)
            connection.executemany(
                "INSERT OR IGNORE INTO expected VALUES (?)", ids
            )

        for chunk in pd.read_csv(
            matching_path,
            sep="\t",
            usecols=[output_id_column],
            dtype=str,
            keep_default_na=False,
            chunksize=chunksize,
            encoding="utf-8",
        ):
            ids = [(entity_id,) for entity_id in chunk[output_id_column]]
            output_rows += len(ids)
            connection.executemany("INSERT INTO output VALUES (?)", ids)

        missing = connection.execute(
            "SELECT COUNT(*) FROM expected e "
            "LEFT JOIN output o ON o.entity_id = e.entity_id "
            "WHERE o.entity_id IS NULL"
        ).fetchone()[0]
        duplicates = connection.execute(
            "SELECT COALESCE(SUM(row_count - 1), 0) FROM ("
            "SELECT COUNT(*) AS row_count FROM output "
            "GROUP BY entity_id HAVING COUNT(*) > 1)"
        ).fetchone()[0]
        unexpected = connection.execute(
            "SELECT COUNT(*) FROM output o "
            "LEFT JOIN expected e ON e.entity_id = o.entity_id "
            "WHERE e.entity_id IS NULL"
        ).fetchone()[0]
        connection.close()

    return (
        expected_rows,
        output_rows,
        int(missing),
        int(duplicates),
        int(unexpected),
    )


def run(args: argparse.Namespace) -> int:
    with open(args.model, "rb") as model_file:
        artifact = pickle.load(model_file)
    if artifact.get("feature_names") != FEATURE_NAMES or artifact.get("feature_count") != 33:
        raise ValueError("Saved model metadata does not match the current 33-feature extractor.")
    if artifact["model"].get_booster().num_features() != 33:
        raise ValueError("Saved model does not accept exactly 33 features.")
    expected_config = {
        "hard_negatives_per_entity": 15,
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.04,
        "subsample": 0.88,
        "colsample_bytree": 0.88,
        "min_child_weight": 3,
        "reg_alpha": 0.15,
        "reg_lambda": 1.2,
        "random_state": 42,
        "threshold": 0.985,
        "margin": 0.005,
        "max_k": 7,
    }
    if artifact.get("config") != expected_config:
        raise ValueError("Saved model configuration differs from the requested configuration.")

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidates_path = os.path.join(output_dir, "candidate_pairs.tsv")
    matching_temp = matching_path + ".tmp"
    candidates_temp = candidates_path + ".tmp"
    write_candidates = not args.matching_only
    checkpoint_path = os.path.join(output_dir, ".inference_checkpoint.sqlite3")
    if os.path.exists(matching_path):
        raise FileExistsError(
            f"Refusing to overwrite existing matching results: {matching_path}"
        )

    checkpoint = sqlite3.connect(checkpoint_path)
    checkpoint.execute(
        "CREATE TABLE IF NOT EXISTS progress ("
        "id INTEGER PRIMARY KEY CHECK (id = 1), input_rows INTEGER NOT NULL, "
        "matching_bytes INTEGER NOT NULL, candidates_bytes INTEGER NOT NULL, "
        "candidate_count INTEGER NOT NULL)"
    )
    saved = checkpoint.execute(
        "SELECT input_rows, matching_bytes, candidates_bytes, candidate_count "
        "FROM progress WHERE id = 1"
    ).fetchone()
    if saved is None:
        if os.path.exists(matching_temp) or (
            write_candidates and os.path.exists(candidates_temp)
        ):
            checkpoint.close()
            raise FileExistsError(
                "Temporary inference outputs exist without a checkpoint; "
                "preserving them instead of overwriting."
            )
        with open(matching_temp, "wb") as matching_file:
            matching_file.write(b"source1_entity_id\tmatched_entity_ids\n")
            matching_file.flush()
            os.fsync(matching_file.fileno())
            if write_candidates:
                with open(candidates_temp, "wb") as candidates_file:
                    candidates_file.write(b"source1_entity_id\tcandidate_entity_ids\n")
                    candidates_file.flush()
                    os.fsync(candidates_file.fileno())
            saved = (
                0,
                matching_file.tell(),
                os.path.getsize(candidates_temp) if write_candidates else 0,
                0,
            )
        checkpoint.execute(
            "INSERT INTO progress VALUES (1, ?, ?, ?, ?)", saved
        )
        checkpoint.commit()
    elif not os.path.exists(matching_temp) or (
        write_candidates and not os.path.exists(candidates_temp)
    ):
        checkpoint.close()
        raise FileNotFoundError(
            "Checkpoint exists but one or more temporary output files are missing."
        )
    input_rows_done, matching_bytes, candidates_bytes, total_candidates = saved

    matching_file = open(matching_temp, "r+b")
    matching_file.truncate(matching_bytes)
    matching_file.seek(matching_bytes)
    candidates_file = None
    if write_candidates:
        candidates_file = open(candidates_temp, "r+b")
        candidates_file.truncate(candidates_bytes)
        candidates_file.seek(candidates_bytes)

    with tempfile.TemporaryDirectory(prefix="entity_resolution_final_") as temp_dir:
        database_path = os.path.join(output_dir, ".test_targets.sqlite3")
        connection = build_disk_index(database_path, [args.s2_test, args.s3_test])
        model = artifact["model"]
        processed = 0
        started = time.time()
        rows_seen = 0
        for chunk in pd.read_csv(
            args.s1_test,
            sep="\t",
            usecols=["entity_id", "business_name", "business_address", "country"],
            dtype=str,
            keep_default_na=False,
            chunksize=max(args.s1_batch_size, 1000),
            encoding="utf-8",
        ):
            chunk_start = rows_seen
            rows_seen += len(chunk)
            if rows_seen <= input_rows_done:
                continue
            offset = max(0, input_rows_done - chunk_start)
            for batch_start in range(offset, len(chunk), args.s1_batch_size):
                batch_frame = chunk.iloc[batch_start:batch_start + args.s1_batch_size]
                batch = []
                for entity_id, raw_name, raw_address, raw_country in zip(
                    batch_frame["entity_id"],
                    batch_frame["business_name"],
                    batch_frame["business_address"],
                    batch_frame["country"],
                ):
                    name = normalize_text(raw_name)
                    address = standardize_address_text(raw_address)
                    country = normalize_country(raw_country)
                    name_tokens = set(extract_name_tokens(name))
                    address_tokens = set(extract_address_tokens(address))
                    candidate_ids = generate_ranked_candidates(
                        connection, name, address, name_tokens, address_tokens
                    )
                    total_candidates += len(candidate_ids)
                    batch.append((entity_id, name, address, country, candidate_ids))

                flat_candidate_ids = [
                    candidate_id
                    for item in batch
                    for candidate_id in item[4]
                ]
                records: Dict[str, Tuple[str, str, str]] = {}
                for start in range(0, len(flat_candidate_ids), 500):
                    id_chunk = flat_candidate_ids[start:start + 500]
                    placeholders = ",".join("?" for _ in id_chunk)
                    records.update(
                        {
                            entity_id: (name, address, country)
                            for entity_id, name, address, country in connection.execute(
                                "SELECT entity_id, name, address, country FROM records "
                                f"WHERE entity_id IN ({placeholders})",
                                id_chunk,
                            )
                        }
                    )

                feature_rows = []
                offsets = []
                for entity_id, name, address, country, candidate_ids in batch:
                    start = len(feature_rows)
                    for candidate_id in candidate_ids:
                        if candidate_id not in records:
                            raise RuntimeError(f"Candidate record missing from test targets: {candidate_id}")
                        candidate_name, candidate_address, candidate_country = records[candidate_id]
                        feature_rows.append(
                            extract_pair_features(
                                name,
                                address,
                                country,
                                candidate_name,
                                candidate_address,
                                candidate_country,
                                cand_id=candidate_id,
                            )
                        )
                    offsets.append((entity_id, candidate_ids, start, len(feature_rows)))

                probabilities = (
                    model.predict_proba(np.asarray(feature_rows, dtype=np.float32))[:, 1]
                    if feature_rows
                    else np.asarray([], dtype=np.float32)
                )
                for entity_id, candidate_ids, start, end in offsets:
                    matches = apply_decision_rule(candidate_ids, probabilities[start:end])
                    if candidates_file is not None:
                        candidates_file.write(
                            f"{entity_id}\t{','.join(candidate_ids)}\n".encode("utf-8")
                        )
                    matching_file.write(
                        f"{entity_id}\t{','.join(matches)}\n".encode("utf-8")
                    )
                processed += len(batch)
                matching_file.flush()
                os.fsync(matching_file.fileno())
                if candidates_file is not None:
                    candidates_file.flush()
                    os.fsync(candidates_file.fileno())
                input_rows_done += len(batch)
                checkpoint.execute(
                    "UPDATE progress SET input_rows = ?, matching_bytes = ?, "
                    "candidates_bytes = ?, candidate_count = ? WHERE id = 1",
                    (
                        input_rows_done,
                        matching_file.tell(),
                        candidates_file.tell() if candidates_file is not None else candidates_bytes,
                        total_candidates,
                    ),
                )
                checkpoint.commit()
                if processed % 10000 < args.s1_batch_size:
                    print(
                        f"Scored {processed:,} Source 1 rows; "
                        f"{total_candidates:,} candidates; "
                        f"{time.time() - started:.1f}s elapsed",
                        flush=True,
                    )

        connection.close()

    matching_file.close()
    if candidates_file is not None:
        candidates_file.close()
    checkpoint.close()

    expected_rows, output_rows, missing, duplicates, unexpected = verify_matching_output(
        args.s1_test, matching_temp
    )
    if write_candidates:
        (
            candidate_expected,
            candidate_rows,
            candidate_missing,
            candidate_duplicates,
            candidate_unexpected,
        ) = verify_matching_output(
            args.s1_test,
            candidates_temp,
            output_id_column="source1_entity_id",
        )
    else:
        candidate_expected = candidate_rows = expected_rows
        candidate_missing = candidate_duplicates = candidate_unexpected = 0
    if (
        expected_rows != output_rows
        or missing != 0
        or duplicates != 0
        or unexpected != 0
        or input_rows_done != expected_rows
        or candidate_expected != expected_rows
        or candidate_rows != expected_rows
        or candidate_missing != 0
        or candidate_duplicates != 0
        or candidate_unexpected != 0
    ):
        raise RuntimeError(
            "Matching output validation failed: "
            f"expected={expected_rows}, output={output_rows}, missing={missing}, "
            f"duplicates={duplicates}, unexpected={unexpected}, "
            f"processed={input_rows_done}; candidate_rows={candidate_rows}, "
            f"candidate_missing={candidate_missing}, "
            f"candidate_duplicates={candidate_duplicates}, "
            f"candidate_unexpected={candidate_unexpected}"
        )

    if write_candidates:
        os.replace(candidates_temp, candidates_path)
    elif os.path.exists(candidates_temp):
        os.remove(candidates_temp)
    os.replace(matching_temp, matching_path)
    os.remove(checkpoint_path)
    print(f"Expected Source 1 rows: {expected_rows:,}")
    print(f"Output rows: {output_rows:,}; missing: {missing}; duplicates: {duplicates}")
    print(f"Candidate pairs scored: {total_candidates:,}")
    print(f"Matching results: {matching_path}")
    if write_candidates:
        print(f"Candidate pairs: {candidates_path}")
    return processed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-test", default="test/test_source1.tsv")
    parser.add_argument("--s2-test", default="test/test_source2.tsv")
    parser.add_argument("--s3-test", default="test/test_source3.tsv")
    parser.add_argument("--model", default="models/er_matcher_final_tau0985_delta0005_k7.pkl")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--s1-batch-size", type=int, default=64)
    parser.add_argument("--matching-only", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
"""
Main Blocker pipeline for Business Entity Resolution Blocking.

Orchestrates:
1. Dynamic country partitioning
2. Name Token Blocking (with acronym handling)
3. Address Token Blocking (with compound shingles)
4. Character N-Gram Blocking
5. Candidate UNION and deduplication
6. Memory-efficient chunked streaming to output TSV
"""

import os
import gc
import time
from typing import Dict, Set, List, Optional
import pandas as pd
from .normalization import normalize_country
from .indexer import CountryInvertedIndex


class EntityResolutionBlocker:
    """
    Scalable, memory-efficient candidate generator for Entity Resolution.
    """

    def __init__(
        self,
        name_token_cap: int = 2000,
        addr_word_cap: int = 1500,
        shingle_cap: int = 4000,
        ngram_cap: int = 200,
        use_ngrams: bool = True,
        ngram_n: int = 4,
        chunk_size: int = 200000
    ):
        self.name_token_cap = name_token_cap
        self.addr_word_cap = addr_word_cap
        self.shingle_cap = shingle_cap
        self.ngram_cap = ngram_cap
        self.use_ngrams = use_ngrams
        self.ngram_n = ngram_n
        self.chunk_size = chunk_size

    def discover_countries(self, *tsv_paths: str) -> Set[str]:
        """Dynamically scans files to discover all unique country values."""
        countries: Set[str] = set()
        for path in tsv_paths:
            if not os.path.exists(path):
                continue
            for chunk in pd.read_csv(path, sep="\t", usecols=["country"], chunksize=self.chunk_size):
                for val in chunk["country"]:
                    countries.add(normalize_country(val))
        return countries

    def generate_candidates_for_country(
        self,
        country: str,
        s1_path: str,
        s2_path: str,
        s3_path: str,
        candidate_dict: Optional[Dict[str, Set[str]]] = None,
        out_file_handle=None
    ) -> int:
        """
        Builds inverted index for a single country partition from S2 and S3,
        then queries all S1 records belonging to that country.
        """
        print(f"[{country}] Building inverted index from S2 and S3...")
        t0 = time.time()
        
        index = CountryInvertedIndex(
            country=country,
            name_token_cap=self.name_token_cap,
            addr_word_cap=self.addr_word_cap,
            shingle_cap=self.shingle_cap,
            ngram_cap=self.ngram_cap,
            use_ngrams=self.use_ngrams,
            ngram_n=self.ngram_n
        )
        
        # 1. Index S2 records for this country
        for chunk in pd.read_csv(s2_path, sep="\t", chunksize=self.chunk_size):
            for eid, name, addr, cntry in zip(
                chunk["entity_id"], chunk["business_name"], chunk["business_address"], chunk["country"]
            ):
                c_norm = normalize_country(cntry)
                if country == "UNKNOWN" or c_norm == country:
                    index.add_record(eid, name, addr)
                    
        # 2. Index S3 records for this country
        for chunk in pd.read_csv(s3_path, sep="\t", chunksize=self.chunk_size):
            for eid, name, addr, cntry in zip(
                chunk["entity_id"], chunk["business_name"], chunk["business_address"], chunk["country"]
            ):
                c_norm = normalize_country(cntry)
                if country == "UNKNOWN" or c_norm == country:
                    index.add_record(eid, name, addr)
                    
        t_index = time.time() - t0
        print(f"[{country}] Indexed {index.total_indexed:,} S2/S3 records in {t_index:.2f}s")
        
        # 3. Query S1 records for this country
        print(f"[{country}] Generating candidate sets for S1 records...")
        t_query_start = time.time()
        s1_processed = 0
        
        for chunk in pd.read_csv(s1_path, sep="\t", chunksize=self.chunk_size):
            for eid, name, addr, cntry in zip(
                chunk["entity_id"], chunk["business_name"], chunk["business_address"], chunk["country"]
            ):
                c_norm = normalize_country(cntry)
                if country == "UNKNOWN" or c_norm == country:
                    s1_processed += 1
                    cands = index.query_candidates(name, addr)
                    
                    # Ensure S1 is not included and candidates are unique
                    cands.discard(eid)
                    
                    if candidate_dict is not None:
                        if eid in candidate_dict:
                            candidate_dict[eid].update(cands)
                        else:
                            candidate_dict[eid] = set(cands)
                            
                    if out_file_handle is not None:
                        cands_str = ",".join(sorted(cands))
                        out_file_handle.write(f"{eid}\t{cands_str}\n")
                        
        t_query = time.time() - t_query_start
        print(f"[{country}] Processed {s1_processed:,} S1 records in {t_query:.2f}s")
        
        # Free memory for this partition
        index.clear()
        del index
        gc.collect()
        
        return s1_processed

    def run_blocking(
        self,
        s1_path: str,
        s2_path: str,
        s3_path: str,
        output_path: Optional[str] = None,
        return_candidates: bool = False
    ) -> Dict[str, Set[str]]:
        """
        Full blocking pipeline:
        1. Dynamically discovers country partitions.
        2. Iterates country-by-country to ensure low memory consumption.
        3. Streams candidate pairs to output_path if provided.
        4. Optionally returns candidate dictionary in memory.
        """
        print("Starting Entity Resolution Blocking Pipeline...")
        t_start = time.time()
        
        countries = self.discover_countries(s1_path, s2_path, s3_path)
        print(f"Discovered {len(countries)} country partition(s): {sorted(countries)}")
        
        candidate_map: Dict[str, Set[str]] = {} if return_candidates else None
        
        out_f = None
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            out_f = open(output_path, "w", encoding="utf-8")
            out_f.write("source1_entity_id\tcandidate_entity_ids\n")
            
        try:
            total_s1 = 0
            for country in sorted(countries):
                s1_count = self.generate_candidates_for_country(
                    country=country,
                    s1_path=s1_path,
                    s2_path=s2_path,
                    s3_path=s3_path,
                    candidate_dict=candidate_map,
                    out_file_handle=out_f
                )
                total_s1 += s1_count
        finally:
            if out_f:
                out_f.close()
                
        total_time = time.time() - t_start
        print(f"Blocking complete in {total_time:.2f}s for {total_s1:,} S1 records.")
        return candidate_map or {}

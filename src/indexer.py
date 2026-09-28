"""
Inverted Index implementation for Business Entity Resolution Blocking.

Provides country-partitioned inverted indices with tiered frequency caps
and zero-candidate fallback handling:
1. Business Name Tokens (cap ~2000)
2. Address Words (cap ~1500)
3. Address Shingles & Numbers (cap ~4000)
4. Character N-Grams (cap ~200)
5. Zero-Candidate Fallback (guarantees candidate generation via rarest token)
"""

from collections import defaultdict
from typing import Dict, List, Set, Optional
from .tokenization import (
    extract_name_tokens,
    extract_address_tokens,
    extract_name_ngrams
)


class CountryInvertedIndex:
    """
    Maintains inverted index structures for a specific country partition.
    Maps:
      token -> list of entity IDs (from Source 2 and Source 3)
    """

    def __init__(
        self,
        country: str,
        name_token_cap: int = 2000,
        addr_word_cap: int = 1500,
        shingle_cap: int = 4000,
        ngram_cap: int = 200,
        use_ngrams: bool = True,
        ngram_n: int = 4
    ):
        self.country = country
        self.name_token_cap = name_token_cap
        self.addr_word_cap = addr_word_cap
        self.shingle_cap = shingle_cap
        self.ngram_cap = ngram_cap
        self.use_ngrams = use_ngrams
        self.ngram_n = ngram_n
        
        self.index: Dict[str, List[str]] = defaultdict(list)
        self.total_indexed: int = 0

    def add_record(self, entity_id: str, business_name: Optional[str], business_address: Optional[str]) -> None:
        """Indexes a Source 2 or Source 3 record into the inverted index."""
        self.total_indexed += 1
        
        # 1. Business name tokens
        name_tokens = extract_name_tokens(business_name)
        for token in set(name_tokens):
            self.index[token].append(entity_id)
            
        # 2. Address tokens (words, numbers, compound shingles)
        addr_tokens = extract_address_tokens(business_address)
        for token in set(addr_tokens):
            self.index[token].append(entity_id)
            
        # 3. Optional Character N-grams
        if self.use_ngrams:
            ngrams = extract_name_ngrams(business_name, n=self.ngram_n)
            for ng in set(ngrams):
                self.index[ng].append(entity_id)

    def query_candidates(
        self,
        business_name: Optional[str],
        business_address: Optional[str]
    ) -> Set[str]:
        """
        Generates candidates for a query record by taking the UNION of:
        - Name token candidates
        - Address token candidates
        - Optional N-gram candidates
        Applying tiered safety posting caps to balance recall and precision.
        """
        candidates: Set[str] = set()
        
        # 1. Query Name tokens
        name_tokens = extract_name_tokens(business_name)
        for token in name_tokens:
            postings = self.index.get(token)
            if postings and len(postings) <= self.name_token_cap:
                candidates.update(postings)
                
        # 2. Query Address tokens (tiered: higher cap for compound shingles and numbers)
        addr_tokens = extract_address_tokens(business_address)
        for token in addr_tokens:
            postings = self.index.get(token)
            if postings:
                cap = self.shingle_cap if (token.startswith("sh_") or token.startswith("num_")) else self.addr_word_cap
                if len(postings) <= cap:
                    candidates.update(postings)
                    
        # 3. Query N-gram tokens
        ngrams: List[str] = []
        if self.use_ngrams:
            ngrams = extract_name_ngrams(business_name, n=self.ngram_n)
            for ng in ngrams:
                postings = self.index.get(ng)
                if postings and len(postings) <= self.ngram_cap:
                    candidates.update(postings)
                    
        # 4. Zero-Candidate Fallback: if no candidates found due to caps, query rarest available token
        if not candidates:
            all_tokens = name_tokens + addr_tokens + ngrams
            available_tokens = [(len(self.index[t]), t) for t in all_tokens if t in self.index]
            if available_tokens:
                rarest_token = min(available_tokens)[1]
                candidates.update(self.index[rarest_token])
                
        return candidates

    def clear(self) -> None:
        """Clears index to release memory."""
        self.index.clear()
        self.total_indexed = 0

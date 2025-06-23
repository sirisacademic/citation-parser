# core.py

# Import necessary modules
import re
import time
import json
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib

import requests
import pandas as pd
from tqdm.notebook import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    pipeline,
)
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset

THRESHOLD_PARWISE_MODEL = 0.80 # Higher threshold for SELECT model
THRESHOLD_NER_SIMILARITY = 0.70 # Lower threshold for NER similarity

def _safe_import():
    """Safely import internal modules with fallback handling"""
    # Try absolute imports first (works with pip install)
    try:
        from references_tractor.search import search_api
        from references_tractor.search import citation_formatter
        from references_tractor.utils.entity_validation import EntityValidator
        return search_api, citation_formatter, EntityValidator
    except ImportError:
        # Fallback to relative imports (works with -e install and development)
        try:
            from .search import search_api
            from .search import citation_formatter
            from .utils.entity_validation import EntityValidator
            return search_api, citation_formatter, EntityValidator
        except ImportError:
            # Final fallback for development/testing
            import sys
            import os
            from pathlib import Path
            
            # Add project root to path
            current_dir = Path(__file__).parent
            project_root = current_dir.parent if current_dir.name == 'references_tractor' else current_dir
            sys.path.insert(0, str(project_root))
            
            try:
                from references_tractor.search import search_api
                from references_tractor.search import citation_formatter
                from references_tractor.utils.entity_validation import EntityValidator
                return search_api, citation_formatter, EntityValidator
            except ImportError as e:
                raise ImportError(f"Could not import required modules. Please ensure references_tractor is properly installed. Error: {e}")

# Import internal modules
search_api, citation_formatter, EntityValidator = _safe_import()

class ReferencesTractor:

    def __init__(
        self,
        ner_model_path: str = "SIRIS-Lab/citation-parser-ENTITY",
        select_model_path: str = "SIRIS-Lab/citation-parser-SELECT",
        prescreening_model_path: str = "SIRIS-Lab/citation-parser-TYPE",
        span_model_path: str = "SIRIS-Lab/citation-parser-SPAN",
        device: Union[int, str] = "auto",
        enable_caching: bool = True,
        cache_size_limit: int = 1000,
        select_threshold: float = THRESHOLD_PARWISE_MODEL,
        ner_threshold: float = THRESHOLD_NER_SIMILARITY
    ):
        # Auto-detect device if not specified
        if device == "auto":
            device = self._detect_best_device()
            print(f"Auto-detected device: {device}")
        elif device == "cpu":
            print("Using CPU for model inference")
        else:
            print(f"Using specified device: {device}")
        
        # Initialize three different transformer pipelines:
        # 1. NER for citation entity extraction
        # 2. Selection model to rank possible citation matches
        # 3. Prescreening model to filter non-citation inputs
        self.ner_pipeline = self._init_pipeline("ner", ner_model_path, device, agg_strategy="simple")
        self.select_pipeline = self._init_pipeline("text-classification", select_model_path, device)
        self.prescreening_pipeline = self._init_pipeline("text-classification", prescreening_model_path, device)
        self.span_pipeline = self._init_pipeline("ner", span_model_path, device, agg_strategy="simple")
        self.searcher = search_api.SearchAPI()
        
        # Initialize caching system
        self.enable_caching = enable_caching
        self.cache_size_limit = cache_size_limit
        self._citation_cache = {}
        self._cache_stats = {'hits': 0, 'misses': 0, 'size': 0}

        # Initialize thresholds
        self.select_threshold = select_threshold
        self.ner_threshold = ner_threshold
        

    def _detect_best_device(self) -> str:
        """
        Auto-detect the best available device for model inference
        Returns: device string ("cuda", "mps", or "cpu")
        """
        try:
            import torch
            
            # Check for NVIDIA CUDA GPU
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                print(f"CUDA GPU detected: {gpu_name} (GPU count: {gpu_count})")
                return "cuda"
            
            # Check for Apple Silicon MPS (Metal Performance Shaders)
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("Apple Silicon GPU (MPS) detected")
                return "mps"
            
            # Fallback to CPU
            else:
                import multiprocessing
                cpu_count = multiprocessing.cpu_count()
                print(f"No GPU detected, using CPU ({cpu_count} cores)")
                return "cpu"
                
        except ImportError:
            print("PyTorch not found, defaulting to CPU")
            return "cpu"
        except Exception as e:
            print(f"Error during device detection: {e}, defaulting to CPU")
            return "cpu"

    def _init_pipeline(
        self, task: str, model_path: str, device: Union[int, str], agg_strategy: Optional[str] = None
    ):
        # Helper to initialize the appropriate transformer pipeline with enhanced device handling
        try:
            kwargs = {
                "model": AutoModelForTokenClassification.from_pretrained(model_path)
                if task == "ner"
                else AutoModelForSequenceClassification.from_pretrained(model_path),
                "tokenizer": AutoTokenizer.from_pretrained(model_path),
                "device": device,
            }
            if agg_strategy:
                kwargs["aggregation_strategy"] = agg_strategy
            
            pipeline_obj = pipeline(task, **kwargs)
            
            # Verify device placement
            actual_device = next(pipeline_obj.model.parameters()).device
            
            # More specific logging based on model path
            model_name = model_path.split('/')[-1].replace('citation-parser-', '').upper()
            print(f"{model_name} model loaded on device: {actual_device}")
            
            return pipeline_obj
            
        except Exception as e:
            model_name = model_path.split('/')[-1].replace('citation-parser-', '').upper()
            print(f"Error loading {model_name} model on {device}: {e}")
            print("Falling back to CPU...")
            
            # Fallback to CPU
            kwargs["device"] = "cpu"
            pipeline_obj = pipeline(task, **kwargs)
            print(f"{model_name} model loaded on CPU (fallback)")
            return pipeline_obj

    def _generate_cache_key(self, citation: str, api_target: str, output: str) -> str:
        """Generate a consistent cache key for the given parameters"""
        # Create a hash of the parameters to handle long citations
        key_string = f"{citation}|{api_target}|{output}"
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()
    
    def _manage_cache_size(self):
        """Remove oldest entries if cache exceeds size limit"""
        if len(self._citation_cache) > self.cache_size_limit:
            # Remove oldest 20% of entries (simple FIFO)
            items_to_remove = len(self._citation_cache) - int(self.cache_size_limit * 0.8)
            keys_to_remove = list(self._citation_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self._citation_cache[key]
            self._cache_stats['size'] = len(self._citation_cache)
    
    def clear_cache(self):
        """Clear the citation cache and reset stats"""
        self._citation_cache.clear()
        self._cache_stats = {'hits': 0, 'misses': 0, 'size': 0}
        print("Citation cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get current cache statistics"""
        total_requests = self._cache_stats['hits'] + self._cache_stats['misses']
        hit_rate = (self._cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_enabled': self.enable_caching,
            'cache_size': len(self._citation_cache),
            'cache_limit': self.cache_size_limit,
            'cache_hits': self._cache_stats['hits'],
            'cache_misses': self._cache_stats['misses'],
            'hit_rate': f"{hit_rate:.1f}%",
            'total_requests': total_requests
        }

    def process_ner_entities(self, citation: str) -> Dict[str, List[str]]:
        # Extract named entities from the citation using the NER pipeline
        output = self.ner_pipeline(citation)
        entities = {}
        for entity in output:
            key = entity.get("entity_group")
            entities.setdefault(key, []).append(entity.get("word", ""))
        # Clean and validate entities using utility
        cleaned_entities = EntityValidator.validate_and_clean_entities(entities)
        return cleaned_entities

    def generate_apa_citation(self, data: dict, api: str = "openalex") -> str:
        # Format a citation from retrieved metadata in APA style
        # PubMed now uses parsed dict structure like other APIs - no special handling needed
        formatter = citation_formatter.CitationFormatterFactory.get_formatter(api)
        return formatter.generate_apa_citation(data)

    def compute_ner_similarity(self, original_citation: str, candidate_citation: str) -> float:
        """
        Compute similarity between two citations based on NER-extracted fields.
        Returns a score between 0.0 and 1.0, where 1.0 is perfect match.
        
        Args:
            original_citation: The input citation string
            candidate_citation: The formatted candidate citation to compare
            
        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        import re
        from difflib import SequenceMatcher
        
        # Extract NER entities from both citations
        original_entities = self.process_ner_entities(original_citation)
        candidate_entities = self.process_ner_entities(candidate_citation)
        
        total_score = 0.0
        field_weights = {
            'TITLE': 0.6,
            'AUTHORS': 0.3,
            'PUBLICATION_YEAR': 0.2,
            'DOI': 0.05
        }
        
        def normalize_text(text):
            """Normalize text for comparison"""
            if not text:
                return ""
            # Convert to lowercase, remove extra spaces, punctuation
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        def fuzzy_similarity(text1, text2):
            """Compute fuzzy string similarity using SequenceMatcher"""
            if not text1 or not text2:
                return 0.0
            norm1 = normalize_text(text1)
            norm2 = normalize_text(text2)
            if not norm1 or not norm2:
                return 0.0
            return SequenceMatcher(None, norm1, norm2).ratio()
        
        def author_similarity(authors1, authors2):
            """Compute author similarity using fuzzy matching with word overlap"""
            if not authors1 or not authors2:
                return 0.0
            
            # Simple approach: combine fuzzy similarity with word overlap
            fuzzy_score = fuzzy_similarity(authors1, authors2)
            
            # Extract meaningful words (length > 2, not common words)
            def extract_author_words(text):
                # Clean text and extract words
                clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
                words = clean_text.split()
                # Filter out common non-name words and very short words
                stop_words = {'et', 'al', 'and', 'the', 'of', 'in', 'at', 'to', 'for', 'with'}
                return [w for w in words if len(w) > 2 and w not in stop_words and not w.isdigit()]
            
            words1 = set(extract_author_words(authors1))
            words2 = set(extract_author_words(authors2))
            
            if not words1 or not words2:
                return fuzzy_score
            
            # Calculate word overlap (Jaccard similarity)
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            word_overlap = intersection / union if union > 0 else 0.0
            
            # Combine fuzzy similarity with word overlap
            return max(fuzzy_score, word_overlap * 0.8)  # Give word overlap slightly less weight
        
        def extract_any_year(text):
            """Extract any 4-digit year from text"""
            if not text:
                return None
            year_match = re.search(r'\b(19|20)\d{2}\b', str(text))
            return int(year_match.group()) if year_match else None
        
        def year_similarity(year1, year2):
            """Year matching with fallback extraction from any field"""
            # Try direct extraction first
            y1 = extract_any_year(year1)
            y2 = extract_any_year(year2)
            
            if y1 is None or y2 is None:
                return 0.0
            
            # Year comparison with tolerance
            diff = abs(y1 - y2)
            if diff == 0:
                return 1.0
            elif diff == 1:
                return 0.7  # 1-year difference (common for online/print dates)
            elif diff == 2:
                return 0.3  # 2-year difference
            else:
                return 0.0
        
        def doi_similarity(doi1, doi2):
            """DOI exact matching"""
            if not doi1 or not doi2:
                return 0.0
            
            # Normalize DOIs (remove prefixes)
            def normalize_doi(doi):
                doi = doi.lower().strip()
                prefixes = ['https://doi.org/', 'http://doi.org/', 'doi:', 'doi ']
                for prefix in prefixes:
                    if doi.startswith(prefix):
                        doi = doi[len(prefix):]
                return doi.strip()
            
            norm_doi1 = normalize_doi(doi1)
            norm_doi2 = normalize_doi(doi2)
            
            return 1.0 if norm_doi1 == norm_doi2 else 0.0
        
        # Calculate field-specific similarities with fallback extraction
        field_scores = {}
        
        # Create combined text for fallback searches
        orig_combined = f"{original_citation} " + " ".join([" ".join(v) for v in original_entities.values()])
        cand_combined = f"{candidate_citation} " + " ".join([" ".join(v) for v in candidate_entities.values()])
        
        for field, weight in field_weights.items():
            orig_value = original_entities.get(field, [])
            cand_value = candidate_entities.get(field, [])
            
            # Get first value from lists (cleaned by EntityValidator)
            orig_text = orig_value[0] if orig_value else ""
            cand_text = cand_value[0] if cand_value else ""
            
            # Generic fallback: if field not found in NER, try extracting from full text
            if not orig_text and field == 'PUBLICATION_YEAR':
                orig_text = str(extract_any_year(orig_combined) or "")
            if not cand_text and field == 'PUBLICATION_YEAR':
                cand_text = str(extract_any_year(cand_combined) or "")
            
            if field == 'TITLE':
                field_scores[field] = fuzzy_similarity(orig_text, cand_text)
            elif field == 'AUTHORS':
                field_scores[field] = author_similarity(orig_text, cand_text)
            elif field == 'PUBLICATION_YEAR':
                field_scores[field] = year_similarity(orig_text, cand_text)
            elif field == 'DOI':
                field_scores[field] = doi_similarity(orig_text, cand_text)
            
            # Add weighted score
            total_score += field_scores[field] * weight
        
        return min(total_score, 1.0)  # Ensure score doesn't exceed 1.0

    def get_highest_true_position(
        self, outputs: List[List[Dict[str, Any]]], inputs: List[Any], 
        original_citation: str = None, api_target: str = None
    ) -> Tuple[Optional[Any], Optional[float]]:

        # Get True labels with confidence threshold
        confident_true_scores = []
        uncertain_true_scores = []
        
        for i, result in enumerate(outputs):
            if result[0]['label'] is True:
                score = result[0]['score']
                if score >= self.select_threshold:
                    confident_true_scores.append((i, score))
                else:
                    uncertain_true_scores.append((i, score))
        
        # First try confident True labels
        if confident_true_scores:
            best_index, best_score = max(confident_true_scores, key=lambda x: x[1])  # Fixed: unpack tuple
            self._last_scoring_method = 'select_model'
            return inputs[best_index], best_score
        
        # If only uncertain True labels, validate with NER
        if uncertain_true_scores and original_citation and api_target:
            validated_candidates = []
            for i, select_score in uncertain_true_scores:
                try:
                    formatted_citation = self.generate_apa_citation(inputs[i], api=api_target)
                    ner_score = self.compute_ner_similarity(original_citation, formatted_citation)
                    
                    if ner_score >= self.ner_threshold:  # NER validation threshold
                        validated_candidates.append((i, ner_score))  # Use NER score

                except Exception as e:
                    print(f"Error computing NER for candidate {i}: {e}")
            
            if validated_candidates:
                best_index, best_score = max(validated_candidates, key=lambda x: x[1])  # Fixed: unpack tuple
                self._last_scoring_method = 'ner_similarity'
                return inputs[best_index], best_score
        
        # Fallback to NER-only similarity (no True labels or validation failed)
        
        if not original_citation or not api_target:
            return None, None
        
        ner_scores = []
        for i, pub in enumerate(inputs):
            try:
                formatted_citation = self.generate_apa_citation(pub, api=api_target)
                ner_score = self.compute_ner_similarity(original_citation, formatted_citation)
                ner_scores.append((i, ner_score))
            except Exception as e:
                print(f"Error computing NER similarity for candidate {i}: {e}")
                ner_scores.append((i, 0.0))
        
        if ner_scores:
            best_index, best_score = max(ner_scores, key=lambda x: x[1])  # Fixed: unpack tuple
            
            # Apply NER threshold even in fallback
            if best_score < self.ner_threshold:
                return None, None
            
            self._last_scoring_method = 'ner_similarity'
            return inputs[best_index], best_score
        
        return None, None

    def search_api(self, ner_entities: Dict[str, List[str]], api: str = "openalex", 
                      target_count: int = 10) -> List[dict]:
            # Search a bibliographic API using extracted NER entities with progressive strategy
            return self.searcher.search_api(ner_entities, api=api, target_count=target_count)
        
    def get_uri(self, pid: Optional[str], doi: Optional[str], api: str) -> Optional[str]:
        # Construct the canonical URL to the publication based on available identifiers
        uri_templates = {
            "openalex": lambda: f"https://openalex.org/{pid}" if pid else None,
            "openaire": lambda: f"https://explore.openaire.eu/search/publication?pid={doi or pid}" if pid else None,
            "pubmed": lambda: f"https://pubmed.ncbi.nlm.nih.gov/{pid}" if pid else None,
            "crossref": lambda: f"https://doi.org/{doi}" if doi else None,
            "hal": lambda: f"https://hal.science/{pid}" if pid else None,
        }
        return uri_templates.get(api, lambda: None)()

    def extract_id(self, publication: dict, api: str) -> Optional[str]:
        # Extract the publication ID depending on the API source
        if publication and isinstance(publication, dict):
            if api == "openalex":
                return publication.get("id", "").replace("https://openalex.org/", "")
            elif api == "openaire":
                return publication.get('id')
            elif api == "pubmed":
                return publication.get("pmid") or publication.get("id")
            elif api == "crossref":
                return publication.get("DOI")
            elif api == "hal":
                return publication.get("halId_s")
        return None

    def extract_doi(self, publication: dict, api: str) -> Optional[str]:
        """Extract the DOI depending on the API source - Updated for PubMed dict structure"""
        
        # First try to get from enhanced structure
        if isinstance(publication, dict) and 'main_doi' in publication:
            return publication.get('main_doi')
        
        # Handle PubMed - now expects parsed dict structure
        if api == "pubmed":
            if isinstance(publication, dict):
                # Try direct DOI field from parsed structure
                doi = publication.get('doi')
                if doi:
                    return doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
                
                # Fallback: if still has xml_content, try parsing
                if 'xml_content' in publication:
                    try:
                        import xml.etree.ElementTree as ET
                        root = ET.fromstring(publication['xml_content'])
                        doi_elem = root.find(".//ELocationID[@EIdType='doi']")
                        if doi_elem is not None:
                            return doi_elem.text.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
                    except Exception:
                        pass
            return None
        
        # Fallback to original extraction logic for other APIs
        if api == "openalex":
            doi = publication.get("doi")
            if isinstance(doi, str):
                return doi.replace("https://doi.org/", "")
            return None
        elif api == "openaire":
            # Check enhanced structure first
            if 'pids' in publication:
                identifiers = publication.get('pids', [])
                if isinstance(identifiers, dict):
                    identifiers = [identifiers]
                for pid in identifiers:
                    if isinstance(pid, dict) and pid.get("scheme") == "doi":
                        return pid.get("value", "").replace("https://doi.org/", "")
            return None
        elif api == "crossref":
            return publication.get("DOI")
        elif api == "hal":
            return publication.get("doiId_s")
        return None

    def link_citation(self, citation: str, output: str = 'simple', api_target: str = 'openalex') -> Dict[str, Any]:
        """
        Main function to process a citation string with caching support:
        - Check if it's a valid citation
        - Extract entities
        - Search target API
        - Format results and rank them
        """
        # Check cache first if enabled
        if self.enable_caching:
            cache_key = self._generate_cache_key(citation, api_target, output)
            if cache_key in self._citation_cache:
                self._cache_stats['hits'] += 1
                return self._citation_cache[cache_key].copy()  # Return copy to avoid mutation
            else:
                self._cache_stats['misses'] += 1
        
        # Prescreen input to ensure it's likely a citation
        if not self.prescreening_pipeline(citation)[0]["label"]:
            result = {"error": "This text is not a citation. Please introduce a valid citation."}
            if self.enable_caching:
                self._citation_cache[cache_key] = result.copy()
                self._manage_cache_size()
            return result

        ner_entities = self.process_ner_entities(citation)
        pubs = self.search_api(ner_entities, api=api_target, target_count=10)

        if not pubs:
            result = {}
            if self.enable_caching:
                self._citation_cache[cache_key] = result.copy()
                self._manage_cache_size()
            return result

        # Format candidate citations and classify best match
        cits = [self.generate_apa_citation(pub, api=api_target) for pub in pubs]
        pairwise_scores = [self.select_pipeline(f"{citation} [SEP] {cit}") for cit in cits]

        # If only one candidate, check if SELECT model is confident
        if len(cits) == 1:
            selected_score = pairwise_scores[0][0]
            pub = pubs[0]
            
            # Check if SELECT model returned False OR low confidence - use NER similarity
            if selected_score['label'] is False or selected_score['score'] < self.select_threshold:
                ner_score = self.compute_ner_similarity(citation, cits[0])
                
                # Apply NER threshold
                if ner_score < self.ner_threshold:
                    result = {}
                    if self.enable_caching:
                        self._citation_cache[cache_key] = result.copy()
                        self._manage_cache_size()
                    return result
                
                # Use NER score
                final_score = ner_score
                score_data = {"score": final_score}
            else:
                # SELECT model confident and True
                final_score = selected_score['score']
                score_data = selected_score
            
            pub_id = self.extract_id(pub, api_target)
            pub_doi = self.extract_doi(pub, api_target)
            url = self.get_uri(pub_id, pub_doi, api_target)
            result = self._format_result(cits[0], score_data, pub_id, pub_doi, url, pub, output, api_target)
            
            if self.enable_caching:
                self._citation_cache[cache_key] = result.copy()
                self._manage_cache_size()
            return result

        # Choose the most likely correct match using classification scores
        reranked_pub, best_score = self.get_highest_true_position(
            pairwise_scores, pubs, citation, api_target
        )
        if reranked_pub:
            pub_id = self.extract_id(reranked_pub, api_target)
            pub_doi = self.extract_doi(reranked_pub, api_target)
            url = self.get_uri(pub_id, pub_doi, api_target)
            formatted_cit = self.generate_apa_citation(reranked_pub, api=api_target)
            result = self._format_result(formatted_cit, {"score": best_score}, pub_id, pub_doi, url, reranked_pub, output, api_target)
            
            if self.enable_caching:
                self._citation_cache[cache_key] = result.copy()
                self._manage_cache_size()
            return result

        result = {}
        if self.enable_caching:
            self._citation_cache[cache_key] = result.copy()
            self._manage_cache_size()
        
        return result

    def _format_result(
        self, citation: str, score_data: dict, pub_id: Optional[str], doi: Optional[str],
        url: Optional[str], pub: dict, output: str, api_target: str
    ) -> Dict[str, Any]:
        """Helper to format the output result with enhanced DOI support"""
        
        # Extract enhanced DOI information from publication
        main_doi = pub.get('main_doi')
        alternative_dois = pub.get('alternative_dois', [])
        total_dois = pub.get('total_dois', 0)
        all_dois = pub.get('all_dois', [])
        
        # If enhanced structure not available, use legacy DOI
        if not main_doi:
            main_doi = doi
            alternative_dois = []
            total_dois = 1 if doi else 0
            all_dois = [doi] if doi else []
        
        result = {
            "result": citation,
            "score": score_data.get("score", False),
            f"{api_target}_id": pub_id,
            "doi": main_doi,  # Backward compatibility - use main DOI
            "url": url,
            # Enhanced DOI information
            "main_doi": main_doi,
            "alternative_dois": alternative_dois,
            "total_dois": total_dois,
            "all_dois": all_dois
        }
        
        if output == "advanced":
            result["full-publication"] = pub
        
        return result

    def _should_include_in_ensemble_voting(self, result: Dict[str, Any], original_citation: str, api: str) -> bool:
        """
        Determine if a result should be included in ensemble DOI voting.
        
        Applies the same quality thresholds that individual APIs use to ensure
        ensemble only considers results that would pass individual API validation.
        
        Args:
            result: The API result to evaluate
            original_citation: The original citation text for similarity comparison
            api: The source API name
            
        Returns:
            bool: True if result should contribute to ensemble voting, False otherwise
        """
        # Must have a DOI to contribute to ensemble voting
        main_doi = result.get('main_doi') or result.get('doi')
        if not main_doi:
            return False
        
        # Apply same quality thresholds as individual APIs
        formatted_citation = result.get('result')
        if formatted_citation:
            try:
                ner_score = self.compute_ner_similarity(original_citation, formatted_citation)
                if ner_score < self.ner_threshold:
                    return False
            except Exception:
                return False
        
        return True

    def link_citation_ensemble(
            self, citation: str,
            output: str = 'simple',
            api_targets: List[str] = ['openalex', 'openaire', 'pubmed', 'crossref', 'hal']
        ) -> Dict[str, Any]:
            """
            Attempts to link a citation using multiple APIs in an ensemble fashion.
            Enhanced to consider alternative DOIs in voting for improved accuracy.
            Selects the most agreed-upon DOI among sources.
            Now benefits from caching - individual API calls may be cached.
            """
            doi_counter = Counter()
            extract_ids = {}
            missing_sources = []
            api_results = {}  # Store full results for enhanced processing

            # Try to link using each API (these calls will use cache if available)
            for api in api_targets:
                try:
                    res = self.link_citation(citation, output=output, api_target=api)

                    if not self._should_include_in_ensemble_voting(res, citation, api):
                        missing_sources.append(api)
                        continue

                    # Store full result for enhanced processing
                    api_results[api] = res
                    
                    # Extract main DOI and alternative DOIs if available
                    main_doi = res.get("doi")
                    alternative_dois = res.get("alternative_dois", [])
                    
                    if main_doi:
                        # Enhanced voting: Count main DOI
                        doi_counter[main_doi] += 1.0
                        
                        # Store the API ID for this DOI
                        api_id_key = f"{api}_id"
                        extract_ids[api_id_key] = res.get(api_id_key, None)
                    
                    # Enhanced voting: Count alternative DOIs with equal weight
                    if alternative_dois:
                        for alt_doi in alternative_dois:
                            if alt_doi and alt_doi != main_doi:  # Avoid double-counting
                                doi_counter[alt_doi] += 1.0
                    
                    # If no DOIs found, mark as missing
                    if not main_doi and not alternative_dois:
                        missing_sources.append(api)
                        
                except Exception as e:
                    missing_sources.append(api)

            # Enhanced ensemble decision
            if not doi_counter:
                return {"doi": None, "external_ids": {}}

            # Choose DOI with most votes (main + alternative DOIs counted equally)
            best_doi, vote_count = doi_counter.most_common(1)[0]

            # Enhanced backfill: Try to get IDs for the selected DOI from APIs that missed it
            for api in missing_sources:
                try:
                    # Search specifically for the selected DOI (this will also use cache)
                    pubs = self.search_api({'DOI': [best_doi]}, api=api, target_count=1)
                    if pubs:
                        pub_id = self.extract_id(pubs[0], api) 
                        if pub_id:
                            extract_ids[f"{api}_id"] = pub_id
                except Exception as e:
                    pass  # Fail silently for backfill

            # Enhanced result: Include voting information and alternative DOI info
            result = {
                "doi": best_doi,
                "external_ids": extract_ids
            }
            
            # Add enhanced ensemble metadata
            if output == 'advanced':
                # Find all DOIs that point to the same paper
                related_dois = [doi for doi, votes in doi_counter.items() if votes > 0]
                
                # Count contributing APIs
                contributing_apis = []
                for api, api_result in api_results.items():
                    main_doi = api_result.get("doi")
                    alternative_dois = api_result.get("alternative_dois", [])
                    
                    # Check if this API contributed to the selected DOI
                    if (main_doi == best_doi or 
                        best_doi in alternative_dois or
                        main_doi in related_dois):
                        contributing_apis.append(api)
                
                result["ensemble_metadata"] = {
                    "selected_doi_votes": vote_count,
                    "total_dois_found": len(related_dois),
                    "all_related_dois": related_dois,
                    "contributing_apis": contributing_apis,
                    "doi_vote_breakdown": dict(doi_counter.most_common())
                }

            return result
    

    def extract_and_link_from_text(self, text: str, api_target: str = 'openalex') -> Dict[str, Dict[str, Any]]:
        """
        Extract citation entities from the provided text and link them to bibliographic data.
        Now benefits from caching - repeated citations will be cached.
        
        Args:
            text (str): The input text from which entities will be extracted.
            api_target (str): The target API to use for citation linking (default is 'openalex').

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary where each key is an entity and the value is the linked citation data.
        """
        # Step 1: Use the NER pipeline to extract entities from the text
        ner_entities = self.span_pipeline(text)
        
        # Initialize the result dictionary
        linked_entities = {}

        long_entities = [entity['word'] for entity in ner_entities if len(entity['word']) > 20]


        # Step 2: For each entity group, extract each entity and link it (with caching)
        for entity in long_entities:
            # Step 3: Link the entity to a citation using the link_citation method (cached)
            linked_data = self.link_citation(entity, api_target=api_target)

            # Step 4: Add the linked citation data to the result dictionary
            linked_entities[entity] = linked_data

        return linked_entities

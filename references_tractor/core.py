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

# Import internal search and citation formatting tools
from .search import search_api
from .search import citation_formatter

# Import NER entity cleaner/validator
from .utils.entity_validation import EntityValidator

class ReferencesTractor:
    """
    Class to extract citation entities, search bibliographic databases, 
    and generate and link citations to their canonical records.
    Enhanced with pipeline-level caching to avoid duplicate API calls.
    """

    def __init__(
        self,
        ner_model_path: str = "SIRIS-Lab/citation-parser-ENTITY",
        select_model_path: str = "SIRIS-Lab/citation-parser-SELECT",
        prescreening_model_path: str = "SIRIS-Lab/citation-parser-TYPE",
        span_model_path: str = "SIRIS-Lab/citation-parser-SPAN",
        device: Union[int, str] = "cpu",
        enable_caching: bool = True,
        cache_size_limit: int = 1000,
    ):
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

    def _init_pipeline(
        self, task: str, model_path: str, device: Union[int, str], agg_strategy: Optional[str] = None
    ):
        # Helper to initialize the appropriate transformer pipeline
        kwargs = {
            "model": AutoModelForTokenClassification.from_pretrained(model_path)
            if task == "ner"
            else AutoModelForSequenceClassification.from_pretrained(model_path),
            "tokenizer": AutoTokenizer.from_pretrained(model_path),
            "device": device,
        }
        if agg_strategy:
            kwargs["aggregation_strategy"] = agg_strategy
        return pipeline(task, **kwargs)

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

    def get_highest_true_position(
        self, outputs: List[List[Dict[str, Any]]], inputs: List[Any]
    ) -> Tuple[Optional[Any], Optional[float]]:
        # Return the input with the highest "True" classification score
        true_scores = [
            (i, result[0]['score']) if result[0]['label'] is True else (i, 0.0)
            for i, result in enumerate(outputs)
        ]
        if not true_scores:
            return None, None
        best_index = max(true_scores, key=lambda x: x[1])[0]
        return inputs[best_index], true_scores[best_index][1]
       
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
        if api == "openalex":
            return publication.get("id", "").replace("https://openalex.org/", "")
        elif api == "openaire":
            return publication.get('header', {}).get('dri:objIdentifier', {}).get('$')
        elif api == "pubmed":
            # Now expects parsed dict structure instead of raw XML
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

        # If only one candidate, return it
        if len(cits) == 1:
            selected_score = pairwise_scores[0][0]
            pub = pubs[0]
            pub_id = self.extract_id(pub, api_target)
            pub_doi = self.extract_doi(pub, api_target)
            url = self.get_uri(pub_id, pub_doi, api_target)
            result = self._format_result(cits[0], selected_score, pub_id, pub_doi, url, pub, output, api_target)
            
            if self.enable_caching:
                self._citation_cache[cache_key] = result.copy()
                self._manage_cache_size()
            return result

        # Choose the most likely correct match using classification scores
        reranked_pub, best_score = self.get_highest_true_position(pairwise_scores, pubs)
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

    def link_citation_ensemble(
            self, citation: str, output: str = 'simple',
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
                    res = self.link_citation(citation, output="advanced", api_target=api)
                    
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

# search/progressive_search.py
"""
Implements progressive search strategy with target-based candidate collection.
Enhanced with multiple DOI deduplication support and API-specific retry configurations.
Clean output with reduced verbosity.
"""

import time
import requests
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict

from .api_capabilities import APICapabilities
from .field_mapper import FieldMapper, DOIResult

class ResultDeduplicator:
    """Handles deduplication of search results across different strategies with multiple DOI support"""
    
    def __init__(self, field_mapper: FieldMapper):
        self.field_mapper = field_mapper

    def deduplicate_candidates(self, candidates: List[Dict], api: str) -> List[Dict]:
        """Remove duplicate candidates based on DOI or publication ID with multiple DOI awareness"""
        seen_dois = set()
        seen_ids = set()
        unique_candidates = []
        
        response_fields = APICapabilities.get_response_fields(api)
        
        for candidate in candidates:
            should_keep = True
            candidate_dois = set()
            candidate_id = None
            
            # Extract all DOIs from this candidate (if enhanced with multiple DOI support)
            if 'all_dois' in candidate and candidate['all_dois']:
                candidate_dois.update(candidate['all_dois'])
            else:
                # Fallback: extract DOI using field mapper
                try:
                    doi_result = self.field_mapper.extract_dois_from_result(candidate, api)
                    candidate_dois.update(self.field_mapper.get_all_dois_from_result(doi_result))
                except Exception:
                    # Final fallback: try to get DOI from standard fields
                    doi_config = response_fields.get('doi')
                    if doi_config:
                        doi = self.field_mapper.extract_response_field(candidate, doi_config, api)
                        if doi:
                            cleaned_doi = self.field_mapper.clean_doi(doi)
                            if cleaned_doi:
                                candidate_dois.add(cleaned_doi)
            
            # Check for DOI overlap with previously seen candidates
            if candidate_dois:
                doi_overlap = candidate_dois.intersection(seen_dois)
                if doi_overlap:
                    should_keep = False
                else:
                    seen_dois.update(candidate_dois)
            
            # If no DOI available or no overlap, check by publication ID
            if should_keep and not candidate_dois:
                id_config = response_fields.get('id')
                if id_config:
                    candidate_id = self.field_mapper.extract_response_field(candidate, id_config, api)
                    if candidate_id and candidate_id in seen_ids:
                        should_keep = False
                    elif candidate_id:
                        seen_ids.add(candidate_id)
            
            if should_keep:
                candidate['dedup_info'] = {
                    'dois_found': list(candidate_dois),
                    'id_found': candidate_id,
                    'dedup_method': 'doi' if candidate_dois else ('id' if candidate_id else 'title_year')
                }
                unique_candidates.append(candidate)
        
        return unique_candidates

class ProgressiveSearchStrategy:
    """Implements target-based progressive search with API-specific retry configurations and clean output"""
    
    def __init__(self, field_mapper: FieldMapper, deduplicator: 'ResultDeduplicator', 
                 default_target_count: int = 10):
        self.field_mapper = field_mapper
        self.deduplicator = deduplicator
        self.default_target_count = default_target_count
        
        # Track API call timing to respect rate limits
        self.last_api_call_time = {}
    
    def _get_api_retry_config(self, api: str):
        """Get retry configuration for specific API"""
        return APICapabilities.get_retry_config(api)
    
    def _wait_for_rate_limit(self, api: str, rate_limit_delay: float):
        """Wait for rate limit before making API call with API-specific delay"""
        current_time = time.time()
        last_call_time = self.last_api_call_time.get(api, 0)
        
        time_since_last_call = current_time - last_call_time
        if time_since_last_call < rate_limit_delay:
            sleep_time = rate_limit_delay - time_since_last_call
            time.sleep(sleep_time)
        
        self.last_api_call_time[api] = time.time()
    
    def _calculate_delay(self, attempt: int, api: str, config) -> float:
        """Calculate delay for retry attempt with API-specific configuration"""
        delay = config.base_delay * (config.backoff_multiplier ** attempt)
        return min(delay, config.max_delay)
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is worth retrying"""
        if isinstance(error, requests.exceptions.Timeout):
            return True
        elif isinstance(error, requests.exceptions.ConnectionError):
            return True
        elif isinstance(error, requests.exceptions.HTTPError):
            if hasattr(error, 'response') and error.response:
                status_code = error.response.status_code
                return 500 <= status_code < 600
            return True
        elif isinstance(error, requests.exceptions.RequestException):
            return True
        else:
            return False
    
    def _execute_search_with_retry(self, query_params: Dict[str, Any], api: str, 
                                 combination: List[str], strategy_instance: Any) -> List[Dict]:
        """Execute search with API-specific retry configuration - clean output"""
        
        # Get API-specific configuration
        config = self._get_api_retry_config(api)
        
        for attempt in range(config.max_retries):
            try:
                # Rate limiting with API-specific delay
                self._wait_for_rate_limit(api, config.rate_limit_delay)
                
                # Build API URL
                if hasattr(strategy_instance, '_build_api_url'):
                    api_url = strategy_instance._build_api_url(query_params)                   
                else:
                    return []
                
                # Make the request with API-specific timeout
                response = requests.get(api_url, timeout=config.timeout)
                
                if response.status_code == 200:
                    results = self._parse_api_response(response, api)
                    return results
                else:
                    # Don't retry on client errors (4xx)
                    if 400 <= response.status_code < 500:
                        return []
                    
                    # Retry on server errors (5xx)
                    if attempt < config.max_retries - 1:
                        delay = self._calculate_delay(attempt, api, config)
                        time.sleep(delay)
                        continue
                    else:
                        return []
                    
            except requests.exceptions.Timeout as e:
                if attempt < config.max_retries - 1:
                    delay = self._calculate_delay(attempt, api, config)
                    time.sleep(delay)
                    continue
                else:
                    return []
                    
            except requests.exceptions.ConnectionError as e:
                if attempt < config.max_retries - 1:
                    delay = self._calculate_delay(attempt, api, config)
                    time.sleep(delay)
                    continue
                else:
                    return []
                    
            except requests.exceptions.RequestException as e:
                if self._is_retryable_error(e) and attempt < config.max_retries - 1:
                    delay = self._calculate_delay(attempt, api, config)
                    time.sleep(delay)
                    continue
                else:
                    return []
                    
            except Exception as e:
                return []
        
        return []
    
    def _parse_api_response(self, response: requests.Response, api: str) -> List[Dict]:
        """Parse API response based on API type with comprehensive error handling"""
        try:
            if api == "pubmed":
                return [response.text] if response.text else []
            
            try:
                data = response.json()
            except (ValueError, requests.exceptions.JSONDecodeError):
                return []
            
            if data is None:
                return []
            
            if api == "openalex":
                results = data.get("results", [])
                return results if isinstance(results, list) else []
                
            elif api == "openaire":
                results = data.get("results")
                if results is None:
                    return []
                return results if isinstance(results, list) else []
                    
            elif api == "crossref":
                message = data.get("message")
                if message is None:
                    return []
                items = message.get("items", [])
                return items if isinstance(items, list) else []
                
            elif api == "hal":
                response_data = data.get("response")
                if response_data is None:
                    return []
                docs = response_data.get("docs", [])
                return docs if isinstance(docs, list) else []
            
            else:
                return []
            
        except Exception as e:
            return []
    
    def search_with_target(self, ner_entities: Dict[str, List[str]], api: str, 
                          target_count: Optional[int] = None, 
                          strategy_instance: Any = None) -> List[Dict]:
        """Perform progressive search with API-specific retry configurations - clean output"""
        if target_count is None:
            target_count = self.default_target_count
        
        # Get API-specific configurations
        search_capabilities = APICapabilities.get_search_fields(api)
        field_combinations = APICapabilities.get_field_combinations(api)
        
        if not search_capabilities or not field_combinations:
            return []
        
        all_candidates = []
        
        for combination in field_combinations:
            if len(all_candidates) >= target_count:
                break
            
            available_combination = [
                field for field in combination 
                if field in ner_entities and ner_entities[field] and ner_entities[field][0]
                and APICapabilities.supports_field(api, field)
            ]
            
            if not available_combination:
                continue
            
            query_params = self.field_mapper.build_query_params(
                api, available_combination, ner_entities, search_capabilities
            )
            
            if not query_params:
                continue
            
            # Execute search with API-specific retry configuration
            candidates = self._execute_search_with_retry(
                query_params, api, available_combination, strategy_instance
            )
            
            if candidates:
                all_candidates.extend(candidates)
                
                if len(candidates) == 1 and len(available_combination) >= 3:
                    break
        
        # Deduplicate all candidates (with multiple DOI awareness)
        unique_candidates = self.deduplicator.deduplicate_candidates(all_candidates, api)
        
        return unique_candidates[:target_count]

class SearchOrchestrator:
    """Orchestrates the complete search process across APIs with enhanced DOI support and clean output"""
    
    def __init__(self, target_count: int = 10):
        self.field_mapper = FieldMapper()
        self.deduplicator = ResultDeduplicator(self.field_mapper)
        self.progressive_search = ProgressiveSearchStrategy(
            self.field_mapper, self.deduplicator, target_count
        )
    
    def search_single_api(self, ner_entities: Dict[str, List[str]], api: str, 
                         target_count: Optional[int] = None, 
                         strategy_instance: Any = None) -> List[Dict]:
        """Search a single API with progressive strategy and enhanced DOI support"""
        return self.progressive_search.search_with_target(
            ner_entities, api, target_count, strategy_instance
        )
    
    def search_multiple_apis(self, ner_entities: Dict[str, List[str]], 
                           apis: List[str], target_count_per_api: int = 5,
                           strategy_instances: Dict[str, Any] = None) -> Dict[str, List[Dict]]:
        """Search multiple APIs with API-specific configurations"""
        results = {}
        
        for api in apis:
            if api not in APICapabilities.get_supported_apis():
                continue
            
            strategy_instance = strategy_instances.get(api) if strategy_instances else None
            
            candidates = self.search_single_api(
                ner_entities, api, target_count_per_api, strategy_instance
            )
            
            if candidates:
                results[api] = candidates
        
        return results
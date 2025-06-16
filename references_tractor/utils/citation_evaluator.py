import json
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime
import traceback

class CitationEvaluator:
    """
    Enhanced evaluation framework for citation linking pipeline with multiple DOI support
    Uses pipeline's DOI extraction instead of custom logic
    """
    
    def __init__(self, gold_standard_path: str, pipeline):
        """
        Initialize evaluator with gold standard and pipeline
        
        Args:
            gold_standard_path: Path to JSON file with gold standard data
            pipeline: Citation parser/linker instance (ReferencesTractor)
        """
        self.gold_standard_path = gold_standard_path
        self.pipeline = pipeline
        self.gold_standard = self.load_gold_standard()
        self.apis = ["openalex", "openaire", "pubmed", "crossref", "hal"]
        self.results = []
        
    def load_gold_standard(self) -> Dict[str, Dict[str, str]]:
        """Load gold standard from JSON file"""
        with open(self.gold_standard_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def normalize_doi(self, doi: Optional[str]) -> Optional[str]:
        """
        Normalize DOI for comparison by removing prefixes and standardizing format
        """
        if not doi:
            return None
        
        # Convert to string and strip whitespace
        doi = str(doi).strip()
        if not doi:
            return None
        
        # Remove common prefixes
        prefixes_to_remove = [
            "https://doi.org/",
            "http://doi.org/", 
            "https://dx.doi.org/",
            "http://dx.doi.org/",
            "doi:",
            "DOI:"
        ]
        
        doi_lower = doi.lower()
        for prefix in prefixes_to_remove:
            if doi_lower.startswith(prefix.lower()):
                doi = doi[len(prefix):]
                break
        
        # Remove trailing slashes and whitespace
        doi = doi.strip().rstrip('/')
        
        # Convert to lowercase for comparison
        return doi.lower() if doi else None

    def normalize_api_id(self, api_id: Optional[str], api: str) -> Optional[str]:
        """
        Normalize API ID by extracting the core identifier from URLs
        Handles both full URLs and bare IDs
        """
        if not api_id:
            return None
        
        api_id = str(api_id).strip()
        if not api_id:
            return None
        
        try:
            if api == "openalex":
                # Handle both "https://openalex.org/W123456" and "W123456"
                if "openalex.org/" in api_id:
                    return api_id.split("openalex.org/")[-1]
                else:
                    return api_id
                    
            elif api == "openaire":
                # OpenAIRE IDs are usually complex strings like "doi_dedup___::..."
                # Just compare as-is after cleaning
                return api_id
                
            elif api == "pubmed":
                # Handle both "https://pubmed.ncbi.nlm.nih.gov/12345" and "12345" 
                if "pubmed.ncbi.nlm.nih.gov/" in api_id:
                    return api_id.split("pubmed.ncbi.nlm.nih.gov/")[-1]
                else:
                    return api_id
                    
            elif api == "crossref":
                # CrossRef uses DOIs as IDs, so same as DOI normalization
                return self.normalize_doi(api_id)
                
            elif api == "hal":
                # Handle both "https://hal.science/hal-12345" and "hal-12345"
                if "hal.science/" in api_id:
                    return api_id.split("hal.science/")[-1]
                else:
                    return api_id
            
            # Default: return as-is
            return api_id
            
        except Exception:
            return api_id

    def check_api_id_match(self, result: Dict, expected_id: str, api: str) -> str:
        """
        Check if API-specific ID matches expected ID
        Returns: EXACT, INCORRECT, or N/A
        """
        if not expected_id:
            return "N/A"
        
        normalized_expected = self.normalize_api_id(expected_id, api)
        if not normalized_expected:
            return "N/A"
        
        # Handle different result types
        if isinstance(result, str):
            return "N/A"  # Can't extract ID from raw strings
        
        if not isinstance(result, dict):
            return "N/A"
        
        # Extract API-specific ID from result
        retrieved_id = None
        if api == "openalex":
            retrieved_id = result.get('id') or result.get('openalex_id')
        elif api == "openaire":
            retrieved_id = result.get('id') or result.get('openaire_id')
        elif api == "pubmed":
            retrieved_id = result.get('id') or result.get('pmid') or result.get('pubmed_id')
        elif api == "crossref":
            retrieved_id = result.get('id') or result.get('crossref_id') or result.get('DOI')
        elif api == "hal":
            retrieved_id = result.get('id') or result.get('hal_id') or result.get('halId_s')
        
        if not retrieved_id:
            return "N/A"
        
        normalized_retrieved = self.normalize_api_id(retrieved_id, api)
        if not normalized_retrieved:
            return "N/A"
        
        # Compare normalized IDs
        if normalized_retrieved == normalized_expected:
            return "EXACT"
        else:
            return "INCORRECT"

    def check_doi_match_enhanced(self, result: Dict, expected_doi: str) -> str:
        """
        Enhanced DOI matching using pipeline's multiple DOI support
        Returns: EXACT, INCORRECT, or N/A
        Updated to handle PubMed dict structure
        """
        if not expected_doi:
            return "N/A"
        
        normalized_expected = self.normalize_doi(expected_doi)
        if not normalized_expected:
            return "N/A"
        
        # Handle different result types
        if isinstance(result, str):
            # Legacy XML string - try to parse DOI
            if "pubmed" in str(result).lower():
                try:
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(result)
                    doi_elem = root.find(".//ELocationID[@EIdType='doi']")
                    if doi_elem is not None:
                        normalized_found = self.normalize_doi(doi_elem.text)
                        return "EXACT" if normalized_found == normalized_expected else "INCORRECT"
                except Exception:
                    pass
            return "N/A"
        
        if not isinstance(result, dict):
            return "N/A"
        
        # Check main DOI
        main_doi = result.get('main_doi') or result.get('doi')
        if main_doi:
            normalized_main = self.normalize_doi(main_doi)
            if normalized_main == normalized_expected:
                return "EXACT"
        
        # Check alternative DOIs
        alternative_dois = result.get('alternative_dois', [])
        for alt_doi in alternative_dois:
            normalized_alt = self.normalize_doi(alt_doi)
            if normalized_alt == normalized_expected:
                return "EXACT"
        
        # Check all_dois if available
        all_dois = result.get('all_dois', [])
        for doi in all_dois:
            normalized_doi = self.normalize_doi(doi)
            if normalized_doi == normalized_expected:
                return "EXACT"
        
        # If we have any DOIs but none match
        if main_doi or alternative_dois or all_dois:
            return "INCORRECT"
        
        return "N/A"

    def check_enhanced_match(self, result: Dict, expected_gold: Dict, api: str) -> Dict[str, str]:
        """
        Enhanced matching using both DOI and API-specific ID
        Returns: {
            'doi_match': 'EXACT'|'INCORRECT'|'N/A',
            'id_match': 'EXACT'|'INCORRECT'|'N/A', 
            'overall_match': 'EXACT'|'INCORRECT'|'N/A'
        }
        """
        # Check DOI match
        expected_doi = expected_gold.get('doi')
        doi_match = self.check_doi_match_enhanced(result, expected_doi)
        
        # Check API-specific ID match
        expected_id = expected_gold.get(api)  # e.g., expected_gold.get('openalex')
        id_match = self.check_api_id_match(result, expected_id, api)
        
        # Determine overall match
        overall_match = "N/A"
        if doi_match == "EXACT" or id_match == "EXACT":
            overall_match = "EXACT"
        elif doi_match == "INCORRECT" or id_match == "INCORRECT":
            overall_match = "INCORRECT"
        
        return {
            'doi_match': doi_match,
            'id_match': id_match,
            'overall_match': overall_match
        }

    def extract_metadata_for_comparison(self, api_result: Dict, api: str) -> Dict[str, Any]:
        """
        Extract comparable metadata from API result
        Updated with comprehensive null-safe handling
        """
        metadata = {}
        
        try:
            if api == "openalex":
                # NULL SAFE OpenAlex metadata extraction
                authorships = api_result.get('authorships') or []
                authors = []
                if isinstance(authorships, list):
                    for auth in authorships:
                        if isinstance(auth, dict):
                            raw_name = auth.get('raw_author_name', '')
                            if raw_name:
                                authors.append(raw_name)
                
                # NULL SAFE primary_location access
                primary_location = api_result.get('primary_location')
                journal = ''
                if isinstance(primary_location, dict):
                    source = primary_location.get('source')
                    if isinstance(source, dict):
                        journal = source.get('display_name', '')
                
                metadata = {
                    'title': api_result.get('title', ''),
                    'year': api_result.get('publication_year'),
                    'authors': authors,
                    'journal': journal
                }
                
            elif api == "openaire":
                # NULL SAFE OpenAIRE metadata extraction
                authors_data = api_result.get('authors') or []
                authors = []
                if isinstance(authors_data, list):
                    for author in authors_data:
                        if isinstance(author, dict):
                            full_name = author.get('fullName', '')
                            if full_name:
                                authors.append(full_name)
                
                # NULL SAFE container access
                container = api_result.get('container')
                journal = ''
                if isinstance(container, dict):
                    journal = container.get('name', '')
                
                publication_date = api_result.get('publicationDate', '')
                year = publication_date[:4] if publication_date else ''
                
                metadata = {
                    'title': api_result.get('mainTitle', ''),
                    'year': year,
                    'authors': authors,
                    'journal': journal
                }
                
            elif api == "crossref":
                # NULL SAFE CrossRef metadata extraction
                author_data = api_result.get('author') or []
                authors = []
                if isinstance(author_data, list):
                    for auth in author_data:
                        if isinstance(auth, dict):
                            given = auth.get('given', '')
                            family = auth.get('family', '')
                            full_name = f"{given} {family}".strip()
                            if full_name:
                                authors.append(full_name)
                
                # NULL SAFE issued date access
                issued = api_result.get('issued')
                year = ''
                if isinstance(issued, dict):
                    date_parts = issued.get('date-parts')
                    if isinstance(date_parts, list) and len(date_parts) > 0:
                        if isinstance(date_parts[0], list) and len(date_parts[0]) > 0:
                            year_val = date_parts[0][0]
                            if year_val:
                                year = str(year_val)
                
                # NULL SAFE container-title access
                container_title = api_result.get('container-title')
                journal = ''
                if isinstance(container_title, list) and len(container_title) > 0:
                    journal = container_title[0] or ''
                
                # NULL SAFE title access
                title_data = api_result.get('title')
                title = ''
                if isinstance(title_data, list) and len(title_data) > 0:
                    title = title_data[0] or ''
                
                metadata = {
                    'title': title,
                    'year': year,
                    'authors': authors,
                    'journal': journal
                }
                
            elif api == "hal":
                # NULL SAFE HAL metadata extraction
                title_data = api_result.get('title_s')
                title = ''
                if isinstance(title_data, list) and len(title_data) > 0:
                    title = title_data[0] or ''
                
                # Handle both list and string for authFullName_s
                auth_data = api_result.get('authFullName_s', [])
                authors = []
                if isinstance(auth_data, list):
                    authors = [str(auth) for auth in auth_data if auth]
                elif auth_data:  # Single string
                    authors = [str(auth_data)]
                
                metadata = {
                    'title': title,
                    'year': str(api_result.get('publicationDateY_s', '')),
                    'authors': authors,
                    'journal': api_result.get('journalTitle_s', '')
                }
                
            elif api == "pubmed":
                # NULL SAFE PubMed metadata extraction
                if isinstance(api_result, dict):
                    # Extract from parsed structure
                    authors_data = api_result.get('authors') or []
                    authors = []
                    if isinstance(authors_data, list):
                        for author in authors_data:
                            if isinstance(author, dict):
                                full_name = author.get('full_name', '')
                                if full_name:
                                    authors.append(full_name)
                            elif author:  # String author
                                authors.append(str(author))
                    
                    metadata = {
                        'title': api_result.get('title', ''),
                        'year': api_result.get('publication_year', ''),
                        'authors': authors,
                        'journal': api_result.get('journal', '')
                    }
                    
                    # Fallback to XML parsing if no parsed data but xml_content exists
                    if not any(metadata.values()):
                        xml_content = api_result.get('xml_content')
                        if xml_content:
                            try:
                                import xml.etree.ElementTree as ET
                                root = ET.fromstring(xml_content)
                                article = root.find("PubmedArticle/MedlineCitation/Article")
                                
                                if article is not None:
                                    # Extract title
                                    title = article.findtext("ArticleTitle", "")
                                    
                                    # Extract year
                                    pub_date = article.find("Journal/JournalIssue/PubDate")
                                    year = pub_date.findtext("Year", "") if pub_date is not None else ""
                                    
                                    # Extract authors
                                    authors = []
                                    author_list = article.find("AuthorList")
                                    if author_list is not None:
                                        for author in author_list.findall("Author"):
                                            last_name = author.findtext("LastName", "")
                                            initials = author.findtext("Initials", "")
                                            if last_name:
                                                authors.append(f"{last_name}, {initials}")
                                    
                                    # Extract journal
                                    journal = article.findtext("Journal/Title", "")
                                    
                                    metadata = {
                                        'title': title,
                                        'year': year,
                                        'authors': authors,
                                        'journal': journal
                                    }
                            except Exception as e:
                                print(f"Error parsing PubMed XML for metadata: {e}")
                                metadata = {'title': '', 'year': '', 'authors': [], 'journal': ''}
                else:
                    metadata = {'title': '', 'year': '', 'authors': [], 'journal': ''}
            else:
                # Unknown API
                metadata = {'title': '', 'year': '', 'authors': [], 'journal': ''}
                
        except Exception as e:
            print(f"Error extracting metadata for {api}: {e}")
            metadata = {'title': '', 'year': '', 'authors': [], 'journal': ''}
        
        return metadata
    
    def compare_metadata(self, extracted_metadata: Dict, gold_citation: str) -> str:
        """
        Compare extracted metadata with gold standard citation to determine if likely same paper
        Returns: LIKELY_SAME, LIKELY_DIFFERENT, or N/A
        """
        try:
            title = extracted_metadata.get('title', '').lower().strip()
            year = str(extracted_metadata.get('year', '')).strip()
            
            citation_lower = gold_citation.lower()
            
            # Check if title words appear in citation (basic similarity)
            if title and len(title) > 10:
                title_words = set(title.split())
                # Remove common words
                title_words = title_words - {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
                if len(title_words) >= 3:
                    matches = sum(1 for word in title_words if len(word) > 3 and word in citation_lower)
                    if matches >= max(2, len(title_words) * 0.3):  # At least 30% of significant words match
                        return "LIKELY_SAME"
            
            # Check year match
            if year and len(year) == 4 and year in gold_citation:
                return "LIKELY_SAME"
            
            return "LIKELY_DIFFERENT"
            
        except Exception:
            return "N/A"
    
    def evaluate_single_citation(self, citation: str, expected_results: Dict[str, str]) -> Dict[str, Any]:
        """
        Updated to use enhanced matching with disaggregated DOI/ID results
        """
        citation_result = {
            'citation_id': f"Cit_{len(self.results)+1:03d}",
            'original_citation': citation,
            'gold_standard': expected_results,
            'api_results': {},
            'ensemble': {}
        }
        
        # Test each API individually with enhanced matching
        for api in self.apis:
            try:
                # Get enhanced result with multiple DOI support
                result = self.pipeline.link_citation(citation, api_target=api, output='advanced')
                
                if result and ('result' in result or isinstance(result, str)):
                    # Handle different result types (skip raw XML strings for PubMed)
                    if api == "pubmed" and isinstance(result, str):
                        citation_result['api_results'][api] = {
                            'status': 'NO_RESULT',
                            'score': 'N/A',
                            'doi_match': 'N/A',
                            'id_match': 'N/A', 
                            'overall_match': 'N/A',
                            'metadata_match': 'N/A',
                            'retrieved_data': None,
                            'note': 'PubMed returned XML string - not processed'
                        }
                        continue
                    
                    if not isinstance(result, dict):
                        citation_result['api_results'][api] = {
                            'status': 'ERROR',
                            'score': 'N/A',
                            'doi_match': 'N/A',
                            'id_match': 'N/A',
                            'overall_match': 'N/A', 
                            'metadata_match': 'N/A',
                            'retrieved_data': None,
                            'error': 'Unexpected result format'
                        }
                        continue
                    
                    # Extract metadata if available
                    metadata = {}
                    if 'full-publication' in result:
                        metadata = self.extract_metadata_for_comparison(result['full-publication'], api)
                    
                    # Enhanced matching using both DOI and ID
                    match_results = self.check_enhanced_match(result, expected_results, api)
                    
                    # Determine metadata match (only if overall doesn't match exactly)
                    metadata_match = "N/A"
                    if match_results['overall_match'] == "INCORRECT" and metadata:
                        metadata_match = self.compare_metadata(metadata, citation)
                    
                    # Store structured API result with disaggregated matching
                    citation_result['api_results'][api] = {
                        'status': 'SUCCESS',
                        'score': result.get('score', 'N/A'),
                        'doi_match': match_results['doi_match'],
                        'id_match': match_results['id_match'],
                        'overall_match': match_results['overall_match'],
                        'metadata_match': metadata_match,
                        'retrieved_data': {
                            'id': result.get(f'{api}_id') or result.get('id'),
                            'main_doi': result.get('main_doi') or result.get('doi'),
                            'alternative_dois': result.get('alternative_dois', []),
                            'total_dois': result.get('total_dois', 0),
                            'all_dois': result.get('all_dois', []),
                            'formatted_citation': result.get('result', ''),
                            'metadata': metadata
                        }
                    }
                else:
                    citation_result['api_results'][api] = {
                        'status': 'NO_RESULT',
                        'score': 'N/A',
                        'doi_match': 'N/A',
                        'id_match': 'N/A',
                        'overall_match': 'N/A',
                        'metadata_match': 'N/A',
                        'retrieved_data': None
                    }
                    
            except Exception as e:
                print(f"Error processing {api} for citation: {str(e)}")
                citation_result['api_results'][api] = {
                    'status': 'ERROR',
                    'score': 'N/A',
                    'doi_match': 'N/A',
                    'id_match': 'N/A',
                    'overall_match': 'N/A',
                    'metadata_match': 'N/A',
                    'retrieved_data': None,
                    'error': str(e)
                }
        
        # Test enhanced ensemble approach
        try:
            ensemble_result = self.pipeline.link_citation_ensemble(citation, api_targets=self.apis, output='advanced')
            
            if ensemble_result and ensemble_result.get('doi'):
                ensemble_metadata = ensemble_result.get('ensemble_metadata', {})
                contributing_apis = ensemble_metadata.get('contributing_apis', [])
                
                # Enhanced ensemble matching - check against all possible gold standard values
                ensemble_match_results = self.check_enhanced_match(ensemble_result, expected_results, 'ensemble')
                
                citation_result['ensemble'] = {
                    'status': 'SUCCESS',
                    'consensus_doi': ensemble_result['doi'],
                    'vote_count': ensemble_metadata.get('selected_doi_votes', 0),
                    'total_dois_found': ensemble_metadata.get('total_dois_found', 0),
                    'contributing_apis': contributing_apis,
                    'doi_match': ensemble_match_results['doi_match'],
                    'id_match': 'N/A',  # Ensemble doesn't have specific API ID
                    'overall_match': ensemble_match_results['overall_match'],
                    'metadata_match': 'N/A',
                    'external_ids': ensemble_result.get('external_ids', {}),
                    'doi_vote_breakdown': ensemble_metadata.get('doi_vote_breakdown', {})
                }
            else:
                citation_result['ensemble'] = {
                    'status': 'NO_RESULT',
                    'consensus_doi': None,
                    'vote_count': 0,
                    'total_dois_found': 0,
                    'contributing_apis': [],
                    'doi_match': 'N/A',
                    'id_match': 'N/A',
                    'overall_match': 'N/A',
                    'metadata_match': 'N/A'
                }
                
        except Exception as e:
            print(f"Error processing ensemble for citation: {str(e)}")
            citation_result['ensemble'] = {
                'status': 'ERROR',
                'consensus_doi': None,
                'vote_count': 0,
                'total_dois_found': 0,
                'contributing_apis': [],
                'doi_match': 'N/A',
                'id_match': 'N/A', 
                'overall_match': 'N/A',
                'metadata_match': 'N/A',
                'error': str(e)
            }
        
        return citation_result
    
    def run_evaluation(self, limit: Optional[int] = None) -> None:
        """
        Run evaluation on all citations (or limited subset) using enhanced pipeline
        """
        print(f"Starting enhanced evaluation of {len(self.gold_standard)} citations...")
        
        citations_to_process = list(self.gold_standard.items())
        if limit:
            citations_to_process = citations_to_process[:limit]
            print(f"Limited to first {limit} citations for testing")
        
        for i, (citation, expected_results) in enumerate(citations_to_process):
            print(f"Processing citation {i+1}/{len(citations_to_process)}: {citation[:80]}...")
            
            try:
                result = self.evaluate_single_citation(citation, expected_results)
                self.results.append(result)
            except Exception as e:
                print(f"Failed to process citation {i+1}: {str(e)}")
                traceback.print_exc()
                # Add failed result with enhanced structure
                failed_result = {
                    'citation': citation[:100] + "..." if len(citation) > 100 else citation,
                    'citation_id': f"Cit_{len(self.results)+1:03d}",
                    'original_citation': citation
                }
                # Fill with error status for all APIs (enhanced structure)
                for api in self.apis:
                    api_key = api.upper()
                    failed_result.update({
                        f'{api_key}_ID': None,
                        f'{api_key}_MAIN_DOI': None,
                        f'{api_key}_ALT_DOIS': [],
                        f'{api_key}_TOTAL_DOIS': 0,
                        f'{api_key}_SCORE': "N/A",
                        f'{api_key}_STATUS': "ERROR",
                        f'{api_key}_DOI_MATCH': "N/A",
                        f'{api_key}_METADATA_MATCH': "N/A",
                        f'{api_key}_FORMATTED_CITATION': ""
                    })
                failed_result.update({
                    'ENSEMBLE_DOI': None,
                    'ENSEMBLE_IDS': None,
                    'ENSEMBLE_VOTES': 0,
                    'ENSEMBLE_TOTAL_DOIS': 0,
                    'ENSEMBLE_STATUS': "ERROR",
                    'ENSEMBLE_DOI_MATCH': "N/A",
                    'ENSEMBLE_METADATA_MATCH': "N/A"
                })
                self.results.append(failed_result)
        
        print(f"Enhanced evaluation completed! Processed {len(self.results)} citations.")
    
    def calculate_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate summary metrics with enhanced DOI/ID matching
        Includes disaggregated DOI-only and ID-only match counts
        """
        metrics = {}
        approaches = self.apis + ['ensemble']
        
        for approach in approaches:
            total_citations = len(self.results)
            results_found = 0
            
            # Enhanced matching counters
            overall_exact_match = 0
            doi_exact_match = 0
            id_exact_match = 0
            overall_incorrect = 0
            metadata_likely_same = 0
            
            errors = 0
            no_results = 0
            
            # Enhanced DOI statistics
            total_dois_found = 0
            results_with_multiple_dois = 0
            
            for result in self.results:
                if approach == 'ensemble':
                    # Ensemble results
                    ensemble_data = result.get('ensemble', {})
                    status = ensemble_data.get('status', 'ERROR')
                    overall_match = ensemble_data.get('overall_match', 'N/A')
                    doi_match = ensemble_data.get('doi_match', 'N/A')
                    id_match = ensemble_data.get('id_match', 'N/A')
                    metadata_match = ensemble_data.get('metadata_match', 'N/A')
                    total_dois = ensemble_data.get('total_dois_found', 0)
                else:
                    # API results
                    api_data = result.get('api_results', {}).get(approach, {})
                    status = api_data.get('status', 'ERROR')
                    overall_match = api_data.get('overall_match', 'N/A')
                    doi_match = api_data.get('doi_match', 'N/A')
                    id_match = api_data.get('id_match', 'N/A')
                    metadata_match = api_data.get('metadata_match', 'N/A')
                    retrieved_data = api_data.get('retrieved_data')
                    total_dois = retrieved_data.get('total_dois', 0) if retrieved_data else 0
                
                total_dois_found += total_dois
                if total_dois > 1:
                    results_with_multiple_dois += 1
                
                if status == "SUCCESS":
                    results_found += 1
                    
                    # Count different types of matches
                    if overall_match == "EXACT":
                        overall_exact_match += 1
                    elif overall_match == "INCORRECT":
                        overall_incorrect += 1
                        if metadata_match == "LIKELY_SAME":
                            metadata_likely_same += 1
                    
                    # Count disaggregated matches
                    if doi_match == "EXACT":
                        doi_exact_match += 1
                    if id_match == "EXACT":
                        id_exact_match += 1
                        
                elif status == "NO_RESULT":
                    no_results += 1
                else:  # ERROR
                    errors += 1
            
            # Calculate rates
            success_rate = (results_found / total_citations * 100) if total_citations > 0 else 0
            overall_exact_rate = (overall_exact_match / total_citations * 100) if total_citations > 0 else 0
            doi_exact_rate = (doi_exact_match / total_citations * 100) if total_citations > 0 else 0
            id_exact_rate = (id_exact_match / total_citations * 100) if total_citations > 0 else 0
            metadata_likely_rate = (metadata_likely_same / total_citations * 100) if total_citations > 0 else 0
            
            # Strict metrics (exact matches only)
            strict_correct = overall_exact_match
            strict_correct_rate = overall_exact_rate
            
            # Inclusive metrics (exact + likely metadata matches)
            inclusive_correct = overall_exact_match + metadata_likely_same
            inclusive_correct_rate = (inclusive_correct / total_citations * 100) if total_citations > 0 else 0
            
            metrics[approach] = {
                'Total_Citations': total_citations,
                'Results_Found': results_found,
                'Success_Rate': f"{success_rate:.1f}%",
                
                # Enhanced matching breakdown
                'Overall_Exact_Match': overall_exact_match,
                'Overall_Exact_Rate': f"{overall_exact_rate:.1f}%",
                'DOI_Exact_Match': doi_exact_match,
                'DOI_Exact_Rate': f"{doi_exact_rate:.1f}%",
                'ID_Exact_Match': id_exact_match,
                'ID_Exact_Rate': f"{id_exact_rate:.1f}%",
                'Overall_Incorrect': overall_incorrect,
                'Metadata_Likely_Same': metadata_likely_same,
                'Metadata_Likely_Rate': f"{metadata_likely_rate:.1f}%",
                
                # Strict metrics (exact matches only)
                'Strict_Correct': strict_correct,
                'Strict_Correct_Rate': f"{strict_correct_rate:.1f}%",
                
                # Inclusive metrics (exact + likely matches)
                'Inclusive_Correct': inclusive_correct,
                'Inclusive_Correct_Rate': f"{inclusive_correct_rate:.1f}%",
                
                # Other metrics
                'No_Results': no_results,
                'Errors': errors,
                'Total_DOIs_Found': total_dois_found,
                'Multiple_DOI_Results': results_with_multiple_dois
            }
        
        return metrics
    
    def generate_summary_dashboard(self) -> str:
        """
        Generate Level 1: Enhanced Summary Dashboard with reordered metrics display
        """
        metrics = self.calculate_metrics()
        
        output = []
        output.append("="*80)
        output.append("ENHANCED CITATION LINKING EVALUATION - SUMMARY DASHBOARD")
        output.append("="*80)
        output.append(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append(f"Total Citations Evaluated: {len(self.results)}")
        output.append(f"APIs Tested: {', '.join(self.apis)}")
        output.append("Enhanced Features: Multiple DOI Support, DOI+ID Matching")
        output.append("")
        
        # Create summary table with reordered columns for better readability
        df_metrics = pd.DataFrame(metrics).T
        
        # Reorder columns for display (same order as TSV)
        display_columns = [
            'Total_Citations', 'Overall_Exact_Match', 'Overall_Exact_Rate',
            'DOI_Exact_Match', 'DOI_Exact_Rate', 'ID_Exact_Match', 'ID_Exact_Rate',
            'Metadata_Likely_Same', 'Inclusive_Correct_Rate', 'Results_Found', 
            'No_Results', 'Errors'
        ]
        
        # Select columns that exist
        available_columns = [col for col in display_columns if col in df_metrics.columns]
        df_display = df_metrics[available_columns]
        
        output.append("OVERALL PERFORMANCE METRICS (Key Metrics First):")
        output.append("-" * 55)
        output.append(df_display.to_string())
        output.append("")
        
        # Enhanced insights with reordered focus
        output.append("KEY INSIGHTS:")
        output.append("-" * 15)
        
        # Lead with overall performance (most important)
        best_api_overall = max(self.apis, key=lambda api: metrics[api]['Overall_Exact_Match'])
        output.append(f"• Best Overall Performance: {best_api_overall.title()} ({metrics[best_api_overall]['Overall_Exact_Rate']} exact matches)")
        
        # Break down by match type
        best_api_doi = max(self.apis, key=lambda api: metrics[api]['DOI_Exact_Match'])
        best_api_id = max(self.apis, key=lambda api: metrics[api]['ID_Exact_Match'])
        output.append(f"• Best DOI Matching: {best_api_doi.title()} ({metrics[best_api_doi]['DOI_Exact_Rate']})")
        output.append(f"• Best ID Matching: {best_api_id.title()} ({metrics[best_api_id]['ID_Exact_Rate']})")
        
        # Ensemble performance
        ensemble_overall = metrics['ensemble']['Overall_Exact_Match']
        ensemble_overall_rate = metrics['ensemble']['Overall_Exact_Rate']
        output.append(f"• Ensemble Performance: {ensemble_overall} exact matches ({ensemble_overall_rate})")
        
        # Secondary metrics
        total_likely_matches = sum(metrics[api]['Metadata_Likely_Same'] for api in self.apis)
        output.append(f"• Additional Likely Matches: {total_likely_matches} (based on metadata similarity)")
        
        # Coverage and reliability
        best_coverage_api = max(self.apis, key=lambda api: metrics[api]['Results_Found'])
        output.append(f"• Best Coverage: {best_coverage_api.title()} ({metrics[best_coverage_api]['Results_Found']}/{len(self.results)} citations found)")
        
        # Multiple DOI insights
        total_multiple_dois = sum(metrics[api]['Multiple_DOI_Results'] for api in self.apis)
        total_dois_found = sum(metrics[api]['Total_DOIs_Found'] for api in self.apis)
        output.append(f"• DOI Diversity: {total_multiple_dois} results with multiple DOIs, {total_dois_found} total DOIs found")
        
        return "\n".join(output)
    
    def generate_comparison_table(self) -> str:
        """
        Generate Level 2: Enhanced Comparison Table with multiple DOI columns (no formatted citations)
        """
        if not self.results:
            return "No results available for comparison table."
        
        # Create flattened data for DataFrame
        table_data = []
        for result in self.results:
            row = {
                'citation_id': result['citation_id'],
                'citation': result['original_citation'][:100] + "..." if len(result['original_citation']) > 100 else result['original_citation']
            }
            
            # Add API columns (without formatted citations)
            for api in self.apis:
                api_key = api.upper()
                api_data = result.get('api_results', {}).get(api, {})
                retrieved_data = api_data.get('retrieved_data')
                
                row.update({
                    f'{api_key}_ID': retrieved_data.get('id') if retrieved_data else None,
                    f'{api_key}_MAIN_DOI': retrieved_data.get('main_doi') if retrieved_data else None,
                    f'{api_key}_ALT_DOIS': retrieved_data.get('alternative_dois') if retrieved_data else [],
                    f'{api_key}_TOTAL_DOIS': retrieved_data.get('total_dois') if retrieved_data else 0,
                    f'{api_key}_SCORE': api_data.get('score', 'N/A'),
                    f'{api_key}_STATUS': api_data.get('status', 'ERROR'),
                    f'{api_key}_DOI_MATCH': api_data.get('doi_match', 'N/A'),
                    f'{api_key}_METADATA_MATCH': api_data.get('metadata_match', 'N/A')
                })
            
            # Add ensemble columns (without formatted citations)
            ensemble_data = result.get('ensemble', {})
            row.update({
                'ENSEMBLE_DOI': ensemble_data.get('consensus_doi'),
                'ENSEMBLE_APIS': ', '.join(ensemble_data.get('contributing_apis', [])),
                'ENSEMBLE_VOTES': ensemble_data.get('vote_count', 0),
                'ENSEMBLE_TOTAL_DOIS': ensemble_data.get('total_dois_found', 0),
                'ENSEMBLE_STATUS': ensemble_data.get('status', 'ERROR'),
                'ENSEMBLE_DOI_MATCH': ensemble_data.get('doi_match', 'N/A'),
                'ENSEMBLE_METADATA_MATCH': ensemble_data.get('metadata_match', 'N/A')
            })
            
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        output = []
        output.append("="*150)
        output.append("ENHANCED CITATION LINKING EVALUATION - COMPARISON TABLE") 
        output.append("="*150)
        output.append("All approaches compared side-by-side with multiple DOI support")
        output.append("")
        output.append(df.to_string(index=False, max_colwidth=15))
        
        return "\n".join(output)
    
    def enhance_results_with_metadata(self) -> List[Dict[str, Any]]:
        """
        Enhanced metadata inclusion with formatted citations and multiple DOI support
        Now returns the clean, structured format without duplication
        """
        # The results are already enhanced with the new structure
        # Just return them as-is since they now contain all needed information
        return self.results
    
    def save_results(self, output_dir: str = "evaluation_results"):
        """
        Save all evaluation results to files (TSV for tables, TXT for summary, JSON for raw data)
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save Level 0: Summary Dashboard (text only)
        summary = self.generate_summary_dashboard()
        with open(f"{output_dir}/summary_dashboard_{timestamp}.txt", "w", encoding="utf-8") as f:
            f.write(summary)

        # Save Level 1: Performance Metrics Table (TSV for spreadsheet analysis)
        self.save_performance_metrics_tsv(f"{output_dir}/01_overall_performance_metrics_{timestamp}.tsv")

        # Save Level 2: Detailed Comparison Table (TSV for spreadsheet analysis)
        self.save_comparison_table_tsv(f"{output_dir}/02_detailed_comparison_table_{timestamp}.tsv")
        
        # Save Level 3: Individual API detailed tables (TSV)
        approaches = self.apis + ['ensemble']
        for approach in approaches:
            filename_tsv = f"{output_dir}/03_{approach}_detailed_results_{timestamp}.tsv"
            self.save_individual_table_tsv(approach, filename_tsv)
        
        # Save enhanced raw results as JSON with metadata and formatted citations
        enhanced_results = self.enhance_results_with_metadata()
        with open(f"{output_dir}/full_results_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(enhanced_results, f, indent=2, ensure_ascii=False)
        
        print(f"Enhanced evaluation results saved to {output_dir}/ directory")
        print("Tables saved as TSV files for spreadsheet analysis")
        print("Enhanced JSON includes formatted citations and multiple DOI tracking")
        return output_dir

    def save_performance_metrics_tsv(self, filename: str):
        """
        Save overall performance metrics as TSV file with most significant metrics first
        """
        if not self.results:
            return
        
        metrics = self.calculate_metrics()
        
        # Convert metrics dict to DataFrame format
        metrics_data = []
        for api_name, api_metrics in metrics.items():
            row = {'API': api_name.upper()}
            row.update(api_metrics)
            metrics_data.append(row)
        
        df = pd.DataFrame(metrics_data)
        
        # Reorder columns - MOST SIGNIFICANT FIRST
        column_order = [
            # Basic info
            'API', 
            'Total_Citations',
            
            # KEY PERFORMANCE METRICS (most important)
            'Overall_Exact_Match',           # Main success metric
            'Overall_Exact_Rate',            # Main success rate
            'Strict_Correct',                # Same as Overall_Exact_Match (for clarity)
            'Strict_Correct_Rate',           # Same as Overall_Exact_Rate (for clarity)
            
            # DETAILED MATCHING BREAKDOWN
            'DOI_Exact_Match',               # DOI-only matches
            'DOI_Exact_Rate',                # DOI-only rate
            'ID_Exact_Match',                # ID-only matches  
            'ID_Exact_Rate',                 # ID-only rate
            
            # SECONDARY METRICS
            'Metadata_Likely_Same',          # Likely matches (metadata-based)
            'Metadata_Likely_Rate',          # Likely match rate
            'Inclusive_Correct',             # Exact + Likely matches
            'Inclusive_Correct_Rate',        # Inclusive success rate
            
            # OPERATIONAL METRICS
            'Results_Found',                 # How many results retrieved
            'Success_Rate',                  # Retrieval success rate
            'Overall_Incorrect',             # Wrong matches
            'No_Results',                    # Failed to find anything
            'Errors',                        # Technical errors
            
            # DOI STATISTICS (least critical)
            'Total_DOIs_Found',              # Total DOIs across all results
            'Multiple_DOI_Results'           # Results with multiple DOIs
        ]
        
        # Reorder columns (keep existing ones if column exists)
        existing_columns = [col for col in column_order if col in df.columns]
        remaining_columns = [col for col in df.columns if col not in column_order]
        final_columns = existing_columns + remaining_columns
        
        df = df[final_columns]
        
        # Save as TSV
        df.to_csv(filename, sep='\t', index=False, encoding='utf-8')

    def save_comparison_table_tsv(self, filename: str):
        """
        Save enhanced comparison table with disaggregated DOI/ID matching columns
        """
        if not self.results:
            return
        
        # Create flattened data for DataFrame with enhanced matching columns
        table_data = []
        for result in self.results:
            row = {
                'citation_id': result['citation_id'],
                'citation': result['original_citation']
            }
            
            # Add API columns with disaggregated matching
            for api in self.apis:
                api_key = api.upper()
                api_data = result.get('api_results', {}).get(api, {})
                retrieved_data = api_data.get('retrieved_data')
                
                row.update({
                    f'{api_key}_ID': retrieved_data.get('id') if retrieved_data else None,
                    f'{api_key}_MAIN_DOI': retrieved_data.get('main_doi') if retrieved_data else None,
                    f'{api_key}_ALT_DOIS': str(retrieved_data.get('alternative_dois', [])) if retrieved_data else "[]",
                    f'{api_key}_TOTAL_DOIS': retrieved_data.get('total_dois') if retrieved_data else 0,
                    f'{api_key}_SCORE': api_data.get('score', 'N/A'),
                    f'{api_key}_STATUS': api_data.get('status', 'ERROR'),
                    # Enhanced matching columns
                    f'{api_key}_DOI_MATCH': api_data.get('doi_match', 'N/A'),
                    f'{api_key}_ID_MATCH': api_data.get('id_match', 'N/A'),
                    f'{api_key}_OVERALL_MATCH': api_data.get('overall_match', 'N/A'),
                    f'{api_key}_METADATA_MATCH': api_data.get('metadata_match', 'N/A')
                })
            
            # Add ensemble columns with enhanced matching
            ensemble_data = result.get('ensemble', {})
            row.update({
                'ENSEMBLE_DOI': ensemble_data.get('consensus_doi'),
                'ENSEMBLE_APIS': ', '.join(ensemble_data.get('contributing_apis', [])),
                'ENSEMBLE_VOTES': ensemble_data.get('vote_count', 0),
                'ENSEMBLE_TOTAL_DOIS': ensemble_data.get('total_dois_found', 0),
                'ENSEMBLE_STATUS': ensemble_data.get('status', 'ERROR'),
                # Enhanced matching columns for ensemble
                'ENSEMBLE_DOI_MATCH': ensemble_data.get('doi_match', 'N/A'),
                'ENSEMBLE_ID_MATCH': ensemble_data.get('id_match', 'N/A'),
                'ENSEMBLE_OVERALL_MATCH': ensemble_data.get('overall_match', 'N/A'),
                'ENSEMBLE_METADATA_MATCH': ensemble_data.get('metadata_match', 'N/A')
            })
            
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        # Save as TSV
        df.to_csv(filename, sep='\t', index=False, encoding='utf-8')
        
    def save_individual_table_tsv(self, approach: str, filename: str):
        """
        Save individual approach table with enhanced matching columns
        """
        if approach == 'ensemble':
            # Enhanced ensemble table
            ensemble_data = []
            for result in self.results:
                ensemble_info = result.get('ensemble', {})
                ensemble_data.append({
                    'Citation_ID': result['citation_id'],
                    'Citation': result['original_citation'],
                    'Consensus_DOI': ensemble_info.get('consensus_doi'),
                    'Contributing_APIs': ', '.join(ensemble_info.get('contributing_apis', [])),
                    'Vote_Count': ensemble_info.get('vote_count', 0),
                    'Total_DOIs': ensemble_info.get('total_dois_found', 0),
                    'Status': ensemble_info.get('status', 'ERROR'),
                    # Enhanced matching columns
                    'DOI_Match': ensemble_info.get('doi_match', 'N/A'),
                    'ID_Match': ensemble_info.get('id_match', 'N/A'),
                    'Overall_Match': ensemble_info.get('overall_match', 'N/A'),
                    'Metadata_Match': ensemble_info.get('metadata_match', 'N/A')
                })
            
            df = pd.DataFrame(ensemble_data)
        else:
            # Enhanced individual API table
            api_data = []
            for result in self.results:
                api_info = result.get('api_results', {}).get(approach, {})
                retrieved_data = api_info.get('retrieved_data')
                
                api_data.append({
                    'Citation_ID': result['citation_id'],
                    'Citation': result['original_citation'],
                    'Retrieved_ID': retrieved_data.get('id') if retrieved_data else None,
                    'Main_DOI': retrieved_data.get('main_doi') if retrieved_data else None,
                    'Alternative_DOIs': str(retrieved_data.get('alternative_dois', [])) if retrieved_data else "[]",
                    'Total_DOIs': retrieved_data.get('total_dois', 0) if retrieved_data else 0,
                    'Score': api_info.get('score', 'N/A'),
                    'Status': api_info.get('status', 'ERROR'),
                    # Enhanced matching columns
                    'DOI_Match': api_info.get('doi_match', 'N/A'),
                    'ID_Match': api_info.get('id_match', 'N/A'),
                    'Overall_Match': api_info.get('overall_match', 'N/A'),
                    'Metadata_Match': api_info.get('metadata_match', 'N/A')
                })
            
            df = pd.DataFrame(api_data)
        
        # Save as TSV
        df.to_csv(filename, sep='\t', index=False, encoding='utf-8')

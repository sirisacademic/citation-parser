# search_api.py - Enhanced with multiple DOI support and fixed PubMed handling
import requests
import re
import time
import xml.etree.ElementTree as ET
import urllib
from typing import Dict, List, Any, Optional

# Import enhanced components
from .api_capabilities import APICapabilities
from .field_mapper import FieldMapper, DOIResult
from .progressive_search import SearchOrchestrator, ProgressiveSearchStrategy, ResultDeduplicator

TIMEOUT = 30
TARGET_COUNT_SINGLE_API = 4
TARGET_COUNT_PER_API_ENSEMBLE = 5

class BaseAPIStrategy:
    """Base class for API search strategies with enhanced DOI support"""
    
    def __init__(self):
        self.field_mapper = FieldMapper()
        self.deduplicator = ResultDeduplicator(self.field_mapper)

    def encode_query_value(self, value: str, encoding_type: str = "quote") -> str:
        """Encode a query value for URL safety"""
        if not value:
            return ""
        
        if encoding_type == "quote_plus":
            return urllib.parse.quote_plus(str(value))
        else:
            return urllib.parse.quote(str(value))
    
    def encode_query_params(self, params: Dict[str, Any], encoding_type: str = "quote") -> Dict[str, str]:
        """Encode all query parameters"""
        encoded_params = {}
        for key, value in params.items():
            if value is not None and str(value).strip():
                encoded_params[key] = self.encode_query_value(value, encoding_type)
        return encoded_params

    def enhance_result_with_dois(self, result: Dict[str, Any], api_name: str) -> Dict[str, Any]:
        """
        Enhanced result with multiple DOI information - Fixed for PubMed XML strings
        """
        # Handle PubMed XML string results
        if api_name == "pubmed" and isinstance(result, str):
            # Create dict wrapper for XML string to avoid assignment errors
            xml_content = result
            result = {
                'xml_content': xml_content,
                'api_source': 'pubmed',
                'is_xml': True
            }
        
        enhanced_result = result.copy() if isinstance(result, dict) else {'xml_content': result}
        
        try:
            # Extract DOI information using our enhanced field mapper
            doi_result = self.field_mapper.extract_dois_from_result(result, api_name)
            
            # Add DOI information to the result
            enhanced_result['main_doi'] = doi_result.main_doi
            enhanced_result['alternative_dois'] = doi_result.alternative_dois
            enhanced_result['total_dois'] = doi_result.total_count
            enhanced_result['all_dois'] = self.field_mapper.get_all_dois_from_result(doi_result)
            
            # Maintain backward compatibility - set 'doi' field to main DOI
            if doi_result.main_doi:
                enhanced_result['doi'] = doi_result.main_doi
            
            # Add API source information
            enhanced_result['api_source'] = api_name
            enhanced_result['supports_multiple_dois'] = APICapabilities.supports_multiple_dois(api_name)
            
        except Exception as e:
            # Fallback: try to get DOI from existing field
            existing_doi = None
            if isinstance(result, dict):
                existing_doi = result.get('doi') or result.get('DOI')
            
            if existing_doi:
                cleaned_doi = self.field_mapper.clean_doi(existing_doi)
                enhanced_result['main_doi'] = cleaned_doi
                enhanced_result['alternative_dois'] = []
                enhanced_result['total_dois'] = 1 if cleaned_doi else 0
                enhanced_result['all_dois'] = [cleaned_doi] if cleaned_doi else []
                enhanced_result['doi'] = cleaned_doi  # Backward compatibility
            else:
                enhanced_result['main_doi'] = None
                enhanced_result['alternative_dois'] = []
                enhanced_result['total_dois'] = 0
                enhanced_result['all_dois'] = []
                enhanced_result['doi'] = None
            
            enhanced_result['api_source'] = api_name
            enhanced_result['supports_multiple_dois'] = False
        
        return enhanced_result

    def search(self, ner_entities: Dict[str, List[str]], target_count: int = TARGET_COUNT_SINGLE_API, **kwargs) -> List[Dict]:
        """Main search method using progressive strategy with DOI enhancement"""
        api_name = self.get_api_name()
        
        # Use progressive search orchestrator
        orchestrator = SearchOrchestrator(target_count)
        results = orchestrator.search_single_api(ner_entities, api_name, target_count, self)
        
        # Enhance all results with DOI information
        enhanced_results = []
        #for result in results:
        for i, result in enumerate(results or []):
            enhanced_result = self.enhance_result_with_dois(result, api_name)
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def get_api_name(self) -> str:
        """Return the API name for this strategy"""
        raise NotImplementedError("Subclasses must implement get_api_name")
    
    def _build_api_url(self, query_params: Dict[str, Any]) -> str:
        """Build API-specific URL from query parameters"""
        raise NotImplementedError("Subclasses must implement _build_api_url")

class OpenAlexStrategy(BaseAPIStrategy):
    """OpenAlex API search strategy with enhanced DOI support"""
    
    def get_api_name(self) -> str:
        return "openalex"
        
    def _build_api_url(self, query_params: Dict[str, Any]) -> str:
        """Build OpenAlex API URL from query parameters"""
        base_url = "https://api.openalex.org/works"
        
        # Convert query parameters to OpenAlex filter format
        filter_parts = []
        for key, value in query_params.items():
            if not value:
                continue
            
            # Clean and encode value for OpenAlex compatibility
            cleaned_value = self._clean_openalex_filter_value(str(value))
            if not cleaned_value:  # Skip if cleaning resulted in empty value
                continue
                
            encoded_value = self.encode_query_value(cleaned_value)
            
            if key == "locations.source.id":
                filter_parts.append(f"{key}:{encoded_value}")
            elif key in ["title.search", "raw_author_name.search"]:
                filter_parts.append(f"{key}:{encoded_value}")
            elif key in ["publication_year", "doi", "biblio.volume", "biblio.issue", 
                        "biblio.first_page", "biblio.last_page"]:
                filter_parts.append(f"{key}:{encoded_value}")
        
        if filter_parts:
            filter_string = ",".join(filter_parts)
            return f"{base_url}?filter={filter_string}&mailto=info@sirisacademic.com"
        
        return base_url
    
    def _clean_openalex_filter_value(self, value: str) -> str:
        """Clean filter values for OpenAlex compatibility"""
        if not value:
            return ""
        
        # Remove problematic characters that cause 403 errors
        # Keep alphanumeric, spaces, hyphens, and basic punctuation
        import re
        
        # Remove or replace problematic characters
        cleaned = value
        
        # Remove parentheses and their contents (common in citations)
        cleaned = re.sub(r'\([^)]*\)', '', cleaned)
        
        # Remove or replace other problematic characters
        cleaned = re.sub(r'[,;:"\'`]', '', cleaned)  # Remove commas, semicolons, quotes
        cleaned = re.sub(r'[—–]', '-', cleaned)      # Replace em/en dashes with hyphens
        cleaned = re.sub(r'[^\w\s\-.]', '', cleaned) # Keep only word chars, spaces, hyphens, dots
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Ensure minimum length for meaningful search
        if len(cleaned) < 3:
            return ""
        
        return cleaned

class OpenAIREStrategy(BaseAPIStrategy):
    def get_api_name(self) -> str:
        return "openaire"
    
    def _build_api_url(self, query_params: Dict[str, Any]) -> str:
        """Build OpenAIRE Graph API URL from query parameters"""
        base_url = "https://api.openaire.eu/graph/v1/researchProducts"
        
        # Use parent class encoding method with quote_plus for OpenAIRE
        encoded_params = self.encode_query_params(query_params, "quote_plus")
        
        if encoded_params:
            params_list = [f"{key}={value}" for key, value in encoded_params.items()]
            query_string = "&".join(params_list)
            final_url = f"{base_url}?{query_string}&pageSize={TARGET_COUNT_SINGLE_API}"
            
            return final_url
        
        return f"{base_url}?pageSize={TARGET_COUNT_SINGLE_API}"

class PubMedStrategy(BaseAPIStrategy):
    """PubMed API search strategy with XML parsing to dict structure"""
    
    def get_api_name(self) -> str:
        return "pubmed"
    
    def _build_api_url(self, query_params: Dict[str, Any]) -> str:
        """Build PubMed API URL from query parameters"""
        search_url = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        
        # Build search term from field parameters
        term_parts = []
        for key, value in query_params.items():
            if not value:
                continue
                
            # Encode the value for URL safety
            encoded_value = self.encode_query_value(value)
            
            if key == "doi":
                term_parts.append(f"{encoded_value}[doi]")
            elif key in ["title", "author", "journal", "pdat", "volume", "issue", "first_page", "last_page"]:
                term_parts.append(f"{encoded_value}[{key}]")
        
        if term_parts:
            search_term = " AND ".join(term_parts)
            # Double encode the complete search term since it contains special characters like [, ], AND
            encoded_search_term = self.encode_query_value(search_term, "quote_plus")
            return f"{search_url}?db=pubmed&retmode=json&retmax=100&term={encoded_search_term}"
        
        return f"{search_url}?db=pubmed&retmode=json&retmax=100"
    
    def _parse_pubmed_xml_to_dict(self, xml_content: str) -> Dict[str, Any]:
        """
        Parse PubMed XML content into standardized dict structure
        Returns dict with fields consistent with other APIs
        """
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_content)
            
            # Initialize result dict
            result = {
                'xml_content': xml_content,  # Keep original for fallback
                'api_source': 'pubmed'
            }
            
            # Find the article element
            article = root.find("PubmedArticle/MedlineCitation/Article")
            if article is None:
                return result
            
            # Extract PMID (ID)
            pmid_elem = root.find("PubmedArticle/MedlineCitation/PMID")
            if pmid_elem is not None:
                result['id'] = pmid_elem.text
                result['pmid'] = pmid_elem.text
            
            # Extract DOI
            doi_elem = article.find(".//ELocationID[@EIdType='doi']")
            if doi_elem is not None:
                result['doi'] = doi_elem.text
            
            # Extract title
            title_elem = article.find("ArticleTitle")
            if title_elem is not None:
                result['title'] = title_elem.text or ""
            
            # Extract authors
            authors = []
            author_list = article.find("AuthorList")
            if author_list is not None:
                for author in author_list.findall("Author"):
                    last_name = author.findtext("LastName", "")
                    fore_name = author.findtext("ForeName", "")
                    initials = author.findtext("Initials", "")
                    
                    # Build author name
                    if last_name:
                        author_name = f"{last_name}, {initials or (fore_name[0] if fore_name else '')}."
                        authors.append({
                            'full_name': author_name,
                            'last_name': last_name,
                            'fore_name': fore_name,
                            'initials': initials
                        })
            result['authors'] = authors
            
            # Extract publication year
            pub_date = article.find("Journal/JournalIssue/PubDate")
            if pub_date is not None:
                year_elem = pub_date.find("Year")
                if year_elem is not None:
                    result['publication_year'] = year_elem.text
                else:
                    # Try MedlineDate format
                    medline_date = pub_date.findtext("MedlineDate", "")
                    if medline_date:
                        # Extract year from formats like "2020 Jan-Feb" or "2020"
                        import re
                        year_match = re.search(r'(\d{4})', medline_date)
                        if year_match:
                            result['publication_year'] = year_match.group(1)
            
            # Extract journal information
            journal_elem = article.find("Journal")
            if journal_elem is not None:
                # Journal title
                journal_title = journal_elem.findtext("Title", "")
                if not journal_title:
                    journal_title = journal_elem.findtext("ISOAbbreviation", "")
                result['journal'] = journal_title
                
                # Volume and Issue
                journal_issue = journal_elem.find("JournalIssue")
                if journal_issue is not None:
                    result['volume'] = journal_issue.findtext("Volume", "")
                    result['issue'] = journal_issue.findtext("Issue", "")
            
            # Extract pagination
            pagination = article.find("Pagination")
            if pagination is not None:
                result['first_page'] = pagination.findtext("StartPage", "")
                result['last_page'] = pagination.findtext("EndPage", "")
                
                # Also check for MedlinePgn format
                if not result.get('first_page'):
                    medline_pgn = pagination.findtext("MedlinePgn", "")
                    if medline_pgn:
                        # Handle formats like "123-456" or "e123456"
                        if '-' in medline_pgn:
                            parts = medline_pgn.split('-')
                            result['first_page'] = parts[0]
                            result['last_page'] = parts[1] if len(parts) > 1 else ""
                        else:
                            result['first_page'] = medline_pgn
            
            # Extract abstract (optional)
            abstract_elem = article.find("Abstract/AbstractText")
            if abstract_elem is not None:
                result['abstract'] = abstract_elem.text or ""
            
            return result
            
        except Exception as e:
            # Return minimal dict with original XML on parse error
            return {
                'xml_content': xml_content,
                'api_source': 'pubmed',
                'parse_error': str(e)
            }
    
    def search(self, ner_entities: Dict[str, List[str]], target_count: int = TARGET_COUNT_SINGLE_API, **kwargs) -> List[Dict]:
        """Override search for PubMed's two-step process with XML parsing to dict structure"""
        api_name = self.get_api_name()
        search_capabilities = APICapabilities.get_search_fields(api_name)
        field_combinations = APICapabilities.get_field_combinations(api_name)
        
        all_candidates = []
        
        for combination in field_combinations:
            if len(all_candidates) >= target_count:
                break
            
            # Filter combination to available fields
            available_combination = [
                field for field in combination 
                if field in ner_entities and ner_entities[field] and ner_entities[field][0]
                and APICapabilities.supports_field(api_name, field)
            ]
            
            if not available_combination:
                continue
            
            # Build query parameters
            query_params = self.field_mapper.build_query_params(
                api_name, available_combination, ner_entities, search_capabilities
            )
            
            if not query_params:
                continue
            
            # Execute PubMed search and fetch, then parse XML
            xml_results = self._execute_pubmed_search(query_params)
            
            # Parse each XML result into dict structure
            for xml_content in xml_results:
                if xml_content:  # Skip empty results
                    parsed_dict = self._parse_pubmed_xml_to_dict(xml_content)
                    if parsed_dict:
                        all_candidates.append(parsed_dict)
        
        # Deduplicate using standard logic (now works with dict structure)
        unique_candidates = self.deduplicator.deduplicate_candidates(all_candidates, api_name)
        limited_candidates = unique_candidates[:target_count]
        
        # Enhance results with DOI information
        enhanced_results = []
        for result in limited_candidates:
            enhanced_result = self.enhance_result_with_dois(result, api_name)
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def _execute_pubmed_search(self, query_params: Dict[str, Any]) -> List[str]:
        """Execute PubMed's two-step search and fetch process - returns XML strings"""
        search_url = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        fetch_url = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        
        # Build search term
        term_parts = []
        for key, value in query_params.items():
            if key == "doi":
                term_parts.append(f"{value}[doi]")
            else:
                term_parts.append(f"{value}[{key}]")
        
        search_term = " AND ".join(term_parts)
        
        try:
            # Step 1: Search for PMIDs
            search_params = {
                "db": "pubmed",
                "retmode": "json",
                "retmax": 100,
                "term": search_term,
            }
            
            search_response = requests.get(search_url, params=search_params, timeout=TIMEOUT)
            search_response.raise_for_status()
            search_results = search_response.json()
            
            pmids = search_results.get("esearchresult", {}).get("idlist", [])
            if not pmids:
                return []
            
            # Step 2: Fetch individual records (one by one to avoid huge XML)
            xml_results = []
            for pmid in pmids[:20]:  # Limit to avoid timeouts
                fetch_params = {
                    "db": "pubmed",
                    "retmode": "xml",
                    "id": pmid,
                }
                
                fetch_response = requests.get(fetch_url, params=fetch_params, timeout=TIMEOUT)
                fetch_response.raise_for_status()
                
                if fetch_response.text:
                    xml_results.append(fetch_response.text)
                
                # Small delay to be respectful to NCBI
                time.sleep(0.1)
            
            return xml_results
            
        except Exception as e:
            return []

class CrossRefStrategy(BaseAPIStrategy):
    """CrossRef API search strategy with enhanced DOI support"""
    
    def get_api_name(self) -> str:
        return "crossref"
    
    def _build_api_url(self, query_params: Dict[str, Any]) -> str:
        """Build CrossRef API URL from query parameters"""
        base_url = "https://api.crossref.org/works"
        
        # Build query string from parameters
        encoded_params = self.encode_query_params(query_params, "quote_plus")
        
        if encoded_params:
            params_list = [f"{key}={value}" for key, value in encoded_params.items()]
            query_string = "&".join(params_list)
            return f"{base_url}?{query_string}&rows=20"
        
        return f"{base_url}?rows=20"

class HALSearchStrategy(BaseAPIStrategy):
    """HAL API search strategy with enhanced DOI support"""
    
    def get_api_name(self) -> str:
        return "hal"
    
    def _build_api_url(self, query_params: Dict[str, Any]) -> str:
        """Build HAL API URL from query parameters"""
        base_url = "https://api.archives-ouvertes.fr/search/"
        
        # Build query (HAL uses a single 'q' parameter with field:value syntax)
        query_parts = []
        for key, value in query_params.items():
            if not value:
                continue
            
            # HAL has special handling for quoted vs unquoted fields
            if key in ["title_t", "authFullName_t", "doiId_s", "publicationDateY_s", 
                      "journalTitle_t", "volume_s", "issue_s", "page_s"]:
                
                if key.endswith("_t"):  # Full-text fields - keep quotes but encode content
                    # Remove existing quotes from the original query building, then add them back
                    clean_value = value.strip('"')
                    encoded_value = self.encode_query_value(clean_value)
                    query_parts.append(f"{key}:\"{encoded_value}\"")
                else:  # Exact fields - encode the quoted value
                    clean_value = value.strip('"')
                    encoded_value = self.encode_query_value(clean_value)
                    query_parts.append(f"{key}:\"{encoded_value}\"")
        
        if query_parts:
            query_string = " AND ".join(query_parts)
            # Encode the complete query string for the 'q' parameter
            encoded_query = self.encode_query_value(query_string, "quote_plus")
            return f"{base_url}?q={encoded_query}&fl=*&wt=json&rows=20"
        
        return f"{base_url}?fl=*&wt=json&rows=20"

class SearchAPI:
    """Main search API coordinator with enhanced DOI support and reduced verbosity"""
    
    def __init__(self):
        self.strategies = {
            "openalex": OpenAlexStrategy(),
            "openaire": OpenAIREStrategy(),
            "pubmed": PubMedStrategy(),
            "crossref": CrossRefStrategy(),
            "hal": HALSearchStrategy(),
        }
        self.orchestrator = SearchOrchestrator()
    
    def search_api(self, ner_entities: Dict[str, List[str]], api: str = "openalex", 
                  target_count: int = TARGET_COUNT_SINGLE_API, **kwargs) -> List[Dict]:
        """
        Search using progressive strategy with target candidate count and enhanced DOI support.
        
        Args:
            ner_entities: Dictionary containing extracted NER entities
            api: API name (e.g., "openalex", "openaire")
            target_count: Target number of candidates to retrieve
            
        Returns:
            List of candidates up to target_count, each enhanced with DOI information
        """
        if api not in self.strategies:
            raise ValueError(f"Unsupported API: {api}")
        
        if api not in APICapabilities.get_supported_apis():
            raise ValueError(f"API {api} not configured in APICapabilities")
        
        strategy = self.strategies[api]
        results = strategy.search(ner_entities, target_count=target_count, **kwargs)
        
        return results
           
    def search_multiple_apis(self, ner_entities: Dict[str, List[str]], 
                           apis: List[str], target_count_per_api: int = TARGET_COUNT_PER_API_ENSEMBLE) -> Dict[str, List[Dict]]:
        """Search multiple APIs and return results by API with enhanced DOI support"""
        return self.orchestrator.search_multiple_apis(
            ner_entities, apis, target_count_per_api, self.strategies
        )

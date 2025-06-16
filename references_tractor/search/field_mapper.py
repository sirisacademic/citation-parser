# search/field_mapper.py
"""
Handles API-specific field mapping and preprocessing.
Enhanced with multiple DOI extraction support.
"""

import re
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from googlesearch import search
import requests

class DOIResult(NamedTuple):
    """Structure for DOI extraction results"""
    main_doi: Optional[str]
    alternative_dois: List[str]
    total_count: int

class FieldMapper:
    """Handles field preprocessing and API-specific transformations"""
    
    def __init__(self):
        self._journal_id_cache = {}  # Cache for OpenAlex journal ID lookups
           
    def clean_doi(self, doi: str) -> Optional[str]:
        """Clean and normalize a single DOI"""
        if not doi:
            return None
        
        # Handle case where DOI might be a list
        if isinstance(doi, list):
            doi = doi[0] if doi else ""
        
        if not isinstance(doi, str):
            return None
        
        # Remove common prefixes and whitespace
        cleaned = doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "").strip()
        
        # Validate DOI format
        if re.match(r'^10\.\d{4,}/[^\s]+$', cleaned):
            return cleaned
        
        return None
    
    def extract_dois_from_result(self, api_result: Dict, api: str) -> DOIResult:
        """
        Extract main DOI and alternative DOIs from a single API result
        Returns: DOIResult(main_doi, alternative_dois, total_count)
        """
        main_doi = None
        alternative_dois = []
        
        try:
            if api == "openalex":
                # OpenAlex typically has one DOI per result
                doi = api_result.get("doi")
                if doi:
                    main_doi = self.clean_doi(doi)
                        
            elif api == "openaire":
                # OpenAIRE can have multiple DOIs in pids array
                if 'results' in api_result and api_result['results']:
                    result = api_result['results'][0]
                else:
                    result = api_result
                
                # Extract DOIs from main pids array - NULL SAFE
                doi_values = []
                pids = result.get('pids') or []  # Handle both missing and null
                if isinstance(pids, list):  # Safety check
                    for pid in pids:
                        if isinstance(pid, dict) and pid.get('scheme') == 'doi':
                            cleaned = self.clean_doi(pid.get('value', ''))
                            if cleaned and cleaned not in doi_values:
                                doi_values.append(cleaned)
                
                # Also check instances for additional DOIs - NULL SAFE
                instances = result.get('instances') or []  # Handle both missing and null
                if isinstance(instances, list):  # Safety check
                    for instance in instances:
                        if not isinstance(instance, dict):
                            continue
                        instance_pids = instance.get('pids') or []  # Handle null
                        if isinstance(instance_pids, list):  # Safety check
                            for pid in instance_pids:
                                if isinstance(pid, dict) and pid.get('scheme') == 'doi':
                                    cleaned = self.clean_doi(pid.get('value', ''))
                                    if cleaned and cleaned not in doi_values:
                                        doi_values.append(cleaned)
                
                # Set main DOI as first one, rest as alternatives
                if doi_values:
                    main_doi = doi_values[0]
                    alternative_dois = doi_values[1:] if len(doi_values) > 1 else []
                            
            elif api == "crossref":
                # CrossRef has one main DOI per result
                doi = api_result.get("DOI")
                if doi:
                    main_doi = self.clean_doi(doi)
                        
            elif api == "pubmed":
                # PubMed now uses parsed dict structure - NULL SAFE
                if isinstance(api_result, dict):
                    # Extract from parsed structure
                    doi = api_result.get('doi')
                    if doi:
                        main_doi = self.clean_doi(doi)
                    else:
                        # Fallback to XML parsing if still has xml_content
                        xml_content = api_result.get('xml_content')
                        if xml_content:
                            try:
                                import xml.etree.ElementTree as ET
                                root = ET.fromstring(xml_content)
                                doi_elem = root.find(".//ELocationID[@EIdType='doi']")
                                if doi_elem is not None:
                                    main_doi = self.clean_doi(doi_elem.text)
                            except Exception:
                                pass
                
            elif api == "hal":
                # HAL typically has one DOI per result
                doi = api_result.get("doiId_s")
                if doi:
                    main_doi = self.clean_doi(doi)
                        
        except Exception as e:
            print(f"Error extracting DOIs for {api}: {e}")
        
        # Calculate total count
        total_count = len([d for d in [main_doi] + alternative_dois if d])
        
        return DOIResult(main_doi, alternative_dois, total_count)
    
    def check_doi_match_with_alternatives(self, doi_result: DOIResult, expected_doi: str) -> str:
        """
        Check if main DOI or any alternative DOI matches the expected DOI
        Returns: EXACT, INCORRECT, or N/A
        """
        if not expected_doi:
            return "N/A"
        
        normalized_expected = self.clean_doi(expected_doi)
        if not normalized_expected:
            return "N/A"
        
        # Check main DOI first
        if doi_result.main_doi and self.clean_doi(doi_result.main_doi) == normalized_expected:
            return "EXACT"
        
        # Check alternative DOIs
        for alt_doi in doi_result.alternative_dois:
            if self.clean_doi(alt_doi) == normalized_expected:
                return "EXACT"
        
        # If we have any DOIs but none match
        if doi_result.main_doi or doi_result.alternative_dois:
            return "INCORRECT"
        
        return "N/A"
    
    def get_all_dois_from_result(self, doi_result: DOIResult) -> List[str]:
        """
        Get all DOIs (main + alternatives) as a flat list
        Useful for deduplication and ensemble logic
        """
        all_dois = []
        if doi_result.main_doi:
            all_dois.append(doi_result.main_doi)
        all_dois.extend(doi_result.alternative_dois)
        return all_dois
        
    def clean_title(self, title: str) -> Optional[str]:
        """Clean title for search"""
        if not title:
            return None
        
        # Remove quotes and normalize whitespace
        cleaned = re.sub(r'[\'\"]+', '', title)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Remove trailing punctuation for search
        cleaned = re.sub(r'[.,;:\-\s]+$', '', cleaned)
        
        return cleaned if len(cleaned) >= 3 else None
           
    def extract_author_surname(self, authors: str) -> Optional[str]:
        """Extract surname from author string for search"""
        if not authors:
            return None
        
        # Remove "et al" for OpenAIRE searches
        cleaned = authors.replace(" et al", "").replace(" et al.", "").strip()
        
        # Handle "LastName, FirstName" format
        if "," in cleaned:
            surname = cleaned.split(",")[0].strip()
            return surname if surname else None
        
        # Handle "FirstName LastName" format  
        parts = cleaned.split()
        if parts:
            surname = parts[-1].strip()
            return surname if surname else None
        
        return None
    
    def year_to_date_range(self, year: str) -> Tuple[str, str]:
        """Convert year to date range for OpenAIRE"""
        if not year:
            return None, None
        
        try:
            year_int = int(year)
            if 1800 <= year_int <= 2050:
                return f"{year}-01-01", f"{year}-12-31"
        except ValueError:
            pass
        
        return None, None
    
    def year_to_filter(self, year: str) -> Optional[str]:
        """Convert year to filter format for CrossRef"""
        if not year:
            return None
        
        try:
            year_int = int(year)
            if 1800 <= year_int <= 2050:
                return f"from-pub-date:{year}-01-01,until-pub-date:{year}-12-31"
        except ValueError:
            pass
        
        return None
    
    def resolve_journal_id(self, journal_name: str) -> Optional[str]:
        """Resolve journal name to OpenAlex ID"""
        if not journal_name:
            return None
        
        # Check cache first
        if journal_name in self._journal_id_cache:
            return self._journal_id_cache[journal_name]
        
        source_url = "https://api.openalex.org/sources"
        
        try:
            # Try direct name match first
            response = requests.get(f"{source_url}?filter=display_name.search:{journal_name}", timeout=10)
            if response.status_code == 200:
                journals = response.json().get("results", [])
                if len(journals) == 1:
                    journal_id = journals[0]["id"].split("/")[-1]
                    self._journal_id_cache[journal_name] = journal_id
                    return journal_id
                elif len(journals) > 1:
                    # Multiple matches - return the first one
                    journal_id = journals[0]["id"].split("/")[-1]
                    self._journal_id_cache[journal_name] = journal_id
                    return journal_id
            
            # Fallback to Google search for journal expansion
            try:
                results = search(f"{journal_name} journal", num_results=1, advanced=True)
                expanded_names = [
                    re.split(r'[-:]', result.title.title().replace('The', ''))[0].strip() 
                    for result in results
                ]
                
                if expanded_names:
                    response = requests.get(f"{source_url}?filter=display_name.search:{expanded_names[0]}", timeout=10)
                    if response.status_code == 200:
                        journals = response.json().get("results", [])
                        if journals:
                            journal_id = journals[0]["id"].split("/")[-1]
                            self._journal_id_cache[journal_name] = journal_id
                            return journal_id
            except Exception:
                pass  # Google search failed, continue without it
                
        except Exception:
            pass  # API call failed
        
        # Cache negative result to avoid repeated lookups
        self._journal_id_cache[journal_name] = None
        return None
    
    def preprocess_field(self, field_name: str, value: str, preprocessing_func: Optional[str] = None) -> Any:
        """Apply preprocessing to a field value"""
        if not value or not preprocessing_func:
            return value
        
        # Get the preprocessing function by name
        if hasattr(self, preprocessing_func):
            func = getattr(self, preprocessing_func)
            return func(value)
        
        return value
    
    def build_query_params(self, api: str, field_combination: List[str], ner_entities: Dict[str, List[str]], 
                          search_capabilities: Dict[str, Any]) -> Dict[str, Any]:
        """Build API-specific query parameters with comprehensive error handling"""
        query_params = {}
        
        for field in field_combination:
            try:
                # Get the value from NER entities with better handling
                entity_values = ner_entities.get(field, [])
                if not entity_values:
                    continue
                
                # Handle nested lists and get first valid value
                value = None
                for val in entity_values:
                    if isinstance(val, list) and val:
                        # Handle nested lists (e.g., [["value"]])
                        inner_val = val[0] if val else None
                        if isinstance(inner_val, str) and inner_val.strip():
                            value = inner_val
                            break
                    elif isinstance(val, str) and val.strip():
                        value = val
                        break
                
                if not value:
                    continue
                
                # Get search field configuration
                search_config = search_capabilities.get(field)
                if not search_config:
                    continue
                
                # Apply preprocessing if specified
                if search_config.required_preprocessing:
                    try:
                        processed_value = self.preprocess_field(field, value, search_config.required_preprocessing)
                        
                        # Handle special cases that return tuples (like date ranges)
                        if isinstance(processed_value, tuple):
                            if field == "PUBLICATION_YEAR" and api == "openaire":
                                from_date, to_date = processed_value
                                if from_date and to_date:
                                    query_params["fromPublicationDate"] = from_date
                                    query_params["toPublicationDate"] = to_date
                            elif field == "PUBLICATION_YEAR" and api == "crossref":
                                # CrossRef filter format
                                if processed_value and isinstance(processed_value, str):
                                    query_params["filter"] = processed_value
                            continue
                        
                        if processed_value is None:
                            continue
                        value = processed_value
                        
                    except Exception as e:
                        print(f"Error preprocessing {field} for {api}: {e}")
                        continue
                
                # Get API-specific parameter name
                api_field_name = search_config.api_field_name
                
                # Handle special field formatting with error handling
                try:
                    if api == "openaire":
                        # OpenAIRE Graph API special handling
                        if field == "DOI":
                            cleaned_doi = self.clean_doi(value) if hasattr(self, 'clean_doi') else value
                            if cleaned_doi:
                                query_params["pid"] = cleaned_doi
                        elif field == "TITLE":
                            # Use mainTitle parameter for OpenAIRE Graph API
                            query_params["mainTitle"] = value
                        elif field == "AUTHORS":
                            # Use authorFullName parameter
                            query_params["authorFullName"] = value
                        elif field == "PUBLICATION_YEAR":
                            # Use correct date parameters for OpenAIRE Graph API
                            from_date, to_date = self.year_to_date_range(value)
                            if from_date and to_date:
                                query_params["fromPublicationDate"] = from_date
                                query_params["toPublicationDate"] = to_date
                        else:
                            # For any other fields, use the api_field_name directly
                            query_params[api_field_name] = value
                            
                    elif api == "pubmed":
                        # PubMed uses [field] syntax in search terms
                        if field == "DOI":
                            cleaned_doi = self.clean_doi(value) if hasattr(self, 'clean_doi') else value
                            if cleaned_doi:
                                query_params["doi"] = cleaned_doi
                        elif field == "PUBLICATION_YEAR":
                            # PubMed date format
                            query_params["pdat"] = value
                        else:
                            # Other fields use [field] syntax
                            field_tag = api_field_name.replace("_", "")  # Remove underscores
                            query_params[field_tag] = value
                            
                    elif api == "hal":
                        # HAL uses specific query syntax
                        if hasattr(search_config, 'field_type') and search_config.field_type == "search":
                            # Text search fields get quoted
                            query_params[api_field_name] = f'"{value}"'
                        else:
                            # Exact match fields
                            query_params[api_field_name] = value
                            
                    elif api == "crossref":
                        # CrossRef query handling
                        if field == "PUBLICATION_YEAR":
                            # Already handled in preprocessing as tuple
                            continue
                        elif field == "DOI":
                            # CrossRef uses query for DOI search
                            cleaned_doi = self.clean_doi(value) if hasattr(self, 'clean_doi') else value
                            if cleaned_doi:
                                query_params["query"] = cleaned_doi
                        else:
                            query_params[api_field_name] = value
                            
                    elif api == "openalex":
                        # OpenAlex field handling
                        if field == "DOI":
                            cleaned_doi = self.clean_doi(value) if hasattr(self, 'clean_doi') else value
                            if cleaned_doi:
                                query_params["doi"] = cleaned_doi
                        elif field == "JOURNAL":
                            # Special journal ID resolution for OpenAlex
                            journal_id = self.resolve_journal_id(value) if hasattr(self, 'resolve_journal_id') else None
                            if journal_id:
                                query_params["locations.source.id"] = journal_id
                            else:
                                # Fallback to journal name search
                                query_params["primary_location.source.display_name.search"] = value
                        else:
                            query_params[api_field_name] = value
                            
                    else:
                        # Default handling for other APIs
                        query_params[api_field_name] = value
                        
                except Exception as e:
                    print(f"Error formatting {field} for {api}: {e}")
                    continue
                    
            except Exception as e:
                print(f"Error processing field {field} for {api}: {e}")
                continue
        
        return query_params
    
    def extract_response_field(self, data: Any, field_config: Any, api: str) -> Any:
        """Extract a field from API response using the field configuration"""
        if not data or not field_config:
            return None
        
        path = field_config.path
        
        try:
            if api == "pubmed":
                # Handle both dict structure and XML fallback
                if isinstance(data, dict):
                    # Try direct field access first (from parsed structure)
                    if path == "PubmedArticle/MedlineCitation/PMID":
                        return data.get('pmid') or data.get('id')
                    elif path == "PubmedArticle/MedlineCitation/Article/ELocationID[@EIdType='doi']":
                        return data.get('doi')
                    elif path == "PubmedArticle/MedlineCitation/Article/ArticleTitle":
                        return data.get('title')
                    elif path == "PubmedArticle/MedlineCitation/Article/AuthorList/Author":
                        return data.get('authors', [])
                    elif path == "PubmedArticle/MedlineCitation/Article/Journal/JournalIssue/PubDate/Year":
                        return data.get('publication_year')
                    elif path == "PubmedArticle/MedlineCitation/Article/Journal/Title":
                        return data.get('journal')
                    elif path == "PubmedArticle/MedlineCitation/Article/Journal/JournalIssue/Volume":
                        return data.get('volume')
                    elif path == "PubmedArticle/MedlineCitation/Article/Journal/JournalIssue/Issue":
                        return data.get('issue')
                    
                    # Fallback to XML parsing if xml_content exists
                    if 'xml_content' in data:
                        import xml.etree.ElementTree as ET
                        root = ET.fromstring(data['xml_content'])
                        return root.findtext(path)
                
                return None
            
            # Handle JSON path extraction for other APIs
            return self._extract_json_path(data, path)
            
        except Exception:
            # Try fallback paths if available
            if field_config.fallback_paths:
                for fallback_path in field_config.fallback_paths:
                    try:
                        return self._extract_json_path(data, fallback_path)
                    except Exception:
                        continue
            return None
    
    def _extract_json_path(self, data: Any, path: str) -> Any:
        """Extract value from nested JSON using dot notation path"""
        if not path:
            return data
        
        parts = path.split('.')
        current = data
        
        for part in parts:
            if current is None:
                return None
            
            # Handle array indexing like "title[0]"
            if '[' in part and ']' in part:
                field_name = part.split('[')[0]
                index_str = part.split('[')[1].split(']')[0]
                
                if isinstance(current, dict) and field_name in current:
                    current = current[field_name]
                    if isinstance(current, list):
                        try:
                            index = int(index_str)
                            current = current[index] if 0 <= index < len(current) else None
                        except (ValueError, IndexError):
                            current = None
                    else:
                        current = None
                else:
                    current = None
            else:
                # Regular field access
                if isinstance(current, dict):
                    current = current.get(part)
                else:
                    current = None
        
        return current

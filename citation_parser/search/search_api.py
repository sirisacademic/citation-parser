import requests
from googlesearch import search
import re
import time
import requests
import time


class BaseAPIStrategy:
    def search(self, ner_entities, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def filter_field_combinations(self, fields, field_combinations):
        """
        Filter field combinations to only include those with non-None values in fields.
        
        :param fields: Dictionary of field values
        :param field_combinations: List of field combinations
        :return: Filtered list of field combinations
        """
        return [
            combination
            for combination in field_combinations
            if all(fields.get(field) is not None for field in combination)
        ]


class OpenAlexStrategy(BaseAPIStrategy):
    def search(self, ner_entities, **kwargs):
        base_url = "https://api.openalex.org/works"
        source_url = "https://api.openalex.org/sources"

        # Extract fields from NER
        fields = {
            "title": ner_entities.get("TITLE", [None])[0],
            "journal": ner_entities.get("JOURNAL", [None])[0],
            "first_page": ner_entities.get("PAGE_FIRST", [None])[0],
            "last_page": ner_entities.get("PAGE_LAST", [None])[0],
            "volume": ner_entities.get("VOLUME", [None])[0],
            "issue": ner_entities.get("ISSUE", [None])[0],
            "year": ner_entities.get("PUBLICATION_YEAR", [None])[0],
            "doi": ner_entities.get("DOI", [None])[0].replace("https://doi.org/", "").replace(' ', '') if ner_entities.get("DOI") else None,
            "author": ner_entities.get("AUTHORS", [None])[0].split(" ")[0].strip() if ner_entities.get("AUTHORS") else None
        }

        source_url = "https://api.openalex.org/sources"
        # Map journal name to its ID if journal is provided
        if fields["journal"]:
            response = requests.get(f"{source_url}?filter=display_name.search:{fields['journal']}")
            journals = response.json().get("results", [])
            if len(journals) == 1:
                fields["journal"] = journals[0]["id"].split("/")[-1]
            else:
                res = search(fields['journal']+" journal", num_results=1,advanced=True)
                try:
                    expanded_version = [re.split(r'[-:]', i.title.title().replace('The', ''))[0].strip() for i in res][0]
                    response = requests.get(f"{source_url}?filter=display_name.search:{expanded_version}")
                    journals = response.json().get("results", [])
                    if len(journals) == 1:
                        fields["journal"] = journals[0]["id"].split("/")[-1]
                    else:
                        fields["journal"] = None
                except:
                    fields["journal"] = None

        # Define field combinations for the search
        field_combinations = [
                ['doi'],
                ["title", "year"],
                ["title"],
                ["title", "year", "author"],
                ["title", "year", "author", "journal", "first_page", "last_page", "volume", "issue"],
                ["title", "year", "author", "journal", "first_page", "last_page", "volume"],
                ["title", "year", "author", "first_page", "last_page", "volume"],
                ["title", "year", "author", "journal", "volume", "issue"],
                ["title", "year", "author", "journal"],
                ["year", "author","first_page", "last_page"],
                ["year", "author","journal","volume"],
                ["year","journal","volume","first_page"],
            ]
        

        # Filter field combinations based on available fields
        filtered_combinations = self.filter_field_combinations(fields, field_combinations)
        
        if 'title' in [key for key, value in fields.items() if value is not None]:# in not null keys
            filtered_combinations.append(["title-half","year"])
            filtered_combinations.append(["title-half"])

        candidates = []
        results = []

        for combination in filtered_combinations:
            # Build query parameters dynamically based on available fields
            query_params = {}
            for field in combination:
                field_search = field.split('-')[0]
                value = fields.get(field_search)  # Use .get() to handle missing keys

                if value:
                    if field == "title":
                        value = value.replace(',', '').replace(':', '')
                        query_params["title.search"] = value

                    elif field == "doi":
                        query_params["doi"] = value

                    elif field == "title-half":
                        words = value.split()
                        mid_point = len(words) // 2
                        first_half = " ".join(words[:mid_point])
                        second_half = " ".join(words[mid_point:])
                        query_params["title.search"] = f"{first_half}|{second_half}"

                    elif field == "author":
                        query_params["raw_author_name.search"] = value

                    elif field == "year":
                        query_params["publication_year"] = value

                    elif field == "journal":
                        query_params["locations.source.id"] = value

                    elif field == "first_page":
                        query_params["biblio.first_page"] = value

                    elif field == "last_page":
                        query_params["biblio.last_page"] = value

                    elif field == "volume":
                        query_params["biblio.volume"] = value

                    elif field == "issue":
                        query_params["biblio.issue"] = value

            # Combine query parameters into the query string
            query_string = "&".join(f"{k}:{v}" for k, v in query_params.items() if v)
            api_url = f"{base_url}?filter={query_string}&mailto=info@sirisacademic.com".replace(',,',',')

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(api_url, timeout=2)  # Set timeout to 2 seconds
                    if response.status_code == 200:
                        results = response.json().get("results", [])
                        candidates.extend(results)
                        break  # Exit retry loop on success
                    else:
                        print(f"Attempt {attempt + 1}: Received status code {response.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"Attempt {attempt + 1}: Request failed with error: {e}")
                # Wait before retrying
                if attempt < max_retries - 1:
                    time.sleep(1)

            if len(results)==1:  # Return results if found
                return results
                
            if len(candidates)>1:
                return candidates
                        
        return candidates

class OpenAIREStrategy(BaseAPIStrategy):
    def search(self, ner_entities, **kwargs):
        base_url = "https://api.openaire.eu/search/publications"
        
        # Map fields to OpenAIRE API-specific parameters (this will differ from OpenAlex)

        year = ner_entities.get("PUBLICATION_YEAR", [None])[0]

        fields = {
            "title": ner_entities.get("TITLE", [None])[0],
            "author": ner_entities.get("AUTHORS", [None])[0],
            "doi": ner_entities.get("DOI", [None])[0].replace("https://doi.org/", "").replace(' ', '') if ner_entities.get("DOI") else None,
            "year": ner_entities.get("PUBLICATION_YEAR", [None])[0],
        }

        if year:
            fields["fromDateAccepted"] = f"{year}-01-01"
            fields["toDateAccepted"] = f"{year}-12-31"

        field_combinations = [
                ['doi'],
                ["title", "author", "fromDateAccepted","toDateAccepted"],
                ["title", "fromDateAccepted","toDateAccepted"],
                ["title"],
        ]

        filtered_combinations = self.filter_field_combinations(fields, field_combinations)

        candidates = []
        results = []

        for combination in filtered_combinations:
            # Build query parameters dynamically based on available fields
            query_params = {}
            for field in combination:
                field_search = field.split('-')[0] # for the title
                value = fields[field_search]
                if value:
                    query_params[field] = value
                    
            # Construct query string
            query_string = "&".join(f"{k}={v}" for k, v in query_params.items() if v)
            #query_string = ",".join(query_string)
            api_url = f"{base_url}?{query_string}&format=json"

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(api_url, timeout=2)  # Set timeout to 2 seconds
                    if response.status_code == 200:
                        try:
                            json_response = response.json()
                            if json_response is None:
                                raise ValueError("API response is None.")

                            results = (
                                json_response.get("response", {})
                                .get("results", {})
                                .get("result", [])
                            )

                            if not results:  # Handle empty or missing results
                                raise ValueError("No results found in API response.")

                            candidates.extend(results)
                            
                            break  # Exit retry loop on success
                        except Exception as e:
                            print(f"Error processing API response: {e}")
                            continue
                    else:
                        print(f"Attempt {attempt + 1}: Received status code {response.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"Attempt {attempt + 1}: Request failed with error: {e}")
                # Wait before retrying
                if attempt < max_retries - 1:
                    time.sleep(1)

            if len(results)==1:  # Return results if found
                return results
                
            if len(candidates)>1:
                return candidates
                        
        return candidates
        
class PubMedStrategy(BaseAPIStrategy):
    def search(self, ner_entities, **kwargs):
        # PubMed API base URLs
        search_url = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        fetch_url = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

        # Extract fields from NER
        fields = {
            "title": ner_entities.get("TITLE", [None])[0],
            "journal": ner_entities.get("JOURNAL", [None])[0],
            "doi": ner_entities.get("DOI", [None])[0],
            "author": ner_entities.get("AUTHORS", [None])[0],
            "year": ner_entities.get("PUBLICATION_YEAR", [None])[0],
            "volume": ner_entities.get("VOLUME", [None])[0],
            "issue": ner_entities.get("ISSUE", [None])[0],
        }

        # Define field combinations
        field_combinations = [
            ["doi"],
            ["title", "author", "year", "journal", "volume", "issue"],
            ["title", "year", "journal", "volume", "issue"],
            ["title", "author", "year"],
            ["title", "year"],
            ["title"]
        ]

        # Filter field combinations based on available fields
        filtered_combinations = self.filter_field_combinations(fields, field_combinations)

        candidates = []
        results = []

        for combination in filtered_combinations:
            # Build query dynamically based on the current field combination
            query_parts = []
            for field in combination:
                value = fields.get(field)
                if value:
                    if field == "title":
                        query_parts.append(f"{value}[title]")
                    elif field == "doi":
                        query_parts.append(f"{value}[doi]")
                    elif field == "author":
                        query_parts.append(f"{value}[author_name]")
                    elif field == "year":
                        query_parts.append(f"{value}[pdat]")
                    elif field == "journal":
                        query_parts.append(f"{value}[journal]")
                    elif field == "volume":
                        query_parts.append(f"{value}[volume]")
                    elif field == "issue":
                        query_parts.append(f"{value}[issue]")

            query = " AND ".join(query_parts)

            # Perform the search
            params = {
                "db": "pubmed",
                "retmode": "json",
                "retmax": 100,
                "term": query,
            }

            try:
                search_response = requests.get(search_url, params=params, timeout=5)
                search_response.raise_for_status()
                search_results = search_response.json()

                pmids = search_results.get("esearchresult", {}).get("idlist", [])
                if not pmids:
                    continue  # Skip to the next combination if no results found

                # Fetch details for the returned PMIDs
                fetch_params = {
                    "db": "pubmed",
                    "retmode": "xml",
                    "id": ",".join(pmids),
                }
                fetch_response = requests.get(fetch_url, params=fetch_params, timeout=5)
                fetch_response.raise_for_status()

                # Parse fetched results (as XML or plain text)
                results = fetch_response.text  # Raw XML can be parsed further if needed
                candidates.append(results)

                if len(pmids) == 1:  # If a single result is found, return it
                    return candidates

            except requests.exceptions.RequestException as e:
                print(f"Error in PubMed API request: {e}")
                continue

        return candidates

class CrossRefStrategy(BaseAPIStrategy):
    def search(self, ner_entities, **kwargs):
        base_url = "https://api.crossref.org/works"

        # Extract fields from NER entities
        fields = {
            "title": ner_entities.get("TITLE", [None])[0],
            "year": ner_entities.get("PUBLICATION_YEAR", [None])[0],
            "journal": ner_entities.get("JOURNAL", [None])[0],
            "doi": ner_entities.get("DOI", [None])[0].replace("https://doi.org/", "").replace(' ', '')
                    if ner_entities.get("DOI") and ner_entities.get("DOI")[0] is not None
                    else None,
            "author": ner_entities.get("AUTHORS", [None])[0].split(" ")[0].strip() if ner_entities.get("AUTHORS") else None,

        }

        # Define field combinations for search queries
        field_combinations = [
            ["doi"],
            ["title", "year", "author", "journal"],
            ["title", "author", "year"],
            ["title", "year"],
            ["title", "author"],
            ["year", "author", "journal"],
            ["title"],
        ]

        # Filter combinations based on available fields
        filtered_combinations = self.filter_field_combinations(fields, field_combinations)

        candidates = []
        results = []

        for combination in filtered_combinations:
            query_params = {}

            for field in combination:
                value = fields.get(field)
                if value:
                    if field == "title":
                        query_params["query.title"] = value
                    elif field == "author":
                        query_params["query.author"] = value
                    elif field == "year":
                        query_params["filter"] = f"from-pub-date:{value}-01-01,until-pub-date:{value}-12-31"
                    elif field == "doi":
                        query_params["query"] = value  # DOI search
                    elif field == "journal":
                        query_params["query.container-title"] = value  # DOI search

            # Construct the API request
            query_string = "&".join(f"{k}={v}" for k, v in query_params.items())
            api_url = f"{base_url}?{query_string}"

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(api_url, timeout=2)
                    if response.status_code == 200:
                        data = response.json()
                        items = data.get("message", {}).get("items", [])
                        candidates.extend(items)
                        break  # Exit retry loop if successful
                    else:
                        print(f"Attempt {attempt + 1}: Status code {response.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"Attempt {attempt + 1}: Request failed - {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(1)

            if len(results)==1:  # Return results if found
                return results
                
            if len(candidates)>1:
                return candidates
        return candidates

class HALSearchStrategy(BaseAPIStrategy):
    def __init__(self, base_url="https://api.archives-ouvertes.fr/search/"):
        self.base_url = base_url

    def build_query(self, fields, combination):
        """Build query string for specific field combination with AND operator."""
        query_parts = []
        for field in combination:
            field_search = field.split('-')[0]
            value = fields.get(field_search)

            if value:
                if field == "title":
                    value = value.replace(",", "").replace(":", "")
                    query_parts.append(f"title_t:\"{value}\"")  # Changed to title_t for full-text search
                elif field == "doi":
                    query_parts.append(f"doiId_s:\"{value}\"")
                elif field == "title-half":
                    words = value.split()
                    mid_point = len(words) // 2
                    first_half = " ".join(words[:mid_point])
                    second_half = " ".join(words[mid_point:])
                    query_parts.append(f"title_t:\"{first_half}\" OR title_t:\"{second_half}\"")  # Changed to title_t
                elif field == "author":
                    query_parts.append(f"authFullName_t:\*{value}\*")
                elif field == "year":
                    query_parts.append(f"publicationDateY_s:{value}")
                elif field == "journal":
                    query_parts.append(f"journalTitle_t:\*{value}\*")
                elif field == "first_page":
                    query_parts.append(f"page_s:\*{value}\*")
                elif field == "last_page":
                    query_parts.append(f"page_s:\*{value}\*")
                elif field == "volume":
                    query_parts.append(f"volume_s:\"{value}\"")
                elif field == "issue":
                    query_parts.append(f"issue_s:\"{value}\"")
        
        # Join the query parts with AND operator to combine fields
        return " AND ".join(query_parts)

    def search(self, ner_entities, **kwargs):
        """Perform search based on extracted fields from NER."""
        fields = {
            "title": ner_entities.get("TITLE", [None])[0],
            "journal": ner_entities.get("JOURNAL", [None])[0],
            "first_page": ner_entities.get("PAGE_FIRST", [None])[0],
            "last_page": ner_entities.get("PAGE_LAST", [None])[0],
            "volume": ner_entities.get("VOLUME", [None])[0],
            "issue": ner_entities.get("ISSUE", [None])[0],
            "year": ner_entities.get("PUBLICATION_YEAR", [None])[0],
            "doi": ner_entities.get("DOI", [None])[0].replace("https://doi.org/", "").replace(' ', '') if ner_entities.get("DOI") else None,
            "author": ner_entities.get("AUTHORS", [None])[0].split(" ")[0].strip() if ner_entities.get("AUTHORS") else None
        }

        # Define field combinations for querying
        field_combinations = [
            ["doi"],
            ["title", "year"],
            ["title"],
            ["title", "year", "author"],
            ["title", "year", "author", "journal", "first_page", "last_page", "volume", "issue"],
            ["title", "year", "author", "journal", "first_page", "last_page", "volume"],
            ["title", "year", "author", "first_page", "last_page", "volume"],
            ["title", "year", "author", "journal", "volume", "issue"],
            ["title", "year", "author", "journal"],
            ["year", "author", "first_page", "last_page"],
            ["year", "author", "journal", "volume"],
            ["year", "journal", "volume", "first_page"],
        ]

        filtered_combinations = self.filter_field_combinations(fields, field_combinations)

        # Add title-half variations if title exists
        if fields.get("title"):
            filtered_combinations.append(["title-half", "year"])
            filtered_combinations.append(["title-half"])

        candidates = []
        results = []

        for combination in filtered_combinations:
            query_string = self.build_query(fields, combination)

            if not query_string:  # Skip empty queries
                continue

            # Build the API request URL with the generated query string
            api_url = f"{self.base_url}?q={query_string}&fl=*&wt=json&rows=20"

            # Retry logic for failed requests
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(api_url, timeout=3)  # 3-second timeout
                    if response.status_code == 200:
                        docs = response.json().get("response", {}).get("docs", [])
                        candidates.extend(docs)
                        break  # Exit retry loop on success
                    else:
                        print(f"Attempt {attempt + 1}: Received status code {response.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"Attempt {attempt + 1}: Request failed with error: {e}")

                if attempt < max_retries - 1:
                    time.sleep(1)

            # If only one result found, return immediately
            if len(results) == 1:
                return results

            # If multiple results found, return them
            if len(candidates) > 1:
                return candidates

        return candidates  # Return all collected results if no perfect match
               

class SearchAPI:
    def __init__(self):
        self.strategies = {
            "openalex": OpenAlexStrategy(),
            "openaire": OpenAIREStrategy(),
            "pubmed": PubMedStrategy(),
            "crossref": CrossRefStrategy(),  # Added CrossRef Strategy
            "hal": HALSearchStrategy(),  # Added CrossRef Strategy
        }

    def search_api(self, ner_entities, api="openalex", **kwargs):
        """
        Search using the specified API strategy.
        :param ner_entities: Dictionary containing extracted NER entities.
        :param api: API name (e.g., "openalex", "openaire").
        :return: JSON response or None if no results found.
        """
        if api not in self.strategies:
            raise ValueError(f"Unsupported API: {api}")
        
        strategy = self.strategies[api]
        return strategy.search(ner_entities, **kwargs)
    


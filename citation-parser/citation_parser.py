# citation_parser.py
import requests
from tqdm.notebook import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, AutoModelForSequenceClassification
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
import requests
import string
from googlesearch import search
import re


class CitationParser:
    def __init__(self, ner_model_path="SIRIS-Lab/citation-parser-ENTITY", select_model_path="SIRIS-Lab/citation-parser-SELECT",prescreening_model_path="SIRIS-Lab/citation-parser-TYPE", device="cpu"):
        self.ner_pipeline = pipeline(
            "ner",
            model=AutoModelForTokenClassification.from_pretrained(ner_model_path),
            tokenizer=AutoTokenizer.from_pretrained(ner_model_path),
            aggregation_strategy="simple",
            device=device
        )
        self.select_pipeline = pipeline(
            "text-classification",
            model=AutoModelForSequenceClassification.from_pretrained(select_model_path),
            tokenizer=AutoTokenizer.from_pretrained(select_model_path),
            device='cpu'
        )

        self.prescreening_pipeline = pipeline(
            "text-classification",
            model=AutoModelForSequenceClassification.from_pretrained(prescreening_model_path),
            tokenizer=AutoTokenizer.from_pretrained(prescreening_model_path),
            device='cpu'
        )

    def process_ner_entities(self, citation):
        output = self.ner_pipeline(citation)

        ner_entities = {}
        for entity in output:
            entity_group = entity.get("entity_group")
            word = entity.get("word", "")
            if entity_group not in ner_entities:
                ner_entities[entity_group] = []
            ner_entities[entity_group].append(word)
        return ner_entities
    
    def generate_apa_citation(self, data):
        # Extract authors and limit to a maximum of 3
        authors_list = [auth['raw_author_name'] for auth in data.get('authorships', [])]
        if len(authors_list) > 3:
            authors = ", ".join(authors_list[:3]) + ", et al."
        else:
            authors = ", ".join(authors_list)
        
        # Extract other required fields, handling None values
        title = data.get('title', "Unknown Title")
        year = data.get('publication_year', "n.d.")
        try:
            journal = data.get('primary_location', {}).get('source', {}).get('display_name', None)
        except:
            journal = None

        volume = data.get('biblio', {}).get('volume', None)
        issue = data.get('biblio', {}).get('issue', None)
        pages = f"{data.get('biblio', {}).get('first_page', '')}-{data.get('biblio', {}).get('last_page', '')}".strip("-")
        doi = data.get('doi', None)
        if doi:
            doi = doi.replace("https://doi.org/", "")
        
        # Construct citation parts
        citation_parts = [
            f"{authors} ({year}).",
            f"{title}.",
            f"{journal}," if journal else "",
            f"{volume}" if volume or issue else "",
            f"{pages}." if pages else "",
            f"doi: {doi}" if doi else ""
        ]
        
        # Join non-empty parts with spaces
        citation = " ".join(part for part in citation_parts if part)

        return citation
    
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
            if all(fields[field] is not None for field in combination)
        ]
    
    def get_highest_true_position(self, outputs, inputs):
        # Iterate through the outputs to collect indices and scores of 'True' labels
        true_scores = [
            (index, result[0]['score'])
            for index, result in enumerate(outputs)
            if result[0]['label'] == 'True'
        ]
        
        # Find the entry with the highest score or return None if no 'True' labels exist
        if true_scores:
            pos =  max(true_scores, key=lambda x: x[1])[0]  # Return the index with the highest score
            return inputs[pos]
        return None
    
    def search_api(self, ner_entities):
        """
        Search API using flexible field combinations.

        :param ner_entities: Dictionary containing extracted NER entities
        :param source_url: Base URL for the API
        :return: JSON response or None if no results found
        """
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
            "doi": ner_entities.get("DOI", [None])[0].replace("https://doi.org/", "").replace(' ','') if ner_entities.get("DOI") else None,
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

        for combination in filtered_combinations:
            # Build query parameters dynamically based on available fields
            query_params = []
            for field in combination:
                field_search = field.split('-')[0]
                value = fields[field_search]
                if value:
                    if field == "title":
                        query_params.append(f"title.search:{value}")
                    
                    if field == "doi":
                        query_params.append(f"doi:{value}")

                    if field == "title-half":
                        words = value.split()
                        mid_point = len(words) // 2
                        first_half = " ".join(words[:mid_point])
                        second_half = " ".join(words[mid_point:])
                        query_params.append(f"title.search:{first_half}|{second_half}")
                        
                    if field == "author":
                        query_params.append(f"raw_author_name.search:{value}")
                    if field == "year":
                        query_params.append(f"publication_year:{value}")
                    if field == "journal":
                        query_params.append(f"locations.source.id:{value}")
                    if field == "first_page":
                        query_params.append(f"biblio.first_page:{value}")
                    if field == "last_page":
                        query_params.append(f"biblio.last_page:{value}")
                    if field == "volume":
                        query_params.append(f"biblio.volume:{value}")
                    if field == "issue":
                        query_params.append(f"biblio.issue:{value}")

            # Combine query parameters
            query_string = ",".join(query_params)
            api_url = f"{base_url}?filter={query_string}".replace(',,',',')

            # Make the API call
            response = requests.get(api_url)
            if response.status_code == 200:
                results = response.json().get("results", [])

                candidates+=results
                if len(results)==1:  # Return results if found
                    return results
                
                if len(candidates)>1:
                    return candidates
                        
        return candidates

    def link_citation(self, citation, results='simple'):

        prescreening_style = self.prescreening_pipeline(citation)
        if prescreening_style[0]['label'] == 'False':  # Assuming the label structure
            return {"error": "This text is not a citation. Please introduce a valid citation."}


        ner_entities = self.process_ner_entities(citation)
        pubs = self.search_api(ner_entities)
        cits = [self.generate_apa_citation(pub) for pub in pubs]
        
        if len(cits)==1:
            pairwise = [self.select_pipeline(f"{citation} [SEP] {cit}") for cit in cits]
            if pairwise[0][0]['label']=='True':
                if results=='simple':
                    return {'result':cits[0], 'score':pairwise[0][0]['score'],'id':pubs[0]['id']}
                if results=='advanced':
                    return {'result':cits[0], 'score':pairwise[0][0]['score'], 'full-publication':pubs[0]}
            else:
                if results=='simple':
                    return {'result':cits[0], 'score':False,'id':pubs[0]['id']}
                if results=='advanced':
                    return {'result':cits[0], 'score':False, 'full-publication':pubs[0]}
        
        if len(cits)>1:
            outputs = [self.select_pipeline(f"{citation} [SEP] {cit}") for cit in cits]
            get_reranked_pubs = self.get_highest_true_position(outputs, pubs)
            if get_reranked_pubs!=None:
                if results=='simple':
                    return {'result':self.generate_apa_citation(get_reranked_pubs), 'score':'to implement','id':get_reranked_pubs['id']}
                if results=='advanced':
                    return {'result':self.generate_apa_citation(get_reranked_pubs), 'score':'to implement', 'id':get_reranked_pubs['id'],'full-publication':get_reranked_pubs} 
            else:
                return {'result':None}
                    
        else:
            return {'result':None}
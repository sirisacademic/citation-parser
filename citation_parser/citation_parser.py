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
import time
from .search import search_api
from .search import citation_formatter
import xml.etree.ElementTree as ET
import json
from collections import Counter

class CitationParser:
    def __init__(self, ner_model_path="SIRIS-Lab/citation-parser-ENTITY", select_model_path="SIRIS-Lab/citation-parser-SELECT",prescreening_model_path="SIRIS-Lab/citation-parser-TYPE", device="cpu"):
        # Initialize the NER pipeline
        self.ner_pipeline = pipeline(
            "ner",
            model=AutoModelForTokenClassification.from_pretrained(ner_model_path),
            tokenizer=AutoTokenizer.from_pretrained(ner_model_path),
            aggregation_strategy="simple",
            device=device
        )

        # Initialize the select pipeline
        self.select_pipeline = pipeline(
            "text-classification",
            model=AutoModelForSequenceClassification.from_pretrained(select_model_path),
            tokenizer=AutoTokenizer.from_pretrained(select_model_path),
            device=device
        )
        # Initialize the prescreening pipeline
        self.prescreening_pipeline = pipeline(
            "text-classification",
            model=AutoModelForSequenceClassification.from_pretrained(prescreening_model_path),
            tokenizer=AutoTokenizer.from_pretrained(prescreening_model_path),
            device=device
        )

        # Initialize the search API
        self.searcher = search_api.SearchAPI()


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
    
    def generate_apa_citation(self, data, api = 'openalex'):

        formatter = citation_formatter.CitationFormatterFactory.get_formatter(api)

        return formatter.generate_apa_citation(data)
    
    def get_highest_true_position(self, outputs, inputs):
        # Iterate through the outputs to collect indices and scores of 'True' labels
        true_scores = [
            (index, result[0]['score']) if result[0]['label'] == True else (index, 0)
            for index, result in enumerate(outputs)
            ]
        
        # Find the entry with the highest score or return None if no 'True' labels exist
        if true_scores:
            # Get the index with the highest score
            highest_index = max(true_scores, key=lambda x: x[1])[0]
            
            # Return both the input at that position and the highest score
            return inputs[highest_index], true_scores[highest_index][1]
        
        return None, None  # Return None for both input and score if no 'True' labels exist
    
    def search_api(self, ner_entities, api="openalex"):
        """
        Search API using flexible field combinations.

        :param ner_entities: Dictionary containing extracted NER entities
        :param source_url: Base URL for the API
        :return: JSON response or None if no results found
        """

        candidates = self.searcher.search_api(ner_entities, api=api)
                        
        return candidates
    
    def extract_id(self, publication, api_target):
            """Helper function to extract the ID based on the API."""
            if api_target == 'openalex':
                return publication.get('id')
            elif api_target == 'openaire':
                #https://api.openaire.eu/search/publications?openairePublicationID=doi_dedup___::99fc3bb794e0789acc3f5a7195a1c9c1&format=json
                return publication.get('header',{}).get('dri:objIdentifier',{}).get('$',None)
            elif api_target=='pubmed':
                root = ET.fromstring(publication)
                pmid = root.findtext("PubmedArticle/MedlineCitation/PMID", None)
                if pmid:
                    pmid = pmid#f'https://pubmed.ncbi.nlm.nih.gov/{pmid}'
                return pmid
            elif api_target=='crossref':
                return publication.get('DOI')
            elif api_target=='hal':
                return publication.get('halId_s')
            return None

    def link_citation(self, citation, output='simple', api_target='openalex'):
        """
        Links a citation to its corresponding data using the specified API (OpenAlex or OpenAIRE).
        
        :param citation: The input citation text.
        :param results: The output format ('simple' or 'advanced').
        :param api: The selected API ('openalex' or 'openaire').
        :return: A dictionary with the linked citation result or an error message.
        """

        # Prescreening step to check validity of the citation
        prescreening_style = self.prescreening_pipeline(citation)
        if prescreening_style[0]['label'] == 'False':  # Assuming the label structure
            return {"error": "This text is not a citation. Please introduce a valid citation."}

        ner_entities = self.process_ner_entities(citation)
        pubs = self.search_api(ner_entities, api=api_target)
        # remove duplicates
        #pubs = [json.loads(t) for t in {json.dumps(d, sort_keys=True) for d in pubs}]

        cits = [self.generate_apa_citation(pub,api=api_target) for pub in pubs]
        
        if len(cits)==1:
            pairwise = [self.select_pipeline(f"{citation} [SEP] {cit}") for cit in cits]
            if pairwise[0][0]['label']==True:
                pub_id = self.extract_id(pubs[0], api_target)
                if output=='simple':
                    return {'result':cits[0], 'score':pairwise[0][0]['score'],'id':pub_id}
                if output=='advanced':
                    return {'result':cits[0], 'score':pairwise[0][0]['score'],  'id':pub_id, 'full-publication':pubs[0]}
            else:
                pub_id = self.extract_id(pubs[0], api_target)
                if output=='simple':
                    return {'result':cits[0], 'score':False,'id':pub_id}
                if output=='advanced':
                    return {'result':cits[0], 'score':False, 'full-publication':pubs[0]}
        
        if len(cits)>1:
            outputs = [self.select_pipeline(f"{citation} [SEP] {cit}") for cit in cits]
            get_reranked_pub, score = self.get_highest_true_position(outputs, pubs)
            if get_reranked_pub!=None:
                pub_id = self.extract_id(get_reranked_pub, api_target)
                if output=='simple':
                    return {'result':self.generate_apa_citation(get_reranked_pub, api = api_target), 'score':score,'id':pub_id}
                if output=='advanced':
                    return {'result':self.generate_apa_citation(get_reranked_pub, api = api_target), 'score':score, 'id':pub_id,'full-publication':get_reranked_pub} 
            else:
                return {}
                    
        else:
            return {}
        
    def get_doi(self, publication, api_target):
        """
        Extracts the DOI from a publication data dictionary based on the API source.

        :param publication: Dictionary containing the publication data.
        :param api_target: The source API name (e.g., 'openalex', 'openaire', 'pubmed', 'crossref', 'hal').
        :return: Extracted DOI or None if not found.
        """
        doi = None

        if api_target == "openalex":
            doi = publication.get("doi", None)
            if doi:
                doi = doi.replace("https://doi.org/", "")

        elif api_target == "openaire":
            # Extract DOI
            identifiers = publication.get('metadata', {}).get('oaf:entity', {}).get('oaf:result', {}).get('pid', [])
            # Ensure identifiers is always a list
            if isinstance(identifiers, dict):  
                identifiers = [identifiers]  
            doi = None
            for identifier in identifiers:
                if identifier.get('@classid') == 'doi':
                    doi =identifier.get('$',None)
                    doi = doi.replace("https://doi.org/", "")

        elif api_target == "pubmed":
                root = ET.fromstring(publication)
                article = root.find("PubmedArticle/MedlineCitation/Article")
                doi_element = article.findtext("ELocationID[@EIdType='doi']", None)
                if doi_element is not None:
                    doi = doi_element.text

        elif api_target == "crossref":
            doi = publication.get("DOI", None)
            if doi:
                doi = doi.replace("https://doi.org/", "")

        elif api_target == "hal":
            doi = publication.get("doiId_s", None)

        return doi

    def link_citation_ensemble(self, citation, output='simple', api_targets=['openalex', 'openaire', 'pubmed', 'crossref', 'hal']):
        """
        Calls link_citation on multiple API targets and selects the most frequent DOI for cross-validation.

        :param citation: The input citation text.
        :param output: Output format ('simple' or 'advanced').
        :param api_targets: List of API targets to query.
        :return: A dictionary with the best DOI and extract IDs from all sources.
        """
        doi_counter = Counter()
        extract_ids = {}  # Stores extract_id for each source
        missing_sources = []  # Sources that didn't return a DOI

        for api in api_targets:

            print(api)
            output = self.link_citation(citation, output="advanced", api_target=api)
            
            print(output)

            if not output or not isinstance(output, dict) or 'full-publication' not in output:
                # If output is empty, not a dictionary, or missing 'full-publication', handle appropriately
                doi = None  # Ensure no DOI processing occurs

                missing_sources.append(api)  # Keep track of sources that failed to find a DOI
            else:
                # If the output is valid and contains 'full-publication'
                try:
                    doi = self.get_doi(output['full-publication'], api)
                    
                    if doi:  # Only consider non-None DOIs
                        doi_counter[doi] += 1
                        extract_id = self.extract_id(output['full-publication'], api_target=api)  # Extract the native ID
                        
                        if extract_id:
                            extract_ids[api] = extract_id  # Store the extract_id
                except KeyError as e:
                    # Handle the case where 'full-publication' might be missing despite the checks
                    print(f"KeyError: {e} while processing {api}.")
                    missing_sources.append(api)  # Append to missing sources

        if not doi_counter:
            return {"doi": None, "extract_ids": {}}

        # Get the most common DOI
        best_doi, _ = doi_counter.most_common(1)[0]
        # add "https://doi.org/" if this prefix is not before the DOI
        
        # Query missing sources using the best DOI
        for api in missing_sources:
            pubs = self.search_api({'DOI': [best_doi]}, api=api)

            if len(pubs)>0:
                pub_id = self.extract_id(pubs[0], api)

            if output:
                extract_id = pub_id
                if extract_id:
                    extract_ids[api] = extract_id  # Store the native ID

            else:
                extract_ids[api] = None

        if not best_doi.startswith("https://doi.org/"):
            best_doi = "https://doi.org/" + best_doi


        return {
            "doi": best_doi,
            "extract_ids": extract_ids
        }

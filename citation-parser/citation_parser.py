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
from search import search_api
from search import citation_formatter
import xml.etree.ElementTree as ET

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
            device=device
        )

        self.prescreening_pipeline = pipeline(
            "text-classification",
            model=AutoModelForSequenceClassification.from_pretrained(prescreening_model_path),
            tokenizer=AutoTokenizer.from_pretrained(prescreening_model_path),
            device=device
        )

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
            (index, result[0]['score'])
            for index, result in enumerate(outputs)
            if result[0]['label'] == 'True'
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

    def link_citation(self, citation, output='simple', api_target='openalex'):
        """
        Links a citation to its corresponding data using the specified API (OpenAlex or OpenAIRE).
        
        :param citation: The input citation text.
        :param results: The output format ('simple' or 'advanced').
        :param api: The selected API ('openalex' or 'openaire').
        :return: A dictionary with the linked citation result or an error message.
        """
        def extract_id(publication, api_target):
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
                    pmid = f'https://pubmed.ncbi.nlm.nih.gov/{pmid}'
                return pmid
            return None

        # Prescreening step to check validity of the citation
        prescreening_style = self.prescreening_pipeline(citation)
        if prescreening_style[0]['label'] == 'False':  # Assuming the label structure
            return {"error": "This text is not a citation. Please introduce a valid citation."}

        ner_entities = self.process_ner_entities(citation)
        pubs = self.search_api(ner_entities, api=api_target)
        cits = [self.generate_apa_citation(pub,api=api_target) for pub in pubs]
        
        if len(cits)==1:
            pairwise = [self.select_pipeline(f"{citation} [SEP] {cit}") for cit in cits]
            if pairwise[0][0]['label']=='True':
                pub_id = extract_id(pubs[0], api_target)
                if output=='simple':
                    return {'result':cits[0], 'score':pairwise[0][0]['score'],'id':pub_id}
                if output=='advanced':
                    return {'result':cits[0], 'score':pairwise[0][0]['score'],  'id':pub_id, 'full-publication':pubs[0]}
            else:
                if output=='simple':
                    return {'result':cits[0], 'score':False,'id':pub_id}
                if output=='advanced':
                    return {'result':cits[0], 'score':False, 'full-publication':pubs[0]}
        
        if len(cits)>1:
            outputs = [self.select_pipeline(f"{citation} [SEP] {cit}") for cit in cits]
            get_reranked_pub, score = self.get_highest_true_position(outputs, pubs)
            if get_reranked_pub!=None:
                pub_id = extract_id(get_reranked_pub, api_target)
                if output=='simple':
                    return {'result':self.generate_apa_citation(get_reranked_pub, api = api_target), 'score':score,'id':pub_id}
                if output=='advanced':
                    return {'result':self.generate_apa_citation(get_reranked_pub, api = api_target), 'score':score, 'id':pub_id,'full-publication':get_reranked_pub} 
            else:
                return {'result':None}
                    
        else:
            return {'result':None}
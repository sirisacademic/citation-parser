# references_tractor.py

# Import necessary modules
import re
import time
import json
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

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


class ReferencesTractor:
    """
    Class to extract citation entities, search bibliographic databases, 
    and generate and link citations to their canonical records.
    """

    def __init__(
        self,
        ner_model_path: str = "SIRIS-Lab/citation-parser-ENTITY",
        select_model_path: str = "SIRIS-Lab/citation-parser-SELECT",
        prescreening_model_path: str = "SIRIS-Lab/citation-parser-TYPE",
        span_model_path: str = "SIRIS-Lab/citation-parser-SPAN",
        device: Union[int, str] = "cpu",
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

    def process_ner_entities(self, citation: str) -> Dict[str, List[str]]:
        # Extract named entities from the citation using the NER pipeline
        output = self.ner_pipeline(citation)
        entities = {}
        for entity in output:
            key = entity.get("entity_group")
            entities.setdefault(key, []).append(entity.get("word", ""))
        return entities

    def generate_apa_citation(self, data: dict, api: str = "openalex") -> str:
        # Format a citation from retrieved metadata in APA style
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

    def search_api(self, ner_entities: Dict[str, List[str]], api: str = "openalex") -> List[dict]:
        # Search a bibliographic API using extracted NER entities
        return self.searcher.search_api(ner_entities, api=api)

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
            root = ET.fromstring(publication)
            return root.findtext("PubmedArticle/MedlineCitation/PMID")
        elif api == "crossref":
            return publication.get("DOI")
        elif api == "hal":
            return publication.get("halId_s")
        return None

    def extract_doi(self, publication: dict, api: str) -> Optional[str]:
        # Extract the DOI depending on the API source
        if api == "openalex":
            doi = publication.get("doi")
            if isinstance(doi, str):
                return doi.replace("https://doi.org/", "")
            return None
        elif api == "openaire":
            identifiers = publication.get('metadata', {}).get('oaf:entity', {}).get('oaf:result', {}).get('pid', [])
            if isinstance(identifiers, dict):
                identifiers = [identifiers]
            for pid in identifiers:
                if pid.get("@classid") == "doi":
                    return pid.get("$", "").replace("https://doi.org/", "")
        elif api == "pubmed":
            root = ET.fromstring(publication)
            return root.findtext("PubmedArticle/MedlineCitation/Article/ELocationID[@EIdType='doi']")
        elif api == "crossref":
            return publication.get("DOI")
        elif api == "hal":
            return publication.get("doiId_s")
        return None

    def link_citation(self, citation: str, output: str = 'simple', api_target: str = 'openalex') -> Dict[str, Any]:
        """
        Main function to process a citation string:
        - Check if it's a valid citation
        - Extract entities
        - Search target API
        - Format results and rank them
        """
        # Prescreen input to ensure it's likely a citation
        if self.prescreening_pipeline(citation)[0]["label"] == "False":
            return {"error": "This text is not a citation. Please introduce a valid citation."}

        ner_entities = self.process_ner_entities(citation)
        pubs = self.search_api(ner_entities, api=api_target)

        if not pubs:
            return {}

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
            return self._format_result(cits[0], selected_score, pub_id, pub_doi, url, pub, output, api_target)

        # Choose the most likely correct match using classification scores
        reranked_pub, best_score = self.get_highest_true_position(pairwise_scores, pubs)
        if reranked_pub:
            pub_id = self.extract_id(reranked_pub, api_target)
            pub_doi = self.extract_doi(reranked_pub, api_target)
            url = self.get_uri(pub_id, pub_doi, api_target)
            formatted_cit = self.generate_apa_citation(reranked_pub, api=api_target)
            return self._format_result(formatted_cit, {"score": best_score}, pub_id, pub_doi, url, reranked_pub, output, api_target)

        return {}

    def _format_result(
        self, citation: str, score_data: dict, pub_id: Optional[str], doi: Optional[str],
        url: Optional[str], pub: dict, output: str, api_target: str
    ) -> Dict[str, Any]:
        # Helper to format the output result with optional full metadata
        result = {
            "result": citation,
            "score": score_data.get("score", False),
            f"{api_target}_id": pub_id,
            "doi": doi,
            "url": url
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
        Selects the most agreed-upon DOI among sources.
        """
        doi_counter = Counter()
        extract_ids = {}
        missing_sources = []

        # Try to link using each API
        for api in api_targets:
            try:
                res = self.link_citation(citation, output="advanced", api_target=api)
                doi = res.get("doi")
                if doi:
                    doi_counter[doi] += 1
                    extract_ids[api] = res.get(f"{api}_id", None)
                else:
                    missing_sources.append(api)
            except Exception as e:
                print(f"Error processing API {api}: {e}")
                missing_sources.append(api)

        if not doi_counter:
            return {"doi": None, "external_ids": {}}

        # Choose DOI with most agreement
        best_doi, _ = doi_counter.most_common(1)[0]

        # Attempt to backfill missing sources with best DOI
        for api in missing_sources:
            pubs = self.search_api({'DOI': [best_doi]}, api=api)
            pub_id = self.extract_id(pubs[0], api) if pubs else None
            extract_ids[f"{api}_id"] = pub_id

        return {
            "doi": best_doi,
            "external_ids": extract_ids
        }
    

    def extract_and_link_from_text(self, text: str, api_target: str = 'openalex') -> Dict[str, Dict[str, Any]]:
        """
        Extract citation entities from the provided text and link them to bibliographic data.
        
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


        # Step 2: For each entity group, extract each entity and link it
        for entity in long_entities:
            # Step 3: Link the entity to a citation using the link_citation method
            linked_data = self.link_citation(entity, api_target=api_target)

            # Step 4: Add the linked citation data to the result dictionary
            linked_entities[entity] = linked_data

        return linked_entities

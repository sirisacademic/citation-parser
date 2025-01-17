# citation-parser 🖇️🧻🎓
Citation Parser is a Python package designed to process raw citation texts and link them to scholarly knowledge graphs like OpenAlex, OpenAIRE, and PubMed. It leverages advanced natural language processing techniques powered by three small, fine-tuned language models to deliver accurate and robust citation parsing and linking.

## 🔨 Key steps of the tools:

Citation Parser follows a structured multi-step process to achieve accurate citation linking:

1. **Pre-Screening**: a classification model based on `distilbert/distilbert-base-multilingual-cased` determines whether the given text is a valid citation or not.
![image](https://github.com/user-attachments/assets/39748a7d-192c-4787-aa35-5827c8f0c9bf)
3. **Citation Parsing (NER)**: sophisticated Named Entity Recognition (NER) extracts key fields from the citation. The citation is parsed into structured fields using a fine-tuned Named Entity Recognition model. The extracted fields can include:
    - `TITLE`, `AUTHORS`, `VOLUME`, `ISSUE`, `YEAR`, `DOI`, `ISSN`, `ISBN`, `FIRST_PAGE`, `LAST_PAGE`, `JOURNAL`, and `EDITOR`.

![image](https://github.com/user-attachments/assets/8cb6a723-5373-4685-80c0-4a09ae47ad06)

3. **Candidate Identification**: a set of carefully crafted queries to the OpenAlex API retrieves one or more candidate publications based on the parsed citation fields. The parsed information is used to construct a series of queries to the OpenAlex API, retrieving one or more potential matches for the citation.
4. **Pairwise Classification**: a pairwise classification model predicts the likelihood of the identified candidates matching the original citation. This model is fine-tuned on a dataset of citation pairs in the format: `"CITATION 1 [SEP] CITATION 2"`. If multiple candidates are retrieved, the publication with the highest likelihood score is returned.

The best-matching candidate is selected based on the likelihood score and returned as the final linked publication.

## 💻 Installation

```bash
pip install git+https://github.com/sirisacademic/citation-parser.git
```

## Usage

Here’s a basic example of how to use Citation Parser:

```python
from citation_parser import CitationParser

# Initialize the parser
parser = CitationParser()

# Raw citation text
citation = "MURAKAMI, H等: 'Unique thermal behavior of acrylic PSAs bearing long alkyl side groups and crosslinked by aluminum chelate', 《EUROPEAN POLYMER JOURNAL》"

# Parse and link the citation
result = parser.link_citation(citation, api_target = "openalex", output = 'simple')
```

The output would look like this:
```python
{'result': 'Hiroto Murakami, Keisuke Futashima, Minoru Nanchi, et al. (2010). Unique thermal behavior of acrylic PSAs bearing long alkyl side groups and crosslinked by aluminum chelate. European Polymer Journal, 47 378-384. doi: 10.1016/j.eurpolymj.2010.12.012',
 'score': 0.9997150301933289,
 'id': 'https://openalex.org/W2082866977'}
```

## Parameters
- **api_target**: Specifcy knowledge graphs to query. Options include:
    - `openalex` - [*default*] Links to OpenAlex
    - `openaire` - Links to OpenAIRE
    - `pubmed` - Links to PubMed
- **output**: Specifies the type of result returned:
    - `simple` – Returns a concise, structured citation match.
    - `full` – Returns a detailed, full citation with additional metadata.
- **device**:
    - `cpu`: Utilises the CPU for model inference, suitable for environments without GPU support. Recommended for smaller workloads or when GPU is unavailable.
    - `cuda`: Utilises GPUs via CUDA for faster inference. Recommended for environments where GPUs are available and high performance is required.

### Returns
- result: *dict* with the folling attributues:
    - `result`: Citation from the linked source.
    - `score`: Similarity score with the input citatio
    - `id`: `publication_id` in the target Scholarly Knowledge Graph (OpenAlex, OpenAIRE, or PubMed)
    - (if `output='full'`) `full-publication`: Publication object from the target API

## Dependencies

Ensure you have all necessary dependencies installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```

## Applications

Citation Parser is ideal for:
- Automated metadata enrichment: extract structured metadata from raw citation texts.
- Citation Validation: verify the correctness of citations in manuscripts.
- Scholarly Database Integration: link citations to knowledge graphs like OpenAlex and OpenAIRE.

## Models
- 🤗 **TYPE model available at: https://huggingface.co/SIRIS-Lab/citation-parser-TYPE**
- 🤗 **NER model available at: https://huggingface.co/SIRIS-Lab/citation-parser-ENTITY**
- 🤗 **SELECT model available at: https://huggingface.co/SIRIS-Lab/citation-parser-SELECT**

### 📊 Model Performance Metrics
The performance of each model used in the Citation Parser is evaluated using the F1 score. Below are the F1 scores for each of the key models involved in citation parsing and linking:

| Model            | F1 Score |
|---------------------|----------|
| **TYPE Model (Citation Pre-screening)**  | 0.941638 |
| **NER Model (Citation Citation)**  | 0.949772 |
 **SELECT Model (Candidate Selection)**  | 0.846972 |

# Future features
- Improved candidate retrieval: advanced query strategies for ambiguous or incomplete citations.
- Translation to multilingual input to do multiple searches in both input language and English

## 📫 Contact

For further information, please contact <nicolau.duransilva@sirisacademic.com>.

## ⚖️ License

This work is distributed under a [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

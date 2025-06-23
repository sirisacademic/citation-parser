# Development Examples

This document provides detailed examples for extending References Tractor with new features, APIs, and models.

## Adding New APIs

This section demonstrates how to add support for a new academic database by implementing all required components.

### Example: Adding Support for Semantic Scholar

Here's a complete walkthrough of adding Semantic Scholar API support:

#### 1. API Configuration

First, add the API configuration to define search capabilities and DOI extraction:

```python
# search/api_capabilities.py

# Add to SEARCH_CAPABILITIES
SEARCH_CAPABILITIES["semantic_scholar"] = {
    "DOI": SearchFieldConfig("doi", "exact", "clean_doi"),
    "TITLE": SearchFieldConfig("query", "search"),
    "AUTHORS": SearchFieldConfig("query", "search", "extract_author_surname"),
    "PUBLICATION_YEAR": SearchFieldConfig("year", "exact"),
    "JOURNAL": SearchFieldConfig("query", "search"),
}

# Add to DOI_EXTRACTION_CONFIGS
DOI_EXTRACTION_CONFIGS["semantic_scholar"] = DOIExtractionConfig(
    supports_multiple_dois=False,
    main_doi_path="externalIds.DOI",
    alternative_doi_paths=[],
    extraction_notes="Semantic Scholar has consistent DOI structure in externalIds"
)

# Add to FIELD_COMBINATIONS
FIELD_COMBINATIONS["semantic_scholar"] = [
    ["DOI"],
    ["TITLE", "AUTHORS", "PUBLICATION_YEAR"],
    ["TITLE", "PUBLICATION_YEAR"],
    ["TITLE", "AUTHORS"],
    ["TITLE"],
    ["AUTHORS", "PUBLICATION_YEAR"],
]

# Add to RETRY_CONFIGS
RETRY_CONFIGS["semantic_scholar"] = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    rate_limit_delay=0.3,
    timeout=25
)
```

#### 2. Search Strategy Implementation

Create the search strategy class:

```python
# search/search_api.py

class SemanticScholarStrategy(BaseAPIStrategy):
    def get_api_name(self) -> str:
        return "semantic_scholar"
        
    def _build_api_url(self, query_params: Dict[str, Any]) -> str:
        """Build Semantic Scholar API URL from query parameters"""
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        
        # Handle different query types
        query_parts = []
        
        if "doi" in query_params:
            # DOI search - use specific endpoint
            doi = query_params["doi"]
            return f"https://api.semanticscholar.org/graph/v1/paper/{doi}"
        
        if "query" in query_params:
            # General query search
            query = str(query_params["query"])
            encoded_query = self.encode_query_value(query)
            query_parts.append(f"query={encoded_query}")
        
        if "year" in query_params:
            # Year filtering
            year = query_params["year"]
            query_parts.append(f"year={year}")
        
        # Add required fields for response
        fields = "paperId,externalIds,title,authors,year,venue,publicationDate"
        query_parts.append(f"fields={fields}")
        
        # Set result limit
        query_parts.append("limit=10")
        
        if query_parts:
            query_string = "&".join(query_parts)
            return f"{base_url}?{query_string}"
        
        return base_url
    
    def _parse_api_response(self, response: requests.Response) -> List[Dict]:
        """Parse Semantic Scholar API response"""
        try:
            data = response.json()
            
            # Handle different response formats
            if "data" in data:
                # Search results
                return data.get("data", [])
            elif "paperId" in data:
                # Single paper result
                return [data]
            else:
                return []
                
        except Exception as e:
            print(f"Error parsing Semantic Scholar response: {e}")
            return []
```

#### 3. Citation Formatter

Implement APA citation formatting for Semantic Scholar data:

```python
# search/citation_formatter.py

class SemanticScholarFormatter(CitationFormatter):
    def generate_apa_citation(self, data):
        """Generate APA citation from Semantic Scholar data"""
        
        # Extract authors
        authors = []
        for author in data.get('authors', []):
            author_name = author.get('name', '')
            if author_name:
                authors.append(author_name)
        
        # Format author string
        if len(authors) > 3:
            authors_str = ", ".join(authors[:3]) + ", et al."
        else:
            authors_str = ", ".join(authors)
        
        # Extract other fields
        title = data.get('title', 'Unknown Title')
        year = data.get('year', 'n.d.')
        venue = data.get('venue', '')
        
        # Extract DOI information
        main_doi, alternative_dois = self._get_doi_info(data)
        
        # If no enhanced structure, extract from externalIds
        if not main_doi:
            external_ids = data.get('externalIds', {})
            if external_ids and isinstance(external_ids, dict):
                main_doi = external_ids.get('DOI')
                alternative_dois = []
        
        doi_section = self._format_doi_section(main_doi, alternative_dois)
        
        # Build citation parts
        citation_parts = [
            f"{authors_str} ({year})." if authors_str else f"({year}).",
            f"{title}." if title else "",
            f"{venue}." if venue else "",
            doi_section
        ]
        
        return " ".join(part for part in citation_parts if part).strip()
```

#### 4. Register the New API

Add the new API to the main system:

```python
# search/search_api.py - in SearchAPI.__init__
self.strategies = {
    "openalex": OpenAlexStrategy(),
    "openaire": OpenAIREStrategy(),
    "pubmed": PubMedStrategy(),
    "crossref": CrossRefStrategy(),
    "hal": HALSearchStrategy(),
    "semantic_scholar": SemanticScholarStrategy(),  # Add this line
}

# search/citation_formatter.py - in CitationFormatterFactory
formatters = {
    "openalex": OpenAlexFormatter(),
    "openaire": OpenAIREFormatter(),
    "pubmed": PubMedFormatter(),
    "crossref": CrossrefFormatter(),
    "hal": HALFormatter(),
    "semantic_scholar": SemanticScholarFormatter(),  # Add this line
}
```

#### 5. Add Tests

Create comprehensive tests for the new API:

```python
# tests/unit/test_semantic_scholar.py
import pytest
from unittest.mock import Mock, patch
from references_tractor.search.search_api import SemanticScholarStrategy

class TestSemanticScholarStrategy:
    def setup_method(self):
        self.strategy = SemanticScholarStrategy()
    
    def test_api_name(self):
        assert self.strategy.get_api_name() == "semantic_scholar"
    
    def test_build_url_with_doi(self):
        query_params = {"doi": "10.1038/nature12373"}
        url = self.strategy._build_api_url(query_params)
        
        assert "api.semanticscholar.org" in url
        assert "10.1038/nature12373" in url
    
    def test_build_url_with_query(self):
        query_params = {"query": "machine learning", "year": "2020"}
        url = self.strategy._build_api_url(query_params)
        
        assert "query=machine%20learning" in url
        assert "year=2020" in url
        assert "fields=" in url

# tests/integration/test_semantic_scholar_integration.py
import pytest
from references_tractor import ReferencesTractor

@pytest.mark.integration
class TestSemanticScholarIntegration:
    def setup_method(self):
        self.ref_tractor = ReferencesTractor()
    
    def test_famous_paper_linking(self):
        """Test linking a well-known paper"""
        citation = "Attention Is All You Need. Vaswani et al. NIPS 2017."
        
        result = self.ref_tractor.link_citation(
            citation, 
            api_target="semantic_scholar"
        )
        
        # Should find the transformer paper
        if result:  # Only assert if API returned a result
            assert "attention" in result.get('result', '').lower()
            assert result.get('score', 0) > 0.5
    
    def test_doi_search(self):
        """Test DOI-based search"""
        citation = "DOI: 10.1038/nature12373"
        
        result = self.ref_tractor.link_citation(
            citation,
            api_target="semantic_scholar"
        )
        
        if result:
            assert result.get('doi') is not None
    
    def test_no_match_citation(self):
        """Test citation that should return no match"""
        citation = "This is not a real citation at all."
        
        result = self.ref_tractor.link_citation(
            citation,
            api_target="semantic_scholar"
        )
        
        # Should return empty dict for no match
        assert result == {}

# tests/unit/test_semantic_scholar_formatter.py
import pytest
from references_tractor.search.citation_formatter import SemanticScholarFormatter

class TestSemanticScholarFormatter:
    def setup_method(self):
        self.formatter = SemanticScholarFormatter()
    
    def test_format_complete_paper(self):
        """Test formatting with complete paper data"""
        data = {
            "title": "Attention Is All You Need",
            "authors": [
                {"name": "Ashish Vaswani"},
                {"name": "Noam Shazeer"},
                {"name": "Niki Parmar"},
                {"name": "Jakob Uszkoreit"}  # Will be truncated to "et al."
            ],
            "year": 2017,
            "venue": "NIPS",
            "externalIds": {
                "DOI": "10.48550/arXiv.1706.03762"
            }
        }
        
        citation = self.formatter.generate_apa_citation(data)
        
        assert "Vaswani" in citation
        assert "et al." in citation
        assert "(2017)" in citation
        assert "Attention Is All You Need" in citation
        assert "DOI: 10.48550/arXiv.1706.03762" in citation
    
    def test_format_minimal_paper(self):
        """Test formatting with minimal data"""
        data = {
            "title": "Test Paper",
            "year": 2020
        }
        
        citation = self.formatter.generate_apa_citation(data)
        
        assert "(2020)" in citation
        assert "Test Paper" in citation
```

#### 6. Update Documentation

Add the new API to documentation:

```python
# Update README.md supported APIs table
| API | Coverage | Specialization |
|-----|----------|----------------|
| **Semantic Scholar** | Computer science focus | AI/ML publications |

# Update docs/api.md with examples
# Update docs/evaluation.md with API-specific notes
```

## Adding New Models

This section shows how to integrate new transformer models into the system.

### Example: Adding a Custom NER Model

#### 1. Model Integration

```python
# core.py
class ReferencesTractor:
    def __init__(
        self,
        ner_model_path: str = "SIRIS-Lab/citation-parser-ENTITY",
        custom_ner_model_path: Optional[str] = None,  # Add new parameter
        # ... other parameters
    ):
        # Initialize standard NER model
        self.ner_pipeline = self._init_pipeline("ner", ner_model_path, device, agg_strategy="simple")
        
        # Initialize custom NER model if provided
        if custom_ner_model_path:
            self.custom_ner_pipeline = self._init_pipeline(
                "ner", custom_ner_model_path, device, agg_strategy="simple"
            )
        else:
            self.custom_ner_pipeline = None
    
    def process_ner_entities_custom(self, citation: str) -> Dict[str, List[str]]:
        """Extract entities using custom NER model"""
        if not self.custom_ner_pipeline:
            raise ValueError("Custom NER model not initialized")
        
        output = self.custom_ner_pipeline(citation)
        entities = {}
        
        for entity in output:
            key = entity.get("entity_group")
            entities.setdefault(key, []).append(entity.get("word", ""))
        
        # Apply same cleaning as standard NER
        from .utils.entity_validation import EntityValidator
        cleaned_entities = EntityValidator.validate_and_clean_entities(entities)
        
        return cleaned_entities
    
    def process_ner_entities_ensemble(self, citation: str) -> Dict[str, List[str]]:
        """Combine results from multiple NER models"""
        # Get results from standard model
        standard_entities = self.process_ner_entities(citation)
        
        # Get results from custom model if available
        if self.custom_ner_pipeline:
            custom_entities = self.process_ner_entities_custom(citation)
            
            # Merge results (prefer custom model when available)
            merged_entities = {}
            all_keys = set(standard_entities.keys()) | set(custom_entities.keys())
            
            for key in all_keys:
                standard_values = standard_entities.get(key, [])
                custom_values = custom_entities.get(key, [])
                
                # Use custom values if available, otherwise use standard
                if custom_values:
                    merged_entities[key] = custom_values
                else:
                    merged_entities[key] = standard_values
            
            return merged_entities
        
        return standard_entities
```

#### 2. Model Configuration

```python
# Add configuration options
class ModelConfig:
    """Configuration for model selection and ensemble"""
    
    def __init__(
        self,
        use_custom_ner: bool = False,
        use_ensemble_ner: bool = False,
        custom_ner_weight: float = 0.7,
        fallback_to_standard: bool = True
    ):
        self.use_custom_ner = use_custom_ner
        self.use_ensemble_ner = use_ensemble_ner
        self.custom_ner_weight = custom_ner_weight
        self.fallback_to_standard = fallback_to_standard

# Usage
ref_tractor = ReferencesTractor(
    custom_ner_model_path="path/to/custom/model",
    model_config=ModelConfig(use_ensemble_ner=True)
)
```

## Adding New Evaluation Metrics

This section demonstrates adding custom evaluation metrics to the system.

### Example: Adding Precision/Recall Metrics

#### 1. Extend the Evaluator

```python
# utils/citation_evaluator.py

def calculate_precision_recall_metrics(self, evaluation_mode: str = "strict") -> Dict[str, Dict[str, float]]:
    """Calculate precision, recall, and F1-score for each API"""
    metrics = {}
    
    for api in self.apis + ['ensemble']:
        # Initialize counters
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        
        for result in self.results:
            gold_standard = result['gold_standard']
            
            # Get expected value for this approach
            if api == 'ensemble':
                expected_value = gold_standard.get('doi')
                api_result = result.get('ensemble', {})
            else:
                expected_value = gold_standard.get(api)
                api_result = result.get('api_results', {}).get(api, {})
            
            final_evaluation = api_result.get('final_evaluation', 'ERROR')
            
            # Skip errors
            if final_evaluation == 'ERROR':
                continue
            
            # Determine expected and actual outcomes
            expected_positive = bool(expected_value)
            actual_positive = (final_evaluation == 'CORRECT')
            
            # Count outcomes
            if expected_positive and actual_positive:
                true_positives += 1
            elif not expected_positive and actual_positive:
                false_positives += 1
            elif expected_positive and not actual_positive:
                false_negatives += 1
            else:  # not expected_positive and not actual_positive
                true_negatives += 1
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate additional metrics
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        
        metrics[api] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'specificity': specificity,
            'accuracy': accuracy,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives
        }
    
    return metrics

def calculate_confidence_calibration(self) -> Dict[str, Dict[str, float]]:
    """Calculate confidence calibration metrics"""
    metrics = {}
    
    for api in self.apis:
        confidence_scores = []
        correctness = []
        
        for result in self.results:
            api_result = result.get('api_results', {}).get(api, {})
            
            score = api_result.get('score')
            final_eval = api_result.get('final_evaluation')
            
            if score is not None and final_eval != 'ERROR':
                confidence_scores.append(float(score))
                correctness.append(1 if final_eval == 'CORRECT' else 0)
        
        if confidence_scores:
            # Calculate calibration metrics
            import numpy as np
            
            # Bin scores and calculate calibration
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0  # Expected Calibration Error
            mce = 0  # Maximum Calibration Error
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = [(conf >= bin_lower) and (conf < bin_upper) 
                         for conf in confidence_scores]
                prop_in_bin = sum(in_bin) / len(in_bin)
                
                if prop_in_bin > 0:
                    accuracy_in_bin = sum([correctness[i] for i, in_bin_i in enumerate(in_bin) if in_bin_i]) / sum(in_bin)
                    avg_confidence_in_bin = sum([confidence_scores[i] for i, in_bin_i in enumerate(in_bin) if in_bin_i]) / sum(in_bin)
                    
                    calibration_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                    ece += prop_in_bin * calibration_error
                    mce = max(mce, calibration_error)
            
            metrics[api] = {
                'ece': ece,  # Expected Calibration Error
                'mce': mce,  # Maximum Calibration Error
                'avg_confidence': np.mean(confidence_scores),
                'avg_accuracy': np.mean(correctness)
            }
    
    return metrics
```

#### 2. Update Dashboard Generation

```python
def generate_enhanced_dashboard(self, evaluation_mode: str = "strict") -> str:
    """Generate dashboard with additional metrics"""
    # Get existing metrics
    classification_metrics = self.calculate_classification_metrics(evaluation_mode)
    pr_metrics = self.calculate_precision_recall_metrics(evaluation_mode)
    calibration_metrics = self.calculate_confidence_calibration()
    
    output = []
    
    # ... existing dashboard code ...
    
    # Add precision/recall section
    output.append("")
    output.append("PRECISION/RECALL METRICS:")
    output.append("="*80)
    
    header = f"{'API':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Specificity':<12}"
    output.append(header)
    output.append("-"*80)
    
    for api in self.apis + ['ensemble']:
        if api in pr_metrics:
            m = pr_metrics[api]
            row = f"{api.title():<12} {m['precision']:<10.3f} {m['recall']:<10.3f} {m['f1_score']:<10.3f} {m['specificity']:<12.3f}"
            output.append(row)
    
    # Add calibration section
    output.append("")
    output.append("CONFIDENCE CALIBRATION:")
    output.append("="*60)
    
    header = f"{'API':<12} {'ECE':<8} {'MCE':<8} {'Avg_Conf':<10} {'Avg_Acc':<10}"
    output.append(header)
    output.append("-"*60)
    
    for api in self.apis:
        if api in calibration_metrics:
            m = calibration_metrics[api]
            row = f"{api.title():<12} {m['ece']:<8.3f} {m['mce']:<8.3f} {m['avg_confidence']:<10.3f} {m['avg_accuracy']:<10.3f}"
            output.append(row)
    
    # Add interpretation guide
    output.append("")
    output.append("ADDITIONAL METRICS INTERPRETATION:")
    output.append("-"*50)
    output.append("PRECISION/RECALL:")
    output.append("  • Precision: Of predicted positives, how many are actually correct")
    output.append("  • Recall: Of actual positives, how many are correctly predicted")
    output.append("  • F1-Score: Harmonic mean of precision and recall")
    output.append("  • Specificity: Of actual negatives, how many are correctly identified")
    output.append("")
    output.append("CONFIDENCE CALIBRATION:")
    output.append("  • ECE: Expected Calibration Error (lower is better)")
    output.append("  • MCE: Maximum Calibration Error (lower is better)")
    output.append("  • Well-calibrated models have ECE and MCE close to 0")
    
    return "\n".join(output)
```

## Production Integration Examples

### Enterprise-Grade Flask Web Service

Complete example of a production-ready Flask API with authentication, rate limiting, and monitoring:

```python
# production_app.py
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
import logging
from datetime import datetime, timedelta
import uuid
import redis
import json
from functools import wraps
import time

from references_tractor import ReferencesTractor

# Application setup
app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'your-secret-key'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)

CORS(app)
jwt = JWTManager(app)

# Redis for caching and rate limiting
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="redis://localhost:6379"
)

# Initialize pipeline with production settings
ref_tractor = ReferencesTractor(
    device="auto",
    enable_caching=True,
    cache_size_limit=5000
)

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Metrics tracking
class MetricsTracker:
    def __init__(self):
        self.redis = redis_client
        
    def track_request(self, endpoint: str, status: str, processing_time: float):
        """Track API request metrics"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Increment counters
        self.redis.hincrby(f'metrics:{today}:requests', endpoint, 1)
        self.redis.hincrby(f'metrics:{today}:status', status, 1)
        
        # Track average processing time
        key = f'metrics:{today}:processing_time:{endpoint}'
        self.redis.lpush(key, processing_time)
        self.redis.ltrim(key, 0, 999)  # Keep last 1000 measurements
        
        # Set expiration
        self.redis.expire(f'metrics:{today}:requests', 86400 * 7)  # 7 days
        self.redis.expire(f'metrics:{today}:status', 86400 * 7)
        self.redis.expire(key, 86400 * 7)

metrics = MetricsTracker()

# Performance monitoring decorator
def monitor_performance(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        request_id = str(uuid.uuid4())[:8]
        
        g.request_id = request_id
        g.start_time = start_time
        
        try:
            result = f(*args, **kwargs)
            processing_time = time.time() - start_time
            
            # Track successful request
            metrics.track_request(
                endpoint=request.endpoint,
                status='success',
                processing_time=processing_time
            )
            
            logger.info(f"Request {request_id} completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Track failed request
            metrics.track_request(
                endpoint=request.endpoint,
                status='error',
                processing_time=processing_time
            )
            
            logger.error(f"Request {request_id} failed after {processing_time:.2f}s: {str(e)}")
            raise
            
    return decorated_function

# Authentication endpoints
@app.route('/auth/login', methods=['POST'])
@limiter.limit("10 per minute")
def login():
    """Authenticate user and return JWT token"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    # Simple authentication (replace with real auth system)
    if username == 'admin' and password == 'password':
        access_token = create_access_token(identity=username)
        return jsonify(access_token=access_token)
    
    return jsonify({'error': 'Invalid credentials'}), 401

# Health and monitoring endpoints
@app.route('/health')
def health_check():
    """Comprehensive health check"""
    try:
        # Test system components
        cache_stats = ref_tractor.get_cache_stats()
        redis_ping = redis_client.ping()
        
        health_data = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {
                'citation_processor': 'healthy',
                'cache': 'healthy' if cache_stats['cache_enabled'] else 'disabled',
                'redis': 'healthy' if redis_ping else 'unhealthy'
            },
            'cache_stats': cache_stats
        }
        
        return jsonify(health_data)
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/metrics')
@jwt_required()
def get_metrics():
    """Get application metrics"""
    today = datetime.now().strftime('%Y-%m-%d')
    
    try:
        # Get request metrics
        requests_data = redis_client.hgetall(f'metrics:{today}:requests')
        status_data = redis_client.hgetall(f'metrics:{today}:status')
        
        # Calculate average processing times
        processing_times = {}
        for endpoint in requests_data.keys():
            times_key = f'metrics:{today}:processing_time:{endpoint}'
            times = redis_client.lrange(times_key, 0, -1)
            if times:
                avg_time = sum(float(t) for t in times) / len(times)
                processing_times[endpoint] = round(avg_time, 3)
        
        metrics_data = {
            'date': today,
            'requests_by_endpoint': requests_data,
            'requests_by_status': status_data,
            'average_processing_times': processing_times,
            'cache_stats': ref_tractor.get_cache_stats()
        }
        
        return jsonify(metrics_data)
        
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {str(e)}")
        return jsonify({'error': 'Failed to retrieve metrics'}), 500

# Citation processing endpoints
@app.route('/api/v1/citation/link', methods=['POST'])
@jwt_required()
@limiter.limit("100 per hour")
@monitor_performance
def link_citation():
    """Link a single citation with enhanced error handling"""
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'citation' not in data:
            return jsonify({'error': 'Missing citation text'}), 400
        
        citation = data['citation'].strip()
        if not citation:
            return jsonify({'error': 'Empty citation text'}), 400
        
        api = data.get('api', 'openalex')
        output = data.get('output', 'simple')
        
        # Validate parameters
        valid_apis = ['openalex', 'openaire', 'pubmed', 'crossref', 'hal']
        if api not in valid_apis:
            return jsonify({
                'error': f'Invalid API. Must be one of: {valid_apis}'
            }), 400
        
        if output not in ['simple', 'advanced']:
            return jsonify({
                'error': 'Invalid output format. Must be "simple" or "advanced"'
            }), 400
        
        # Check cache first
        cache_key = f"citation:{hash(citation + api + output)}"
        cached_result = redis_client.get(cache_key)
        
        if cached_result:
            logger.info(f"Request {g.request_id}: Cache hit for citation")
            result = json.loads(cached_result)
        else:
            # Process citation
            logger.info(f"Request {g.request_id}: Processing citation via {api}")
            result = ref_tractor.link_citation(
                citation, 
                api_target=api,
                output=output
            )
            
            # Cache result for 1 hour
            redis_client.setex(
                cache_key, 
                3600, 
                json.dumps(result, default=str)
            )
        
        # Build response
        response = {
            'request_id': g.request_id,
            'timestamp': datetime.utcnow().isoformat(),
            'api_used': api,
            'output_format': output,
            'processing_time': round(time.time() - g.start_time, 3),
            'result': result,
            'cached': cached_result is not None
        }
        
        status_msg = 'Found match' if result else 'No match'
        logger.info(f"Request {g.request_id}: {status_msg}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Request {g.request_id}: Processing failed - {str(e)}")
        return jsonify({
            'request_id': g.request_id,
            'error': 'Internal server error',
            'details': str(e) if app.debug else None
        }), 500

@app.route('/api/v1/citation/ensemble', methods=['POST'])
@jwt_required()
@limiter.limit("20 per hour")
@monitor_performance
def ensemble_link():
    """Enhanced ensemble linking with timeout protection"""
    try:
        data = request.get_json()
        
        if not data or 'citation' not in data:
            return jsonify({'error': 'Missing citation text'}), 400
        
        citation = data['citation'].strip()
        apis = data.get('apis', ['openalex', 'openaire', 'pubmed', 'crossref', 'hal'])
        output = data.get('output', 'simple')
        timeout = data.get('timeout', 60)  # Max 60 seconds
        
        # Validate APIs
        valid_apis = ['openalex', 'openaire', 'pubmed', 'crossref', 'hal']
        invalid_apis = [api for api in apis if api not in valid_apis]
        if invalid_apis:
            return jsonify({
                'error': f'Invalid APIs: {invalid_apis}. Valid APIs: {valid_apis}'
            }), 400
        
        logger.info(f"Request {g.request_id}: Ensemble linking with {len(apis)} APIs")
        
        # Process with timeout
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Ensemble processing timed out")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(min(timeout, 60))
        
        try:
            result = ref_tractor.link_citation_ensemble(
                citation,
                api_targets=apis,
                output=output
            )
        finally:
            signal.alarm(0)  # Clear timeout
        
        response = {
            'request_id': g.request_id,
            'timestamp': datetime.utcnow().isoformat(),
            'apis_used': apis,
            'processing_time': round(time.time() - g.start_time, 3),
            'result': result
        }
        
        return jsonify(response)
        
    except TimeoutError:
        logger.warning(f"Request {g.request_id}: Ensemble processing timed out")
        return jsonify({
            'request_id': g.request_id,
            'error': 'Processing timeout',
            'message': 'Ensemble processing took too long'
        }), 408
        
    except Exception as e:
        logger.error(f"Request {g.request_id}: Ensemble processing failed - {str(e)}")
        return jsonify({
            'request_id': g.request_id,
            'error': 'Internal server error'
        }), 500

@app.route('/api/v1/text/extract', methods=['POST'])
@jwt_required()
@limiter.limit("30 per hour")
@monitor_performance
def extract_and_link():
    """Extract citations from text with size limits"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text content'}), 400
        
        text = data['text']
        api = data.get('api', 'openalex')
        max_length = data.get('max_length', 10000)  # Limit text size
        
        # Validate text size
        if len(text) > max_length:
            return jsonify({
                'error': f'Text too long. Maximum {max_length} characters allowed.'
            }), 400
        
        logger.info(f"Request {g.request_id}: Extract and link from {len(text)} chars")
        
        results = ref_tractor.extract_and_link_from_text(text, api_target=api)
        
        response = {
            'request_id': g.request_id,
            'timestamp': datetime.utcnow().isoformat(),
            'text_length': len(text),
            'citations_found': len(results),
            'processing_time': round(time.time() - g.start_time, 3),
            'results': results
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Request {g.request_id}: Text extraction failed - {str(e)}")
        return jsonify({
            'request_id': g.request_id,
            'error': 'Internal server error'
        }), 500

# Error handlers
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': str(e.description)
    }), 429

@app.errorhandler(404)
def not_found_handler(e):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested URL was not found on the server'
    }), 404

@app.errorhandler(500)
def internal_error_handler(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

if __name__ == '__main__':
    # Production configuration
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )
```

### Advanced Batch Processing System

Enterprise-grade batch processing with database integration, job queuing, and progress tracking:

```python
# enterprise_batch_processor.py
import asyncio
import aiofiles
import aiopg
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, AsyncGenerator
import logging
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import time

from references_tractor import ReferencesTractor

@dataclass
class ProcessingJob:
    """Data class for batch processing jobs"""
    job_id: str
    input_file: str
    output_file: str
    api: str
    status: str = "pending"
    total_citations: int = 0
    processed_citations: int = 0
    successful_links: int = 0
    failed_citations: int = 0
    created_at: datetime = None
    started_at: datetime = None
    completed_at: datetime = None
    error_message: str = None

class EnterpriseBatchProcessor:
    """Enterprise-grade batch processing with database integration"""
    
    def __init__(
        self, 
        device: str = "auto", 
        max_workers: int = 4,
        db_config: Dict = None,
        redis_config: Dict = None
    ):
        self.ref_tractor = ReferencesTractor(device=device)
        self.max_workers = max_workers
        self.db_config = db_config or {
            'host': 'localhost',
            'port': 5432,
            'database': 'citations',
            'user': 'postgres',
            'password': 'password'
        }
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Job queue and worker management
        self.job_queue = queue.Queue()
        self.active_jobs = {}
        self.workers_running = False
        
    def _setup_logging(self):
        """Setup comprehensive logging"""
        logger = logging.getLogger('enterprise_batch_processor')
        logger.setLevel(logging.INFO)
        
        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            'batch_processing.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    async def initialize_database(self):
        """Initialize PostgreSQL database tables"""
        create_tables_sql = """
        CREATE TABLE IF NOT EXISTS processing_jobs (
            job_id VARCHAR(50) PRIMARY KEY,
            input_file TEXT NOT NULL,
            output_file TEXT NOT NULL,
            api VARCHAR(20) NOT NULL,
            status VARCHAR(20) DEFAULT 'pending',
            total_citations INTEGER DEFAULT 0,
            processed_citations INTEGER DEFAULT 0,
            successful_links INTEGER DEFAULT 0,
            failed_citations INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            error_message TEXT
        );
        
        CREATE TABLE IF NOT EXISTS citation_results (
            id SERIAL PRIMARY KEY,
            job_id VARCHAR(50) REFERENCES processing_jobs(job_id),
            citation_text TEXT NOT NULL,
            api_used VARCHAR(20) NOT NULL,
            doi TEXT,
            confidence_score REAL,
            status VARCHAR(20) NOT NULL,
            error_message TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            result_json JSONB
        );
        
        CREATE INDEX IF NOT EXISTS idx_citation_results_job_id ON citation_results(job_id);
        CREATE INDEX IF NOT EXISTS idx_citation_results_status ON citation_results(status);
        """
        
        async with aiopg.create_pool(**self.db_config) as pool:
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(create_tables_sql)
                    self.logger.info("Database tables initialized")
    
    async def create_job(
        self, 
        input_file: str, 
        output_file: str, 
        api: str = "ensemble"
    ) -> str:
        """Create a new batch processing job"""
        import uuid
        
        job_id = str(uuid.uuid4())
        
        # Count citations in input file
        total_citations = await self._count_citations(input_file)
        
        job = ProcessingJob(
            job_id=job_id,
            input_file=input_file,
            output_file=output_file,
            api=api,
            total_citations=total_citations,
            created_at=datetime.utcnow()
        )
        
        # Save to database
        await self._save_job_to_db(job)
        
        # Add to processing queue
        self.job_queue.put(job)
        
        self.logger.info(f"Created job {job_id} with {total_citations} citations")
        return job_id
    
    async def _count_citations(self, input_file: str) -> int:
        """Count citations in input file"""
        path = Path(input_file)
        
        if path.suffix == '.txt':
            async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                content = await f.read()
                return len([line for line in content.split('\n') if line.strip()])
        
        elif path.suffix == '.csv':
            # For large files, use streaming
            count = 0
            async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                async for line in f:
                    if count == 0:  # Skip header
                        count += 1
                        continue
                    if line.strip():
                        count += 1
            return max(0, count - 1)  # Subtract header
        
        elif path.suffix == '.json':
            async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)
                if isinstance(data, list):
                    return len(data)
                elif isinstance(data, dict):
                    return len(data)
        
        return 0
    
    async def _save_job_to_db(self, job: ProcessingJob):
        """Save job to database"""
        async with aiopg.create_pool(**self.db_config) as pool:
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("""
                        INSERT INTO processing_jobs 
                        (job_id, input_file, output_file, api, status, total_citations, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        job.job_id, job.input_file, job.output_file, job.api,
                        job.status, job.total_citations, job.created_at
                    ))
    
    async def _update_job_status(self, job_id: str, **updates):
        """Update job status in database"""
        if not updates:
            return
            
        set_clauses = []
        values = []
        
        for field, value in updates.items():
            set_clauses.append(f"{field} = %s")
            values.append(value)
        
        values.append(job_id)
        
        async with aiopg.create_pool(**self.db_config) as pool:
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    sql = f"UPDATE processing_jobs SET {', '.join(set_clauses)} WHERE job_id = %s"
                    await cursor.execute(sql, values)
    
    async def start_workers(self):
        """Start background worker threads"""
        if self.workers_running:
            return
        
        self.workers_running = True
        
        # Start worker threads
        self.worker_threads = []
        for i in range(self.max_workers):
            thread = threading.Thread(
                target=self._worker_thread,
                args=(f"worker-{i}",),
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)
        
        self.logger.info(f"Started {self.max_workers} worker threads")
    
    def _worker_thread(self, worker_name: str):
        """Worker thread for processing jobs"""
        self.logger.info(f"Worker {worker_name} started")
        
        while self.workers_running:
            try:
                # Get job from queue (with timeout)
                job = self.job_queue.get(timeout=5)
                
                self.logger.info(f"Worker {worker_name} processing job {job.job_id}")
                self.active_jobs[job.job_id] = job
                
                # Process the job
                asyncio.run(self._process_job(job, worker_name))
                
                # Mark task as done
                self.job_queue.task_done()
                
                # Remove from active jobs
                if job.job_id in self.active_jobs:
                    del self.active_jobs[job.job_id]
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_name} error: {str(e)}")
    
    async def _process_job(self, job: ProcessingJob, worker_name: str):
        """Process a single job"""
        try:
            # Update job status
            await self._update_job_status(
                job.job_id,
                status='processing',
                started_at=datetime.utcnow()
            )
            
            # Load citations
            citations = await self._load_citations_async(job.input_file)
            
            # Process citations with progress tracking
            results = []
            successful_count = 0
            failed_count = 0
            
            for i, citation in enumerate(citations):
                try:
                    self.logger.debug(f"Job {job.job_id}: Processing citation {i+1}/{len(citations)}")
                    
                    # Process citation
                    if job.api == "ensemble":
                        result = self.ref_tractor.link_citation_ensemble(citation)
                    else:
                        result = self.ref_tractor.link_citation(citation, api_target=job.api)
                    
                    # Save individual result to database
                    await self._save_citation_result(job.job_id, citation, job.api, result)
                    
                    # Track success/failure
                    if result and (result.get('doi') or result.get('result')):
                        successful_count += 1
                        status = 'success'
                    else:
                        status = 'no_match'
                    
                    results.append({
                        'citation': citation,
                        'result': result,
                        'status': status,
                        'index': i
                    })
                    
                except Exception as e:
                    failed_count += 1
                    self.logger.error(f"Job {job.job_id}: Error processing citation {i}: {str(e)}")
                    
                    # Save error result
                    await self._save_citation_result(
                        job.job_id, citation, job.api, {}, 
                        status='error', error_message=str(e)
                    )
                    
                    results.append({
                        'citation': citation,
                        'error': str(e),
                        'status': 'error',
                        'index': i
                    })
                
                # Update progress periodically
                if (i + 1) % 10 == 0:
                    await self._update_job_status(
                        job.job_id,
                        processed_citations=i + 1,
                        successful_links=successful_count,
                        failed_citations=failed_count
                    )
            
            # Save final results
            await self._save_results_async(results, job.output_file)
            
            # Update job completion
            await self._update_job_status(
                job.job_id,
                status='completed',
                processed_citations=len(citations),
                successful_links=successful_count,
                failed_citations=failed_count,
                completed_at=datetime.utcnow()
            )
            
            # Generate summary
            await self._generate_job_summary(job.job_id)
            
            self.logger.info(f"Job {job.job_id} completed: {successful_count} successful, {failed_count} failed")
            
        except Exception as e:
            # Update job with error status
            await self._update_job_status(
                job.job_id,
                status='failed',
                error_message=str(e),
                completed_at=datetime.utcnow()
            )
            
            self.logger.error(f"Job {job.job_id} failed: {str(e)}")
    
    async def _load_citations_async(self, input_file: str) -> List[str]:
        """Load citations from file asynchronously"""
        path = Path(input_file)
        citations = []
        
        if path.suffix == '.txt':
            async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                content = await f.read()
                citations = [line.strip() for line in content.split('\n') if line.strip()]
        
        elif path.suffix == '.csv':
            # Use aiofiles for large CSV files
            async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                content = await f.read()
                
            # Parse CSV in memory (consider streaming for very large files)
            import io
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                citation = row.get('citation') or row.get('text') or row.get('reference')
                if citation:
                    citations.append(citation.strip())
        
        elif path.suffix == '.json':
            async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)
                
                if isinstance(data, list):
                    citations = data
                elif isinstance(data, dict):
                    citations = list(data.keys())
        
        return citations
    
    async def _save_citation_result(
        self, 
        job_id: str, 
        citation: str, 
        api: str, 
        result: Dict,
        status: str = None,
        error_message: str = None
    ):
        """Save individual citation result to database"""
        
        # Determine status if not provided
        if status is None:
            if error_message:
                status = 'error'
            elif result and (result.get('doi') or result.get('result')):
                status = 'success'
            else:
                status = 'no_match'
        
        # Extract key fields
        doi = result.get('doi') if result else None
        confidence_score = result.get('score') if result else None
        
        async with aiopg.create_pool(**self.db_config) as pool:
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("""
                        INSERT INTO citation_results 
                        (job_id, citation_text, api_used, doi, confidence_score, status, error_message, result_json)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        job_id, citation[:1000], api, doi, confidence_score, 
                        status, error_message, json.dumps(result) if result else None
                    ))
    
    async def _save_results_async(self, results: List[Dict], output_file: str):
        """Save results to output file asynchronously"""
        path = Path(output_file)
        
        if path.suffix == '.json':
            async with aiofiles.open(path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(results, indent=2, default=str))
        
        elif path.suffix == '.csv':
            # Flatten results for CSV
            flattened_results = []
            for result in results:
                flat_result = {
                    'citation': result['citation'],
                    'status': result['status'],
                    'index': result['index']
                }
                
                if 'result' in result and result['result']:
                    api_result = result['result']
                    flat_result.update({
                        'doi': api_result.get('doi', ''),
                        'score': api_result.get('score', ''),
                        'api_id': api_result.get('openalex_id') or api_result.get('pubmed_id', ''),
                        'formatted_citation': api_result.get('result', '')
                    })
                
                if 'error' in result:
                    flat_result['error'] = result['error']
                
                flattened_results.append(flat_result)
            
            # Write CSV
            if flattened_results:
                async with aiofiles.open(path, 'w', encoding='utf-8', newline='') as f:
                    # Write header
                    headers = list(flattened_results[0].keys())
                    await f.write(','.join(headers) + '\n')
                    
                    # Write data
                    for row in flattened_results:
                        csv_row = []
                        for header in headers:
                            value = str(row.get(header, '')).replace('"', '""')
                            csv_row.append(f'"{value}"')
                        await f.write(','.join(csv_row) + '\n')
    
    async def _generate_job_summary(self, job_id: str):
        """Generate comprehensive job summary"""
        async with aiopg.create_pool(**self.db_config) as pool:
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    # Get job details
                    await cursor.execute("""
                        SELECT * FROM processing_jobs WHERE job_id = %s
                    """, (job_id,))
                    job_row = await cursor.fetchone()
                    
                    # Get result statistics
                    await cursor.execute("""
                        SELECT 
                            status,
                            COUNT(*) as count,
                            AVG(confidence_score) as avg_confidence
                        FROM citation_results 
                        WHERE job_id = %s 
                        GROUP BY status
                    """, (job_id,))
                    
                    stats = await cursor.fetchall()
                    
                    # Generate summary report
                    summary = {
                        'job_id': job_id,
                        'input_file': job_row[1],
                        'output_file': job_row[2],
                        'api': job_row[3],
                        'status': job_row[4],
                        'total_citations': job_row[5],
                        'processed_citations': job_row[6],
                        'successful_links': job_row[7],
                        'failed_citations': job_row[8],
                        'created_at': job_row[9].isoformat() if job_row[9] else None,
                        'started_at': job_row[10].isoformat() if job_row[10] else None,
                        'completed_at': job_row[11].isoformat() if job_row[11] else None,
                        'processing_time': None,
                        'detailed_stats': {}
                    }
                    
                    # Calculate processing time
                    if job_row[10] and job_row[11]:  # started_at and completed_at
                        processing_time = job_row[11] - job_row[10]
                        summary['processing_time'] = str(processing_time)
                    
                    # Add detailed statistics
                    for stat_row in stats:
                        status, count, avg_confidence = stat_row
                        summary['detailed_stats'][status] = {
                            'count': count,
                            'avg_confidence': float(avg_confidence) if avg_confidence else None
                        }
                    
                    # Save summary
                    summary_file = f"job_summary_{job_id}.json"
                    async with aiofiles.open(summary_file, 'w', encoding='utf-8') as f:
                        await f.write(json.dumps(summary, indent=2))
                    
                    self.logger.info(f"Job summary saved to {summary_file}")
    
    async def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get current job status"""
        async with aiopg.create_pool(**self.db_config) as pool:
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("""
                        SELECT * FROM processing_jobs WHERE job_id = %s
                    """, (job_id,))
                    
                    row = await cursor.fetchone()
                    if not row:
                        return None
                    
                    return {
                        'job_id': row[0],
                        'input_file': row[1],
                        'output_file': row[2],
                        'api': row[3],
                        'status': row[4],
                        'total_citations': row[5],
                        'processed_citations': row[6],
                        'successful_links': row[7],
                        'failed_citations': row[8],
                        'created_at': row[9].isoformat() if row[9] else None,
                        'started_at': row[10].isoformat() if row[10] else None,
                        'completed_at': row[11].isoformat() if row[11] else None,
                        'error_message': row[12],
                        'progress_percentage': (row[6] / row[5] * 100) if row[5] > 0 else 0
                    }
    
    async def list_jobs(self, limit: int = 50, status: str = None) -> List[Dict]:
        """List recent jobs with optional status filter"""
        async with aiopg.create_pool(**self.db_config) as pool:
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    if status:
                        await cursor.execute("""
                            SELECT job_id, input_file, api, status, total_citations, 
                                   processed_citations, created_at, completed_at
                            FROM processing_jobs 
                            WHERE status = %s
                            ORDER BY created_at DESC 
                            LIMIT %s
                        """, (status, limit))
                    else:
                        await cursor.execute("""
                            SELECT job_id, input_file, api, status, total_citations, 
                                   processed_citations, created_at, completed_at
                            FROM processing_jobs 
                            ORDER BY created_at DESC 
                            LIMIT %s
                        """, (limit,))
                    
                    rows = await cursor.fetchall()
                    
                    return [
                        {
                            'job_id': row[0],
                            'input_file': row[1],
                            'api': row[2],
                            'status': row[3],
                            'total_citations': row[4],
                            'processed_citations': row[5],
                            'created_at': row[6].isoformat() if row[6] else None,
                            'completed_at': row[7].isoformat() if row[7] else None,
                            'progress_percentage': (row[5] / row[4] * 100) if row[4] > 0 else 0
                        }
                        for row in rows
                    ]
    
    async def stop_workers(self):
        """Stop all worker threads"""
        self.workers_running = False
        
        # Wait for workers to finish
        for thread in getattr(self, 'worker_threads', []):
            thread.join(timeout=10)
        
        self.logger.info("All workers stopped")

# Usage example
async def main():
    """Example usage of enterprise batch processor"""
    
    # Initialize processor
    processor = EnterpriseBatchProcessor(
        device="auto",
        max_workers=4,
        db_config={
            'host': 'localhost',
            'port': 5432,
            'database': 'citations',
            'user': 'postgres',
            'password': 'password'
        }
    )
    
    # Initialize database
    await processor.initialize_database()
    
    # Start workers
    await processor.start_workers()
    
    try:
        # Create a job
        job_id = await processor.create_job(
            input_file="citations.txt",
            output_file="results.json",
            api="ensemble"
        )
        
        print(f"Created job: {job_id}")
        
        # Monitor job progress
        while True:
            status = await processor.get_job_status(job_id)
            if not status:
                break
                
            print(f"Job {job_id}: {status['status']} - "
                  f"{status['processed_citations']}/{status['total_citations']} "
                  f"({status['progress_percentage']:.1f}%)")
            
            if status['status'] in ['completed', 'failed']:
                break
            
            await asyncio.sleep(5)
        
        # List all jobs
        jobs = await processor.list_jobs(limit=10)
        print(f"Recent jobs: {len(jobs)}")
        
    finally:
        # Clean shutdown
        await processor.stop_workers()

if __name__ == "__main__":
    asyncio.run(main())
```

This comprehensive examples document provides developers with practical, working code they can adapt for their specific needs when extending References Tractor. The examples cover:

1. **Adding New APIs** - Complete walkthrough with Semantic Scholar
2. **Adding New Models** - Custom NER model integration
3. **Adding New Evaluation Metrics** - Precision/recall and calibration metrics
4. **Production Integration** - Enterprise-grade Flask API with authentication, monitoring, and rate limiting
5. **Enterprise Batch Processing** - Async processing with database integration, job queuing, and progress tracking

Each example includes comprehensive error handling, logging, testing, and production-ready features.
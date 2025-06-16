#!/usr/bin/env python3
"""
Quick debug script for OpenAIRE results with the enhanced pipeline
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from references_tractor import ReferencesTractor

def debug_openaire_result():
    """Debug the specific OpenAIRE issue with the test citation"""
    
    # Test citation that worked yesterday
    test_citation = 'GARSHELIS I J: \'A TORQUE TRANSDUCER UTILIZING A CIRCULARLY POLARIZED RING\', PROCEEDINGS OF THE INTERNATIONAL MAGNETICS CONFERENCE (INTERMAG), ST. LOUIS, APR. 13 - 16, 1992, no. -, 13 April 1992 (1992-04-13), INSTITUTE OF ELECTRICAL AND ELECTRONICS ENGINEERS, pages AD08, XP000341743'
    
    print("🔍 DEBUGGING OPENAIRE RESULTS")
    print("=" * 60)
    print(f"Citation: {test_citation[:100]}...")
    
    # Initialize pipeline
    ref_tractor = ReferencesTractor()
    
    print("\n1. Testing OpenAIRE with basic output:")
    try:
        result_simple = ref_tractor.link_citation(test_citation, api_target="openaire", output='simple')
        print(f"   Simple result keys: {list(result_simple.keys())}")
        print(f"   DOI: {result_simple.get('doi')}")
        print(f"   Result found: {'result' in result_simple}")
        if 'result' in result_simple:
            print(f"   Citation: {result_simple['result'][:100]}...")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n2. Testing OpenAIRE with advanced output:")
    try:
        result_advanced = ref_tractor.link_citation(test_citation, api_target="openaire", output='advanced')
        print(f"   Advanced result keys: {list(result_advanced.keys())}")
        print(f"   Main DOI: {result_advanced.get('main_doi')}")
        print(f"   Alternative DOIs: {result_advanced.get('alternative_dois')}")
        print(f"   Total DOIs: {result_advanced.get('total_dois')}")
        print(f"   Legacy DOI: {result_advanced.get('doi')}")
        
        if 'full-publication' in result_advanced:
            pub = result_advanced['full-publication']
            print(f"   Full publication keys: {list(pub.keys()) if isinstance(pub, dict) else 'Not a dict'}")
            
            # Check for DOI-related fields in raw publication
            if isinstance(pub, dict):
                print(f"   Raw pids: {pub.get('pids', 'Not found')}")
                print(f"   Raw instances: {len(pub.get('instances', []))} instances")
                
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n3. Testing field mapper DOI extraction directly:")
    try:
        from references_tractor.search.field_mapper import FieldMapper
        field_mapper = FieldMapper()
        
        # First get the raw publication data
        result_advanced = ref_tractor.link_citation(test_citation, api_target="openaire", output='advanced')
        if 'full-publication' in result_advanced:
            pub = result_advanced['full-publication']
            print(f"   Testing DOI extraction on publication...")
            doi_result = field_mapper.extract_dois_from_result(pub, 'openaire')
            print(f"   DOI extraction result: {doi_result}")
            print(f"   All DOIs: {field_mapper.get_all_dois_from_result(doi_result)}")
            
    except Exception as e:
        print(f"   ❌ Field mapper error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n4. Testing raw API call:")
    try:
        # Test the raw search API
        ner_entities = ref_tractor.process_ner_entities(test_citation)
        print(f"   NER entities: {ner_entities}")
        
        raw_results = ref_tractor.search_api(ner_entities, api="openaire", target_count=3)
        print(f"   Raw search results: {len(raw_results)} results")
        
        if raw_results:
            first_result = raw_results[0]
            print(f"   First result keys: {list(first_result.keys()) if isinstance(first_result, dict) else 'Not a dict'}")
            print(f"   Enhanced fields in first result:")
            print(f"     main_doi: {first_result.get('main_doi')}")
            print(f"     alternative_dois: {first_result.get('alternative_dois')}")
            print(f"     total_dois: {first_result.get('total_dois')}")
            print(f"     api_source: {first_result.get('api_source')}")
            
    except Exception as e:
        print(f"   ❌ Raw API error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_openaire_result()
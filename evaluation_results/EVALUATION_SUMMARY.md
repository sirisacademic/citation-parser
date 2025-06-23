# Citation Linking Evaluation Summary

This document provides an overview of the evaluation results for the References Tractor citation linking system.

## Overview

- **Total Citations Evaluated**: 200
- **APIs Tested**: OpenAlex, OpenAIRE, PubMed, CrossRef, HAL
- **Evaluation Mode**: Strict
- **Evaluation Date**: June 23, 2025

## Performance Summary

| API | Accuracy | Total | Correct | Matches | No Results | Incorrect | Wrong Links | Missing | False Positives |
|-----|----------|-------|---------|---------|------------|-----------|-------------|---------|-----------------|
| **OpenAlex** | 67.5% | 200 | 135 | 103 | 32 | 65 | 7 | 56 | 2 |
| **OpenAIRE** | 65.0% | 200 | 130 | 94 | 36 | 70 | 8 | 59 | 3 |
| **PubMed** | 76.0% | 200 | 152 | 29 | 123 | 48 | 2 | 34 | 12 |
| **CrossRef** | 63.5% | 200 | 127 | 0 | 127 | 73 | 0 | 0 | 12 |
| **HAL** | 98.5% | 200 | 197 | 0 | 197 | 3 | 0 | 0 | 3 |
| **Ensemble** | 62.5% | 200 | 125 | 86 | 39 | 75 | 6 | 68 | 1 |

## Key Findings

### Best Performing APIs
1. **HAL** (98.5%) - Excellent accuracy, primarily correctly identifying when citations are not in their database
2. **PubMed** (76.0%) - Strong performance for biomedical literature
3. **OpenAlex** (67.5%) - Good general-purpose performance across disciplines

### API Characteristics

- **HAL**: Specialized repository with very high precision but limited coverage
- **PubMed**: Excellent for biomedical citations, conservative linking approach
- **OpenAlex**: Balanced performance across all citation types
- **OpenAIRE**: Similar performance to OpenAlex with slightly lower accuracy
- **CrossRef**: Conservative approach, no positive matches in test set
- **Ensemble**: Combines multiple APIs but shows room for improvement in current implementation

### Error Analysis

- **Missing Papers (I_Miss)**: OpenAIRE (59) and OpenAlex (56) have the highest rates of missing expected citations, though these are also the APIs with the most gold standard IDs in the test set
- **False Positives (I_Spur)**: PubMed (12) and CrossRef (12) show the highest false positive rates
- **Wrong Links (I_Match)**: Generally low across all APIs, indicating good precision when matches are found

## Methodology

The evaluation uses a **strict classification approach** with the following metrics:

- **Correct Matches**: Citations correctly linked to expected database records
- **Correct No Results**: Citations correctly identified as not available in the database
- **Wrong Links**: Citations linked to incorrect database records
- **Missing**: Expected citations that were not found
- **False Positives**: Unexpected citations returned when none were expected

## Usage Recommendations

- **For biomedical literature**: Use PubMed for highest accuracy
- **For French academic content**: HAL provides exceptional accuracy
- **For general academic content**: OpenAlex offers the best balance of coverage and accuracy
- **For comprehensive coverage**: Consider ensemble approach with improved voting mechanisms

## Evaluation Data

Complete evaluation results including individual citation analysis are available in the `evaluation_results/` directory with detailed breakdowns by API and citation type.

---

*This evaluation was conducted using the References Tractor v2.0.0 citation linking system. For detailed methodology and individual results, see the complete evaluation reports in this directory.*
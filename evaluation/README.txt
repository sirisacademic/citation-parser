How to run the evaluation from the project root folder:

# Test setup
python evaluation/evaluate_citations.py --test-setup

# Quick test
python evaluation/evaluate_citations.py --limit 3

# Full evaluation
python evaluation/evaluate_citations.py

The results will be saved in:

evaluation_results/  ← Created in project root
├── 01_summary_dashboard_[timestamp].txt
├── 02_comparison_table_[timestamp].txt
├── 03_*_detailed_[timestamp].txt
└── raw_results_[timestamp].json



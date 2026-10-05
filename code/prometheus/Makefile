.PHONY: test quick v1 v1-analyze v1-figures v2-seed v2-analyze v2-figures audit

test:
	PYTHONPATH=src pytest -q

quick:
	PYTHONPATH=src python scripts/run_long_horizon.py --seeds 1 --quick

v1:
	PYTHONPATH=src python scripts/run_experiments.py --seeds 0 1 2 3 4

v1-analyze:
	python scripts/analyze_results.py

v1-figures:
	python scripts/make_figures.py

v2-seed:
	PYTHONPATH=src python scripts/run_long_seed.py --seed 5

v2-analyze:
	python scripts/analyze_long_horizon.py

v2-figures:
	python scripts/make_long_horizon_figures.py

audit:
	PYTHONPATH=src pytest -q
	python scripts/analyze_long_horizon.py

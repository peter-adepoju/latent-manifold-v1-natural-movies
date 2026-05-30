PYTHON ?= python
CONFIG ?= configs/default.yaml

.PHONY: install data preprocess manifold latent-eval eval report app tests publication-audit publication-plan publication-cebra-plan publication-dnn-plan clean

install:
	pip install -e .

data:
	$(PYTHON) scripts/download_allen_data.py --config $(CONFIG)

preprocess:
	$(PYTHON) scripts/preprocess_sessions.py --config $(CONFIG)

manifold:
	$(PYTHON) scripts/run_manifold_analysis.py --config $(CONFIG)

latent-eval:
	$(PYTHON) scripts/evaluate_latent_feature_predictivity.py --config $(CONFIG)

eval: latent-eval
	$(PYTHON) scripts/evaluate_models.py --config $(CONFIG)

report:
	$(PYTHON) scripts/build_report.py --config $(CONFIG)

app:
	streamlit run app/streamlit_app.py

tests:
	pytest

publication-audit:
	$(PYTHON) scripts/publication_audit_readiness.py

publication-plan:
	$(PYTHON) scripts/publication_build_run_plan.py

publication-cebra-plan:
	$(PYTHON) scripts/publication_plan_cebra_variants.py

publication-dnn-plan:
	$(PYTHON) scripts/publication_plan_dnn_features.py

clean:
	find reports/figures -type f ! -name .gitkeep -delete
	find reports/tables -type f ! -name .gitkeep -delete
	find data/interim -type f ! -name .gitkeep -delete
	find data/processed -type f ! -name .gitkeep -delete
	find models -maxdepth 1 -type f -delete

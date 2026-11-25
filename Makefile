.PHONY: test test-fast eval demo baseline data api ocr newman promptfoo promptfoo-degraded all-checks install graph

install:         ## python + node dependencies (also needs the tesseract program for OCR tests)
	pip install -r requirements.txt
	npm install --prefix tools

test:            ## every pytest test (includes ~1 minute of real OCR)
	python -m pytest

test-fast:       ## everything except the slow OCR tests
	python -m pytest -m "not slow"

eval:            ## evaluate the healthy model and gate on drift
	python run_evals.py --gate

baseline:        ## freeze the current healthy run as the new baseline (deliberate action)
	python run_evals.py --profile v1 --update-baseline

demo:            ## the 3-run story: healthy passes, degraded model blocked, flaky model blocked
	python run_evals.py --profile v1 --gate
	-python run_evals.py --profile v2_degraded --gate
	-python run_evals.py --profile flaky --gate

langgraph:       ## same evaluation, run on the LangGraph version of the agent
	python run_evals.py --agent langgraph --gate

ocr:             ## scan-quality experiment with real Tesseract OCR
	python run_ocr_eval.py

newman:          ## Postman collection against the live API (Newman)
	python -m uvicorn app.api:app --port 8000 --log-level warning & echo $$! > /tmp/api.pid; sleep 3; \
	tools/node_modules/.bin/newman run postman/intake.postman_collection.json --env-var runId=run$$(date +%s); \
	status=$$?; kill $$(cat /tmp/api.pid); exit $$status

promptfoo:       ## promptfoo evals on the healthy agent (expect all pass)
	python scripts/make_promptfoo.py
	cd promptfoo && PROMPTFOO_DISABLE_TELEMETRY=1 PROMPTFOO_PYTHON=python3 ../tools/node_modules/.bin/promptfoo eval -c promptfooconfig.yaml --no-cache --no-share

promptfoo-degraded: ## promptfoo on the degraded model (expect failures: proves the evals can fail)
	cd promptfoo && PROMPTFOO_PROFILE=v2_degraded PROMPTFOO_DISABLE_TELEMETRY=1 PROMPTFOO_PYTHON=python3 ../tools/node_modules/.bin/promptfoo eval -c promptfooconfig.yaml --no-cache --no-share

data:            ## regenerate datasets
	python scripts/make_data.py

api:             ## start the REST API on :8000
	python -m uvicorn app.api:app --port 8000

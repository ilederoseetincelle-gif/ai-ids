# AI-IDS — Common commands
# Run `make help` to see all targets

.PHONY: help install test train train-rf train-fast train-synthetic train-tune demo dashboard clean clean-cache clean-all

help:
	@echo "AI-IDS — Available commands:"
	@echo ""
	@echo "  make install          Install all Python dependencies (including shap)"
	@echo "  make test             Run the full pytest test suite"
	@echo "  make train            Train XGBoost (default) + anomaly detector"
	@echo "  make train-rf         Train Random Forest + anomaly detector"
	@echo "  make train-fast       Train XGBoost on 10% of data (fast iteration)"
	@echo "  make train-synthetic  Train on synthetic data (no real dataset needed)"
	@echo "  make train-tune       GridSearchCV RF + full model comparison (1-3 hrs)"
	@echo "  make demo             Run the demonstration replay + dashboard"
	@echo "  make dashboard        Launch only the Streamlit dashboard"
	@echo "  make clean            Remove generated artifacts (models, logs, plots)"
	@echo "  make clean-cache      Remove __pycache__ directories"
	@echo ""

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

train:
	python train_pipeline.py

train-rf:
	python train_pipeline.py --model rf

train-fast:
	python train_pipeline.py --sample 0.1

train-synthetic:
	python train_pipeline.py --synthetic --synthetic-rows 30000

train-tune:
	python train_pipeline.py --tune --compare

demo:
	python main.py --mode demo --clear-log

dashboard:
	python main.py --mode dashboard

live:
	@echo "Live mode requires root. Use: sudo python main.py --mode live --interface <iface>"

clean:
	rm -rf models/*.pkl models/plots/*.png models/feature_importance.csv
	rm -rf logs/*.jsonl logs/*.log
	@echo "Cleaned: models/, logs/"

clean-cache:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned: __pycache__, .pytest_cache"

clean-all: clean clean-cache

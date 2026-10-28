.PHONY: install pipeline notebook app clean

install:
	pip install -r requirements.txt

pipeline:
	python -m src.pipeline

notebook:
	jupyter nbconvert --execute --inplace notebooks/analysis.ipynb

app:
	streamlit run app.py

clean:
	rm -f data/ai4i2020.db reports/metrics.json
	rm -rf reports/figures/*.png reports/sql_results/*.csv src/__pycache__

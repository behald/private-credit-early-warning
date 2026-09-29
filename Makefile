.PHONY: install data run demo test clean

install:
	pip install -r requirements.txt

data:
	python filter_bdc.py

run:
	python run_pipeline.py

demo:
	streamlit run src/app.py

test:
	pytest tests/ -v

clean:
	rm -f data/processed/warehouse.duckdb models/*.json models/*.csv models/*.txt

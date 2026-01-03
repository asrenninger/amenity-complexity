.PHONY: install test clean

install:
	pip install -e .

test:
	pytest tests/

clean:
	rm -rf build dist *.egg-info
	find . -name "__pycache__" -delete

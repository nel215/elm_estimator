test:
	pep8 . && PYTHONPATH=. py.test tests/

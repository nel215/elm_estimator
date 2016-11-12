clean:
	rm -rf build/ dist/
build: clean
	python setup.py build bdist_wheel
install:
	pip install -U .
test: install
	PYTHONPATH=. py.test --pep8 .

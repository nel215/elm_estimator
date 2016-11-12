clean:
	rm -rf build/ dist/
build: clean
	python setup.py build bdist_wheel
test:
	pep8 . && PYTHONPATH=. py.test tests/

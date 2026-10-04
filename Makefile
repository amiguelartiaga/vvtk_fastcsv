.PHONY: install install-all test core-test bench bench-smoke wheel clean

PY ?= python

install:
	$(PY) -m pip install -e '.[dev]'

install-all:
	$(PY) -m pip install -e '.[all]'

test:
	$(PY) -m pytest -q

core-test:
	cmake -S . -B build/core -DFASTCSV_BUILD_PYTHON=OFF -DFASTCSV_BUILD_CORE_TESTS=ON -DFASTCSV_ENABLE_IPO=OFF
	cmake --build build/core -j
	cd build/core && ctest --output-on-failure

bench-smoke:
	$(PY) benchmarks/benchmark.py --scale 0.05 --repeats 2

bench:
	$(PY) benchmarks/benchmark.py --repeats 3

wheel:
	$(PY) -m pip wheel . --no-deps -w dist

clean:
	rm -rf build dist .pytest_cache src/vvtk_fastcsv/*.so src/vvtk_fastcsv/*.pyd
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

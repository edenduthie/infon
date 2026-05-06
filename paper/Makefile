.PHONY: pdf clean lint arxiv-tarball

pdf:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	cd paper && latexmk -C

lint:
	python3 paper/tools/check_numbers_audit.py

arxiv-tarball:
	bash paper/arxiv-submission/build.sh

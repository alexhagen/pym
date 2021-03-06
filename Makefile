all: tests docs publish

tests: FORCE
	nosetests --with-coverage --cover-html --cover-package=pym --cover-html-dir=docs/_build/html/coverage; \
	nosetests --with-coverage --cover-package=pym; \
	cd ~/code/pym/docs; \
	make coverage; \
	cp _build/coverage/python.txt ./doc_coverage.rst; \
	cd ~/code/pym/

docs: FORCE
	jupyter nbconvert docs/pym_readme.ipynb --to html --template=basic --execute; \
	mv docs/pym_readme.html docs/readme.html; \
	jupyter nbconvert docs/pym_readme.ipynb --to markdown --execute; \
	mv docs/pym_readme.md README.md; \
  cd ~/code/pym/docs; \
	make html

publish: FORCE
	mkdir -p ~/pages/pym; \
	cd ~/pages/pym; \
	git rm -r *; \
	cd ~/code/pym/docs; \
	cp -r _build/html/* ~/pages/pym; \
	cd ~/pages/pym; \
	git add *; \
	touch .nojekyll; \
	git add .nojekyll; \
	git commit -am "$(shell git log -1 --pretty=%B | tr -d '\n')"; \
	git push origin gh-pages; \
	cd ~/code/pym

FORCE:

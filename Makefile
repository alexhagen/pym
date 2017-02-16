all: tests

tests: FORCE
	nosetests --with-coverage --cover-html --cover-package=pym --cover-html-dir=docs/_build/html/coverage
	nosetests --with-coverage --cover-package=pym

docs: FORCE
	mkdir -p ~/pages/pym/docs; \
	cd ~/pages/pym/docs/; \
	git rm -r *; \
	mkdir -p ~/pages/pym/docs; \
	cd ~/code/pym/docs/; \
	make coverage; \
	make html; \
	cp -r _build/html/* ~/pages/pym/docs/; \
	cd ~/pages/pym/docs; \
	git add *; \
	git commit -am "$(shell git log -1 --pretty=%B | tr -d '\n')"; \
	git push origin gh-pages; \
	cd ~/code/pym

FORCE:

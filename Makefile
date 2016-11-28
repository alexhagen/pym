all: docs

docs: FORCE
	mkdir -p ~/pages/pym/docs; \
	cd ~/pages/pym/docs/; \
	git rm -r *; \
	mkdir -p ~/pages/pym/docs; \
	cd ~/code/pym/docs/; \
	make html; \
	cp -r _build/html/* ~/pages/pym/docs/; \
	cd ~/pages/pym/docs; \
	git add *; \
	git commit -am "$(shell git log -1 --pretty=%B | tr -d '\n')"; \
	git push origin gh-pages; \
	cd ~/code/pym

FORCE:


Deployment of hubbard
========================

This document describes the deployment details to perform a version release.

Version release
---------------

The release cycle should be performed like this:

1. Tag the commit with:

        git tag -a "[VERSION]" -m "Releasing [VERSION]"

   Include a complete description of the essential code modifications since last release.

2. Merge `main` branch into `gh-pages` and update `html` documentation:

        git checkout gh-pages
        git merge main
        cd docs
        sphinx-build -b html -d build/doctrees . latest
        git add latest
        git commit -m "docs: update html"

   The build needs `sphinx`, `nbsphinx` and `pandoc`, and an importable
   `hubbard`; `conf.py` runs `conf_prepare.sh` itself.

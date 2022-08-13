# Makefile

SHELL = /bin/bash

#environment
.ONESHELL:


.PHONY: dvc
dvc:
    dvc add data/projects.json
    dvc add data/tags.json
    dvc add data/labeled_projects.json
    dvc push

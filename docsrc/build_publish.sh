#!/usr/bin/env bash

make clean
make html
rm -rf ../docs/*
cp -r _build/html/* ../docs
touch ../docs/.nojekyll
git add ../docs

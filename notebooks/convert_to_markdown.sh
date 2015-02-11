#!/bin/sh

ipython nbconvert *.ipynb --to markdown
mv *_files ../markdown_exports
mv *.md ../markdown_exports

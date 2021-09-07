DOT_FILES := $(wildcard *.dot)
SVG_FILES := $(patsubst %.dot,%.dot.svg,$(DOT_FILES))
PDF_FILE := parall_comput_slides.pdf

.PHONY: all clean svg

all: $(PDF_FILE)

svg: $(SVG_FILES)

%.dot.svg: %.dot
	dot -Tsvg $< -o $@

# See .marprc.yml
$(PDF_FILE): parall_comput_slides.md svg
	marp --pdf parall_comput_slides.md

clean:
	rm -f *.dot.svg
	rm -f $(PDF_FILE)

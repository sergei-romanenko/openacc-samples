DOT_FILES := $(wildcard *.dot)
SVG_FILES := $(patsubst %.dot,%.dot.svg,$(DOT_FILES))

all: svg
# all: pdf

svg: $(SVG_FILES)

# pdf: parall_comput_slides.md svg
# 	marp --allow-local-files --pdf parall_comput_slides.md

%.dot.svg: %.dot
	dot -Tsvg $< -o $@

clean:
	rm -f *.dot.svg
	rm -f parall_comput_slides.pdf

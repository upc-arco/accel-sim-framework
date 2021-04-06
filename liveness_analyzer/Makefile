CC=g++

IDIR=include
ODIR=obj
SRCDIR=src

CFLAGS=-Werror -g -std=c++17 -I$(IDIR)

LIBS=
DEPS = $(shell find $(IDIR) -name '*.h')

SOURCES := $(shell find $(SRCDIR) -name '*.cc')
OBJ := $(patsubst $(SRCDIR)/%,$(ODIR)/%,$(SOURCES:.cc=.o))

liveness_analyzer: main.cc $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

$(ODIR)/%.o: $(SRCDIR)/%.cc $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o liveness_analyzer
CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -Wpedantic -std=c99

# --- main emulator CLI ---
PCA_SRCS = src/vm.c src/asm.c src/main.c
PCA_OBJS = $(PCA_SRCS:.c=.o)

pca: $(PCA_OBJS)
	$(CC) $(CFLAGS) -o $@ $^

# --- exhaustive solver ---
ENUM_SRCS = src/vm.c src/asm.c tasks/spec.c tools/enumerate.c
ENUM_OBJS = $(ENUM_SRCS:.c=.o)

enumerate: $(ENUM_OBJS)
	$(CC) $(CFLAGS) -o $@ $^

# --- shared object for Python bindings ---
libpca.so: src/vm.c src/asm.c tasks/spec.c
	$(CC) $(CFLAGS) -shared -fPIC -o $@ $^

# --- pattern rules ---
src/%.o: src/%.c src/pca.h
	$(CC) $(CFLAGS) -c -o $@ $<

tasks/%.o: tasks/%.c tasks/spec.h src/pca.h
	$(CC) $(CFLAGS) -c -o $@ $<

tools/%.o: tools/%.c src/pca.h tasks/spec.h
	$(CC) $(CFLAGS) -c -o $@ $<

# --- targets ---
all: pca enumerate

clean:
	rm -f src/*.o tasks/*.o tools/*.o pca enumerate libpca.so

test: pca
	@echo "=== fibonacci ==="
	./pca programs/fib.s -q
	@echo "=== echo (5 cycles) ==="
	./pca programs/echo.s -c 5 -q || true
	@echo "=== pid (1000 cycles, timeout expected) ==="
	./pca programs/pid.s -c 1000 -q || true

# Quick synthesis test: find optimal negate, double, add programs
synth: enumerate
	@echo "=== negate (expect 4 insns) ==="
	./enumerate tasks/negate.json -d 2
	@echo ""
	@echo "=== double (expect 4 insns) ==="
	./enumerate tasks/double.json -d 2
	@echo ""
	@echo "=== square (expect 4 insns) ==="
	./enumerate tasks/square.json -d 2
	@echo ""
	@echo "=== add (expect 5 insns) ==="
	./enumerate tasks/add.json -d 2

.PHONY: all clean test synth

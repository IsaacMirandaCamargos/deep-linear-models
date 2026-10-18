PYTHON  ?= python3
VENV    ?= .venv
BIN     := $(VENV)/bin
STAMP   := $(VENV)/.installed

.PHONY: setup mamba clean

setup: $(STAMP)

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip setuptools wheel

# Reinstala só se o venv for novo ou o Makefile mudar.
$(STAMP): $(BIN)/python Makefile
	$(BIN)/pip install torch pandas
	$(BIN)/pip install numpy einops tqdm click pydantic wandb flash-linear-attention \
		rotary-embedding-torch einx transformers seaborn matplotlib rich ray PyYAML
	touch $@

# Opcional: só os mixers Mamba precisam disso; exige nvcc (CUDA toolkit).
mamba: $(STAMP)
	$(BIN)/pip install --no-build-isolation causal_conv1d

clean:
	rm -rf $(VENV)

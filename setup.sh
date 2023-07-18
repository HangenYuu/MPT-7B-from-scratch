conda create -y -n nlpbook python=3.10 "pytorch>=2.0.1" pytorch-cuda=11.8 "transformers>=4.28.1" "datasets>=2.11.0" "tokenizers>=0.13.3" sentencepiece optuna einops matplotlib ipywidgets jupyterlab umap-learn seqeval nltk sacrebleu py7zr nlpaug psutil accelerate -c pytorch -c nvidia/label/cuda-11.8.0 -c huggingface -c conda-forge

conda activate nlpbook

pip install fastxtend bertviz triton
pip install triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python
cd /data/mechinterp/wdd
GPU=1 WDD_LAYER=6 WDD_MODELS=gpt2 WDD_CORPORA=wikitext WDD_RESULTS=/data/mechinterp/wdd/post_review/results/z2_L6 ./post_review/run_pr.sh post_review/pr2_z2_variant.py
GPU=1 WDD_LAYER=2 WDD_MODELS=gpt2 WDD_CORPORA=wikitext WDD_RESULTS=/data/mechinterp/wdd/post_review/results/z2_L2 ./post_review/run_pr.sh post_review/pr2_z2_variant.py
GPU=1 WDD_RANDOM_INIT=1 WDD_LAYER=6 WDD_MODELS=gpt2 WDD_CORPORA=wikitext WDD_RESULTS=/data/mechinterp/wdd/post_review/results/z2_init_L6 ./post_review/run_pr.sh post_review/pr2_z2_variant.py
echo CHAIN1_DONE

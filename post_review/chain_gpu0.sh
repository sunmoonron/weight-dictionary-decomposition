cd /data/mechinterp/wdd
GPU=0 PR_OUT=/data/mechinterp/wdd/post_review/results ./post_review/run_pr.sh post_review/pr1_provenance_random.py
GPU=0 WDD_LAYER=10 WDD_MODELS=gpt2 WDD_CORPORA=wikitext WDD_RESULTS=/data/mechinterp/wdd/post_review/results/z2_L10 ./post_review/run_pr.sh post_review/pr2_z2_variant.py
echo CHAIN0_DONE

cd /data/mechinterp/wdd
GPU=0 PR3_PARTS=omp,oracles,dual PR_OUT=/data/mechinterp/wdd/post_review/results ./post_review/run_pr.sh post_review/pr3_oracles_solvers.py
GPU=0 PR_OUT=/data/mechinterp/wdd/post_review/results ./post_review/run_pr.sh post_review/pr4_sae_features.py
GPU=0 PR3_PARTS=lasso PR3_LASSO_LAM=9.9105 PR_OUT=/data/mechinterp/wdd/post_review/results ./post_review/run_pr.sh post_review/pr3_oracles_solvers.py
echo CHAIN2_0_DONE

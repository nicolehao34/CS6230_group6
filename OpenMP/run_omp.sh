#run the application:\
srun -C cpu -N 1 -n 1 -c 64 --cpu_bind=cores ./kcount $KCOUNT_DATASET_PATH
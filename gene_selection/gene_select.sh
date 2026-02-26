

RANDOM_STATE=1
TEST_SIZE=0.2
TARGET_COLUMN="..."


#for the TCGA_gene.csv, please download in the TCGA website
GENE_EXPRESSION_PATH="data/TCGA_gene.csv" 
METADATA_PATH="data/BRCA.csv"
RESULTS_DIR="results"


N_GENES_TO_SELECT=768




python main.py \
  --random-state "${RANDOM_STATE}" \
  --test-size "${TEST_SIZE}" \
  --target-column "${TARGET_COLUMN}" \
  --gene-expression-path "${GENE_EXPRESSION_PATH}" \
  --metadata-path "${METADATA_PATH}" \
  --results-dir "${RESULTS_DIR}" \
  --n-genes "${N_GENES_TO_SELECT}"

echo "Pipeline finished successfully."
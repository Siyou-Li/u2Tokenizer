export PROJECT_DIR=$(cd "$(dirname "$0")/../" ; pwd -P)

# Add python path
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR

mkdir -p $PROJECT_DIR/output

python src/preprocess/abdomen_atlas_rewrite.py
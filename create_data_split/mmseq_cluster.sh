#!/bin/bash

# Usage: ./mmseqs.sh <parent_directory> <function_file> <component_file> <process_file> <min_seq_id>

# Check if correct number of arguments is provided
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <parent_directory> <function_file> <component_file> <process_file> <min_seq_id>"
    exit 1
fi

PARENT_DIR=$1
FUNCTION_FILE=$2
COMPONENT_FILE=$3
PROCESS_FILE=$4
MIN_SEQ_ID=$5

# Check if the provided parent directory exists
if [ ! -d "$PARENT_DIR" ]; then
    echo "Error: Parent directory $PARENT_DIR does not exist."
    exit 1
fi

# Check if the provided FASTA files exist
if [ ! -f "$FUNCTION_FILE" ]; then
    echo "Error: Function file $FUNCTION_FILE does not exist."
    exit 1
fi

if [ ! -f "$COMPONENT_FILE" ]; then
    echo "Error: Component file $COMPONENT_FILE does not exist."
    exit 1
fi

if [ ! -f "$PROCESS_FILE" ]; then
    echo "Error: Process file $PROCESS_FILE does not exist."
    exit 1
fi

# Create output directories
mkdir -p $PARENT_DIR/function
mkdir -p $PARENT_DIR/component
mkdir -p $PARENT_DIR/process

# Change to parent directory
cd $PARENT_DIR

# Create databases
echo "Creating databases..."
mmseqs createdb $FUNCTION_FILE function/function_redundant_db
if [ $? -ne 0 ]; then
    echo "Error creating function_redundant_db"
    exit 1
fi

mmseqs createdb $COMPONENT_FILE component/component_redundant_db
if [ $? -ne 0 ]; then
    echo "Error creating component_redundant_db"
    exit 1
fi

mmseqs createdb $PROCESS_FILE process/process_redundant_db
if [ $? -ne 0 ]; then
    echo "Error creating process_redundant_db"
    exit 1
fi

# Cluster sequences
echo "Clustering sequences..."
mmseqs cluster function/function_redundant_db function/function_clusters_redundant tmp --min-seq-id $MIN_SEQ_ID -c 0.5 --cov-mode 1
if [ $? -ne 0 ]; then
    echo "Error clustering function_redundant_db"
    exit 1
fi

mmseqs cluster component/component_redundant_db component/component_clusters_redundant tmp --min-seq-id $MIN_SEQ_ID -c 0.5 --cov-mode 1
if [ $? -ne 0 ]; then
    echo "Error clustering component_redundant_db"
    exit 1
fi

mmseqs cluster process/process_redundant_db process/process_clusters_redundant tmp --min-seq-id $MIN_SEQ_ID -c 0.5 --cov-mode 1
if [ $? -ne 0 ]; then
    echo "Error clustering process_redundant_db"
    exit 1
fi

# Create TSV files
echo "Creating TSV files..."
mmseqs createtsv function/function_redundant_db function/function_redundant_db function/function_clusters_redundant function/function_clusters_redundant.tsv
if [ $? -ne 0 ]; then
    echo "Error creating function_clusters_redundant.tsv"
    exit 1
fi

mmseqs createtsv component/component_redundant_db component/component_redundant_db component/component_clusters_redundant component/component_clusters_redundant.tsv
if [ $? -ne 0 ]; then
    echo "Error creating component_clusters_redundant.tsv"
    exit 1
fi

mmseqs createtsv process/process_redundant_db process/process_redundant_db process/process_clusters_redundant process/process_clusters_redundant.tsv
if [ $? -ne 0 ]; then
    echo "Error creating process_clusters_redundant.tsv"
    exit 1
fi

# Clean up temporary files
#echo "Cleaning up temporary files..."
#rm -rf tmp

echo "All tasks completed successfully."


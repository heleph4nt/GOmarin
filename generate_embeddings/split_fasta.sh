#!/bin/bash

awk -v seqs_per_file=30000 '
  /^>/ {  # header line
    if (NR!=1 && seq_count % seqs_per_file == 0) {  # new file after 1000 seqs
      file_num++
    }
    seq_count++
    file = sprintf("chunk_%03d.fasta", file_num)
  }
  { print > file }
' uniprot_taxonomy_bacteria_20_11_2025.fasta

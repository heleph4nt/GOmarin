#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os 
import networkx as nx
import obonet
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO
from collections import defaultdict
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from collections import Counter
import subprocess
import requests, sys
import json


np.random.seed(42)

os.chdir(os.path.expanduser('~/PlasmoFP_public/data/raw_data_from_uniprot'))


# MA: Manual annotation
# AA: Automatic annotation

SAR_MA_AA_path = "SAR_MA_AA.tsv" 
Plasmodium_AA_path = "Plasmodium_AA.tsv"
Plasmodium_MA_path = "Plasmodium_MA.tsv"
SAR_MA_path = "SAR_MA.tsv"

#read in the datasets
SAR_MA_AA = pd.read_csv(SAR_MA_AA_path, sep="\t")
Plasmodium_AA = pd.read_csv(Plasmodium_AA_path, sep="\t")
Plasmodium_MA = pd.read_csv(Plasmodium_MA_path, sep="\t")
SAR_MA = pd.read_csv(SAR_MA_path, sep="\t")
print("SAR_MA_AA:", SAR_MA_AA.shape)
print("Plasmodium_AA:", Plasmodium_AA.shape)
print("Plasmodium_MA:", Plasmodium_MA.shape)
print("SAR_MA:", SAR_MA.shape)

#assert columns are the same
assert SAR_MA_AA.columns.tolist() == Plasmodium_AA.columns.tolist() == Plasmodium_MA.columns.tolist() == SAR_MA.columns.tolist()

SAR_MA_AA["GOAssertion"] = np.nan 
Plasmodium_AA["GOAssertion"] = "AA"

SAR_MA_AA.loc[SAR_MA_AA["Entry"].isin(SAR_MA["Entry"]), "GOAssertion"] = "MA"
SAR_MA_AA.loc[SAR_MA_AA["GOAssertion"].isnull(), "GOAssertion"] = "AA"

total_SAR = pd.concat([SAR_MA_AA, Plasmodium_AA])
total_SAR_function = total_SAR[total_SAR["Gene Ontology (molecular function)"].notnull()]
total_SAR_process = total_SAR[total_SAR["Gene Ontology (biological process)"].notnull()]
total_SAR_component = total_SAR[total_SAR["Gene Ontology (cellular component)"].notnull()]

print(total_SAR_function.shape, total_SAR_process.shape, total_SAR_component.shape)

total_SAR_records = create_seq_records(total_SAR)
print(len(total_SAR_records))

SeqIO.write(total_SAR_records, "total_SAR.fasta", "fasta")

seq_records_SAR_function = create_seq_records(total_SAR_function)
seq_records_SAR_process = create_seq_records(total_SAR_process)
seq_records_SAR_component = create_seq_records(total_SAR_component)

assert len(seq_records_SAR_function) == total_SAR_function.shape[0]
assert len(seq_records_SAR_process) == total_SAR_process.shape[0]
assert len(seq_records_SAR_component) == total_SAR_component.shape[0]

SeqIO.write(seq_records_SAR_function, "total_SAR_function_for_reduce.fasta", "fasta")
SeqIO.write(seq_records_SAR_process, "total_SAR_process_for_reduce.fasta", "fasta")
SeqIO.write(seq_records_SAR_component, "total_SAR_component_for_reduce.fasta", "fasta")

subprocess.run(["../../scripts/mmseq_cluster.sh", "reduced_90", "total_SAR_function_for_reduce.fasta", "total_SAR_component_for_reduce.fasta", "total_SAR_process_for_reduce.fasta", "0.9"])
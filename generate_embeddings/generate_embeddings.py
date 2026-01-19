#!/usr/bin/env python3

"""
This script generates protein embeddings using TM-Vec models.
Supports processing single FASTA files or directories containing multiple FASTA files.
Outputs embeddings in multiple formats for maximum compatibility:
  - JSON: Maximum compatibility, works across NumPy versions
  - NPZ: NumPy compressed format, efficient and more compatible than pickle
  - Pickle: Standard Python format with improved compatibility (protocol 4)
  - NumPy array: Raw array format for backward compatibility

## Required Model Files

Download the TM-Vec model files:
- `tm_vec_cath_model.ckpt` - TM-Vec model checkpoint
- `tm_vec_cath_model_params.json` - TM-Vec configuration file

## Usage Examples

### Process a single FASTA file
python generate_embeddings.py \
    --input protein_sequences.fasta \
    --output ./embeddings/ \
    --tm_vec_model tm_vec_cath_model.ckpt \
    --tm_vec_config tm_vec_cath_model_params.json

### Process a directory of FASTA files
python generate_embeddings.py \
    --input ./fasta_files/ \
    --output ./embeddings/ \
    --tm_vec_model tm_vec_cath_model.ckpt \
    --tm_vec_config tm_vec_cath_model_params.json

### Use default model paths (if files are in current directory)
python generate_embeddings.py \
    --input ./fasta_files/ \
    --output ./embeddings/

### Force CPU usage
python generate_embeddings.py \
    --input ./fasta_files/ \
    --output ./embeddings/ \
    --device cpu

### Generate only JSON format (most compatible)
python generate_embeddings.py \
    --input ./fasta_files/ \
    --output ./embeddings/ \
    --output_format json

### Generate only NPZ format (efficient and compatible)
python generate_embeddings.py \
    --input ./fasta_files/ \
    --output ./embeddings/ \
    --output_format npz

"""

import argparse
import logging
import gc
import pickle
import json
import base64
import sys
from pathlib import Path
from typing import List, Dict, Union, Tuple
import os # HELENA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # HELENA
import numpy as np
import torch
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from transformers import T5EncoderModel, T5Tokenizer
from tqdm import tqdm

from tm_vec.embed_structure_model import trans_basic_block, trans_basic_block_Config
from tm_vec.tm_vec_utils import encode


def save_embeddings_json(embeddings_dict: Dict[str, np.ndarray], output_path: Path) -> None:
    """Save embeddings as JSON with base64-encoded arrays for maximum compatibility."""
    json_dict = {}
    for seq_id, embedding in embeddings_dict.items():
        # Convert numpy array to base64-encoded string
        embedding_bytes = embedding.astype(np.float32).tobytes()
        embedding_b64 = base64.b64encode(embedding_bytes).decode('utf-8')
        json_dict[seq_id] = {
            'embedding': embedding_b64,
            'shape': embedding.shape,
            'dtype': str(embedding.dtype)
        }
    
    with open(output_path, 'w') as f:
        json.dump(json_dict, f, indent=2)


def load_embeddings_json(json_path: Path) -> Dict[str, np.ndarray]:
    """Load embeddings from JSON format."""
    with open(json_path, 'r') as f:
        json_dict = json.load(f)
    
    embeddings_dict = {}
    for seq_id, data in json_dict.items():
        # Decode base64 string back to numpy array
        embedding_bytes = base64.b64decode(data['embedding'].encode('utf-8'))
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        embedding = embedding.reshape(data['shape'])
        embeddings_dict[seq_id] = embedding
    
    return embeddings_dict


def save_embeddings_npz(embeddings_dict: Dict[str, np.ndarray], output_path: Path) -> None:
    """Save embeddings as NPZ file (NumPy's native compressed format).
    More efficient than JSON but still more compatible than pickle."""
    np.savez_compressed(output_path, **embeddings_dict)

    
def load_embeddings_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    npz_data = np.load(npz_path)
    return {key: npz_data[key] for key in npz_data.files}


def save_embeddings_pickle_safe(embeddings_dict: Dict[str, np.ndarray], output_path: Path) -> None:
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings_dict, f, protocol=4)


def setup_logging(output_dir: Path, verbose: bool = False) -> logging.Logger:
    log_file = output_dir / "embedding_generation.log"
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Force reconfiguration
    )
    
    logger = logging.getLogger(__name__)
    
    print(f"Embedding generation started!")
    print(f"Output directory: {output_dir}")
    print(f"Log file: {log_file}")
    print(f"Verbose mode: {'ON' if verbose else 'OFF'}")
    print("="*60)
    
    return logger


def validate_inputs(input_path: Path, tm_vec_model: Path, tm_vec_config: Path) -> bool:
    """Validate input paths and model files exist."""
    if not input_path.exists():
        logging.error(f"Input path does not exist: {input_path}")
        return False
    
    if not tm_vec_model.exists():
        logging.error(f"TM-Vec model checkpoint not found: {tm_vec_model}")
        return False
    
    if not tm_vec_config.exists():
        logging.error(f"TM-Vec config file not found: {tm_vec_config}")
        return False
    
    return True


def get_fasta_files(input_path: Path) -> List[Path]:
    """Get FASTA files from input path."""
    if input_path.is_file():
        if input_path.suffix.lower() in ['.fasta', '.fa', '.fas']:
            return [input_path]
        else:
            logging.warning(f"File {input_path} does not have a recognized FASTA extension")
            return [input_path]  # Still try to process it
    
    elif input_path.is_dir():
        fasta_files = []
        for ext in ['*.fasta', '*.fa', '*.fas']:
            fasta_files.extend(input_path.glob(ext))
        
        if not fasta_files:
            logging.error(f"No FASTA files found in directory: {input_path}")
            return []
        
        return sorted(fasta_files)
    
    else:
        logging.error(f"Input path is neither file nor directory: {input_path}")
        return []


def load_models(tm_vec_model_path: Path, tm_vec_config_path: Path, device: torch.device) -> Tuple[object, object, object]:
    """Load and initialize ProtT5 and TM-Vec models."""
    print("Loading ProtT5 tokenizer and model...")
    logging.info("Loading ProtT5 tokenizer and model...")
    
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    
    model = model.to(device)
    model = model.eval()
    
    print("Loading TM-Vec model...")
    logging.info("Loading TM-Vec model...")
    
    tm_vec_config = trans_basic_block_Config.from_json(str(tm_vec_config_path))
    model_deep = trans_basic_block.load_from_checkpoint(
        str(tm_vec_model_path), 
        config=tm_vec_config
    )
    model_deep = model_deep.to(device)
    model_deep = model_deep.eval()
    
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    print(f"Models loaded successfully on device: {device}")
    logging.info(f"Models loaded successfully on device: {device}")
    
    return model_deep, model, tokenizer


def process_fasta_file(
    fasta_path: Path,
    output_dir: Path,
    model_deep: object,
    model: object,
    tokenizer: object,
    device: torch.device,
    output_format: str = "all"
) -> bool:
    """
    Process a single FASTA file and generate embeddings.
    
    Args:
        fasta_path: Path to input FASTA file
        output_dir: Directory for output files
        model_deep: TM-Vec model instance
        model: ProtT5 model instance
        tokenizer: ProtT5 tokenizer instance
        device: PyTorch device
        
    Returns:
        True if processing successful, False otherwise
    """
    try:
        print(f"Processing file: {fasta_path.name}")
        logging.info(f"Processing file: {fasta_path.name}")
        
        sequences = list(SeqIO.parse(str(fasta_path), "fasta"))
        
        if not sequences:
            print(f"No sequences found in {fasta_path}")
            logging.warning(f"No sequences found in {fasta_path}")
            return False
        
        seq_ids = [rec.id for rec in sequences]
        
        print(f"Found {len(sequences)} sequences in {fasta_path.name}")
        logging.info(f"Found {len(sequences)} sequences in {fasta_path.name}")
        
        print("Generating embeddings... (this may take a while)")
        logging.info("Generating embeddings...")
        embeddings = encode(sequences, model_deep, model, tokenizer, device)
        
        embeddings_dict = dict(zip(seq_ids, embeddings))
        
        output_files = []
        
        if output_format in ["pickle", "all"]:
            pkl_output = output_dir / f"{fasta_path.stem}_embeddings.pkl"
            save_embeddings_pickle_safe(embeddings_dict, pkl_output)
            output_files.append(pkl_output.name)
        
        if output_format in ["json", "all"]:
            json_output = output_dir / f"{fasta_path.stem}_embeddings.json"
            save_embeddings_json(embeddings_dict, json_output)
            output_files.append(json_output.name)
        
        if output_format in ["npz", "all"]:
            npz_output = output_dir / f"{fasta_path.stem}_embeddings.npz"
            save_embeddings_npz(embeddings_dict, npz_output)
            output_files.append(npz_output.name)
        
        npy_output = output_dir / f"{fasta_path.stem}_embeddings.npy"
        np.save(npy_output, embeddings)
        output_files.append(npy_output.name)
        
        print(f"Successfully processed {fasta_path.name}: {len(sequences)} sequences")
        print(f"Output files: {', '.join(output_files)}")
        print(f"Embedding shape: {embeddings.shape}")
        
        logging.info(
            f"Successfully processed {fasta_path.name}: "
            f"{len(sequences)} sequences → {', '.join(output_files)}"
        )
        logging.info(f"Embedding shape: {embeddings.shape}")
        
        del embeddings
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"Error processing {fasta_path}: {str(e)}")
        logging.error(f"Error processing {fasta_path}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate protein embeddings using TM-Vec and ProtT5 models"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input FASTA file or directory containing FASTA files"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for embedding files"
    )
    
    parser.add_argument(
        "--tm_vec_model",
        type=Path,
        help="Path to TM-Vec model checkpoint file (.ckpt)"
    )
    
    parser.add_argument(
        "--tm_vec_config",
        type=Path,
        help="Path to TM-Vec configuration file (.json)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for computation (default: auto-detect)"
    )
    
    parser.add_argument(
        "--output_format",
        type=str,
        default="all",
        choices=["pickle", "json", "npz", "all"],
        help="Output format for embeddings (default: all formats)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(args.output, args.verbose)
    
    if not args.tm_vec_model:
        args.tm_vec_model = Path("tm_vec_cath_model.ckpt")
        
    if not args.tm_vec_config:
        args.tm_vec_config = Path("tm_vec_cath_model_params.json")
    
    if not validate_inputs(args.input, args.tm_vec_model, args.tm_vec_config):
        sys.exit(1)
    
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    logging.info(f"Using device: {device}")
    
    fasta_files = get_fasta_files(args.input)
    if not fasta_files:
        print("No FASTA files found!")
        sys.exit(1)
    
    print(f"Found {len(fasta_files)} FASTA file(s) to process:")
    for i, f in enumerate(fasta_files, 1):
        print(f"   {i}. {f.name}")
    print()
    
    logging.info(f"Found {len(fasta_files)} FASTA file(s) to process")
    
    try:
        model_deep, model, tokenizer = load_models(
            args.tm_vec_model, 
            args.tm_vec_config, 
            device
        )
    except Exception as e:
        print(f"Failed to load models: {str(e)}")
        logging.error(f"Failed to load models: {str(e)}")
        sys.exit(1)
    
    successful_files = 0
    failed_files = 0
    
    print("Starting processing...")
    print("=" * 60)
    
    for i, fasta_file in enumerate(fasta_files, 1):
        print(f"[{i}/{len(fasta_files)}] Processing: {fasta_file.name}")
        
        success = process_fasta_file(
            fasta_file,
            args.output,
            model_deep,
            model,
            tokenizer,
            device,
            args.output_format
        )
        
        if success:
            successful_files += 1
            print(f"[{i}/{len(fasta_files)}] Completed: {fasta_file.name}")
        else:
            failed_files += 1
            print(f"[{i}/{len(fasta_files)}] Failed: {fasta_file.name}")
        
        print(f"Progress: {successful_files} success, {failed_files} failed")
    
    print("\n" + "=" * 60)
    print(f"FINAL RESULTS:")
    print(f"Successfully processed: {successful_files} files")
    print(f"Failed to process: {failed_files} files")
    print(f"Output directory: {args.output}")
    print("=" * 60)
    
    logging.info("=" * 50)
    logging.info(f"Successfully processed: {successful_files} files")
    logging.info(f"Failed to process: {failed_files} files")
    logging.info(f"Output directory: {args.output}")
    logging.info("=" * 50)
    
    if failed_files > 0:
        sys.exit(1)


if __name__ == "__main__":
    main() 
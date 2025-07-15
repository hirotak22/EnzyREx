import argparse
import os
import pandas as pd
import torch


def process_enzymecnn_result(dataset, pred_scores, output_path, threshold):
    dataset['Score'] = pred_scores.tolist()
    dataset['Enzyme'] = (dataset['Score'] >= threshold).astype(int)
    dataset.to_csv(output_path, sep='\t', header=True, index=False)
    
    root, ext = os.path.splitext(output_path)
    dataset.query('Enzyme == 1').to_csv(f'{root}.enzyme{ext}', sep='\t', header=True, index=False)
    return None


if __name__ == '__main__':
    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--pred-scores', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--threshold', default=0.5, type=float)
    
    # Parse inputs
    args = parser.parse_args()
    
    # Load input dataset
    dataset = pd.read_table(args.dataset)
    
    # Load pred scores obtained by EnzymeCNN
    pred_scores = torch.load(args.pred_scores)
    
    # Process EnzymeCNN result
    process_enzymecnn_result(dataset, pred_scores, args.output, args.threshold)

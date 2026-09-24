import argparse
import pandas as pd
import torch


def process_enzymeclip_result(protein_dataset, reaction_dataset, pred_scores, output_path, threshold):
    df_list = []
    for i, rxn_id in enumerate(reaction_dataset['ID']):
        indices = torch.nonzero(pred_scores[i] >= threshold).reshape(-1).numpy()
        if len(indices) != 0:
            df = pd.DataFrame([rxn_id]*len(indices), columns=['Reaction_ID'])
            df['Protein_ID'] = protein_dataset['ID'][indices].tolist()
            df['Score'] = pred_scores[i][indices].tolist()
            df_list.append(df)
    
    df_output = pd.concat(df_list, ignore_index=True)
    df_output.to_csv(output_path, sep='\t', header=True, index=False)
    return None


if __name__ == '__main__':
    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--protein-dataset', type=str, required=True)
    parser.add_argument('--reaction-dataset', type=str, required=True)
    parser.add_argument('--pred-scores', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--threshold', default=0.386024, type=float)
    
    # Parse inputs
    args = parser.parse_args()
    
    # Load input datasets
    protein_dataset = pd.read_table(args.protein_dataset)
    reaction_dataset = pd.read_table(args.reaction_dataset)
    
    # Load pred scores (cosine similarity matrix) obtained by EnzymeCLIP
    pred_scores = torch.load(args.pred_scores)
    
    # Process EnzymeCLIP result
    process_enzymeclip_result(protein_dataset, reaction_dataset, pred_scores, args.output, args.threshold)

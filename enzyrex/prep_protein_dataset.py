import argparse
import pandas as pd


def parse_db(dbpath):
    # Amino acid sequence
    with open(dbpath) as f_aa:
        aa_sequences = [line.strip('\0') for line in f_aa.read().splitlines()[:-1]]
    
    # 3Di sequence
    with open(f'{dbpath}_ss') as f_sa:
        sa_sequences = [line.strip('\0') for line in f_sa.read().splitlines()[:-1]]
    
    # Source file name
    lookup_table = pd.read_table(f'{dbpath}.lookup', names=['idx', 'id', 'offset'])
        
    print(len(aa_sequences), len(sa_sequences), len(lookup_table))
    
    # Combine amino acid sequence and 3Di sequence
    combined_sequences = [''.join([aa + sa.lower() for aa, sa in zip(aa_seq, sa_seq)])
                          for aa_seq, sa_seq in zip(aa_sequences, sa_sequences)]
    
    db_table = lookup_table[['id']].copy()
    db_table['aa_sequence'] = aa_sequences
    db_table['struct_sequence'] = sa_sequences
    db_table['combined_sequence'] = combined_sequences
    print(db_table.shape)
    
    return db_table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess protein dataset')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    output_table = parse_db(args.input)
    output_table.to_csv(args.output, sep='\t', header=True, index=False)

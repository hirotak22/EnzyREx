import argparse
import os
import random
import functools
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
torch.set_float32_matmul_precision('medium')


# Ensure reproducibility
def set_seed(seed=42, debug=False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if debug:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# EnzymeCNN models
## TemporalBlock
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv1.bias.data.fill_(0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        self.conv2.bias.data.fill_(0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
            self.downsample.bias.data.fill_(0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


## Stacked TemporalBlocks (Core component of EnzymeCNN)
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dilation_rate, dropout):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = dilation_rate ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size // 2) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


## Multilayer perceptron
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout):
        super().__init__()
        layers = []
        for in_dim, out_dim in zip([input_dim]+hidden_dims, hidden_dims):
            layers += [weight_norm(nn.Linear(in_dim, out_dim)),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
        layers.append(weight_norm(nn.Linear(hidden_dims[-1], output_dim)))
        
        self.network = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(0, 0.01)
                layer.bias.data.fill_(0.01)

    def forward(self, x):
        return self.network(x)


## EnzymeCNN
class MultiTCNClassifier(nn.Module):
    def __init__(self, TCN1, TCN2, classifier):
        super().__init__()
        self.TCN1 = TCN1
        self.TCN2 = TCN2
        self.classifier = classifier
    
    def forward(self, x1, x2):
        output_1 = self.TCN1(x1)
        embedding_1 = torch.mean(output_1, dim=2)
        output_2 = self.TCN2(x2)
        embedding_2 = torch.mean(output_2, dim=2)
        embedding = torch.cat([embedding_1, embedding_2], dim=1)
        return self.classifier(embedding)


## EnzymeCNN ablation models (EnzymeCNN-AA and EnzymeCNN-3Di)
class TCNClassifier(nn.Module):
    def __init__(self, TCN, classifier):
        super(TCNClassifier, self).__init__()
        self.TCN = TCN
        self.classifier = classifier
    
    def forward(self, x):
        output = self.TCN(x)
        embedding = torch.mean(output, dim=2)
        return self.classifier(embedding)


# Data preprocessing
## Amino acid encoding (Reference: https://github.com/google-research/proteinfer)
### Make vocabulary
AMINO_ACID_VOCABULARY = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
]
PFAM_GAP_CHARACTER = '.'

### Other characters representing amino-acids not in AMINO_ACID_VOCABULARY.
ADDITIONAL_AA_VOCABULARY = [
    # Substitutions
    'U',
    'O',
    # Ambiguous Characters
    'B',
    'Z',
    'J',
    'X',
    # Gap Character
    PFAM_GAP_CHARACTER
]

### Vocab of all possible tokens in a valid input sequence
FULL_AA_RESIDUE_VOCAB = AMINO_ACID_VOCABULARY + ADDITIONAL_AA_VOCABULARY

### Map AA characters to their index in FULL_RESIDUE_VOCAB.
AA_RESIDUE_TO_INT = {aa: idx for idx, aa in enumerate(FULL_AA_RESIDUE_VOCAB)}

### Convert amino acid residues to indices
def aa_residues_to_indices(amino_acid_residues):
    return [AA_RESIDUE_TO_INT[c] for c in amino_acid_residues]

### Make reference matrix
@functools.lru_cache(maxsize=1)
def build_aa_one_hot_encodings():
    """Create array of one-hot embeddings.

    Row `i` of the returned array corresponds to the one-hot embedding of amino acid FULL_RESIDUE_VOCAB[i].

    Returns:
        np.array of shape `[len(FULL_RESIDUE_VOCAB), 20]`.
    """
    base_encodings = np.eye(len(AMINO_ACID_VOCABULARY))
    to_aa_index = AMINO_ACID_VOCABULARY.index
    
    special_mappings = {
        'B': .5 * (base_encodings[to_aa_index('D')] + base_encodings[to_aa_index('N')]),
        'Z': .5 * (base_encodings[to_aa_index('E')] + base_encodings[to_aa_index('Q')]),
        'J': .5 * (base_encodings[to_aa_index('I')] + base_encodings[to_aa_index('L')]),
        'X': np.ones(len(AMINO_ACID_VOCABULARY)) / len(AMINO_ACID_VOCABULARY),
        PFAM_GAP_CHARACTER: np.zeros(len(AMINO_ACID_VOCABULARY)),
    }
    special_mappings['U'] = base_encodings[to_aa_index('C')]
    special_mappings['O'] = special_mappings['X']
    special_encodings = np.array([special_mappings[c] for c in ADDITIONAL_AA_VOCABULARY])
    return np.concatenate((base_encodings, special_encodings), axis=0)

### One-hot-encoding
def aa_residues_to_one_hot(amino_acid_residues):
    """Given a sequence of amino acids, return one hot array.

    Supports ambiguous amino acid characters B, Z, J, and X by distributing evenly over possible values,
        e.g. an 'X' gets mapped to [.05, .05, ... , .05].

    Supports gaps and pads with the '.' and '-' characters; which are mapped to the zero vector.

    Args:
        amino_acid_residues: string. consisting of characters from AMINO_ACID_VOCABULARY

    Returns:
        A numpy array of shape (len(amino_acid_residues), len(AMINO_ACID_VOCABULARY)).

    Raises:
        KeyError: if amino_acid_residues has a character not in FULL_RESIDUE_VOCAB.
    """
    residue_encodings = build_aa_one_hot_encodings()
    int_sequence = aa_residues_to_indices(amino_acid_residues)
    return residue_encodings[int_sequence]

## 3Di alphabet encoding
### Make vocabulary
STRUCT_VOCABULARY = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
]
PFAM_GAP_CHARACTER = '.'

### Other characters representing 3Di alphabets not in STRUCT_VOCABULARY.
ADDITIONAL_STRUCT_VOCABULARY = [
    # Ambiguous Characters
    'X',
    # Gap Character
    PFAM_GAP_CHARACTER
]

### Vocab of all possible tokens in a valid input sequence
FULL_STRUCT_RESIDUE_VOCAB = STRUCT_VOCABULARY + ADDITIONAL_STRUCT_VOCABULARY

### Map 3Di alphabet characters to their index in FULL_RESIDUE_VOCAB.
STRUCT_RESIDUE_TO_INT = {alphabet: idx for idx, alphabet in enumerate(FULL_STRUCT_RESIDUE_VOCAB)}

### Convert sequence to indices
def struct_residues_to_indices(struct_residues):
    return [STRUCT_RESIDUE_TO_INT[c] for c in struct_residues]

### Make reference matrix
@functools.lru_cache(maxsize=1)
def build_struct_one_hot_encodings():
    """Create array of one-hot embeddings.

    Row `i` of the returned array corresponds to the one-hot embedding of amino acid FULL_RESIDUE_VOCAB[i].

    Returns:
        np.array of shape `[len(FULL_RESIDUE_VOCAB), 20]`.
    """
    base_encodings = np.eye(len(STRUCT_VOCABULARY))
        
    special_mappings = {
        'X': np.ones(len(STRUCT_VOCABULARY)) / len(STRUCT_VOCABULARY),
        PFAM_GAP_CHARACTER: np.zeros(len(STRUCT_VOCABULARY)),
    }
    special_encodings = np.array([special_mappings[c] for c in ADDITIONAL_STRUCT_VOCABULARY])
    return np.concatenate((base_encodings, special_encodings), axis=0)

### One-hot-encoding
def struct_residues_to_one_hot(struct_residues):
    """Given a sequence of amino acids, return one hot array.

    Supports ambiguous amino acid characters B, Z, J, and X by distributing evenly over possible values,
        e.g. an 'X' gets mapped to [.05, .05, ... , .05].

    Supports gaps and pads with the '.' and '-' characters; which are mapped to the zero vector.

    Args:
        struct_residues: string. consisting of characters from STRUCT_VOCABULARY

    Returns:
        A numpy array of shape (len(struct_residues), len(STRUCT_VOCABULARY)).

    Raises:
        KeyError: if struct_residues has a character not in FULL_STRUCT_RESIDUE_VOCAB.
    """
    residue_encodings = build_struct_one_hot_encodings()
    int_sequence = struct_residues_to_indices(struct_residues)
    return residue_encodings[int_sequence]


## Dataset
class ProteinCombinedDataset(Dataset):
    def __init__(self, aa_sequences, struct_sequences, labels):
        self.aa_sequence = aa_sequences
        self.struct_sequence = struct_sequences
        self.label = labels

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        aa_sequence = self.aa_sequence[idx]
        struct_sequence = self.struct_sequence[idx]
        label = self.label[idx]
        return aa_sequence, struct_sequence, label

class ProteinSingleDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequence = sequences
        self.label = labels

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        sequence = self.sequence[idx]
        label = self.label[idx]
        return sequence, label

## DataLoader
def padding(sequence, max_len):
    seq_len = len(sequence)
    return sequence + PFAM_GAP_CHARACTER*(max_len-seq_len)

def batch_preproccessing_combined(batch):
    aa_sequences, struct_sequences, labels= list(zip(*batch))
    # amino acid sequence
    seq_len = [len(sequence) for sequence in aa_sequences]
    max_len = max(seq_len)
    encoded_aa_sequences = np.array([aa_residues_to_one_hot(padding(sequence, max_len)) for sequence in aa_sequences])
    encoded_aa_sequences = torch.tensor(encoded_aa_sequences, dtype=torch.float32).transpose(1, 2)
    # 3Di sequence
    seq_len = [len(sequence) for sequence in struct_sequences]
    max_len = max(seq_len)
    encoded_struct_sequences = np.array([struct_residues_to_one_hot(padding(sequence, max_len)) for sequence in struct_sequences])
    encoded_struct_sequences = torch.tensor(encoded_struct_sequences, dtype=torch.float32).transpose(1, 2)
    # label
    labels = torch.tensor(labels, dtype=torch.float32)
    return encoded_aa_sequences, encoded_struct_sequences, labels

def batch_preproccessing_aa(batch):
    sequences, labels= list(zip(*batch))
    # sequence
    seq_len = [len(sequence) for sequence in sequences]
    max_len = max(seq_len)
    encoded_sequences = np.array([aa_residues_to_one_hot(padding(sequence, max_len)) for sequence in sequences])
    encoded_sequences = torch.tensor(encoded_sequences, dtype=torch.float32).transpose(1, 2)
    # label
    labels = torch.tensor(labels, dtype=torch.float32)
    return encoded_sequences, labels

def batch_preproccessing_struct(batch):
    sequences, labels= list(zip(*batch))
    # sequence
    seq_len = [len(sequence) for sequence in sequences]
    max_len = max(seq_len)
    encoded_sequences = np.array([struct_residues_to_one_hot(padding(sequence, max_len)) for sequence in sequences])
    encoded_sequences = torch.tensor(encoded_sequences, dtype=torch.float32).transpose(1, 2)
    # label
    labels = torch.tensor(labels, dtype=torch.float32)
    return encoded_sequences, labels


## Dataloader creation
def make_combined_dataloader(dataset_path, batch_size, shuffle, threads):
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(0)
    
    df = pd.read_table(dataset_path)
    if 'label' not in df.columns:
        df['label'] = -1
    dataset = ProteinCombinedDataset(df['aa_sequence'].to_list(),
                                     df['struct_sequence'].to_list(),
                                     df['label'].to_list())
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=threads,
                            collate_fn=batch_preproccessing_combined,
                            worker_init_fn=seed_worker,
                            generator=g)
    return dataloader

def make_aa_dataloader(dataset_path, batch_size, shuffle, threads):
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(0)
    
    df = pd.read_table(dataset_path)
    if 'label' not in df.columns:
        df['label'] = -1
    dataset = ProteinSingleDataset(df['aa_sequence'].to_list(),
                                   df['label'].to_list())
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=threads,
                            collate_fn=batch_preproccessing_aa,
                            worker_init_fn=seed_worker,
                            generator=g)
    return dataloader

def make_struct_dataloader(dataset_path, batch_size, shuffle, threads):
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(0)
    
    df = pd.read_table(dataset_path)
    if 'label' not in df.columns:
        df['label'] = -1
    dataset = ProteinSingleDataset(df['struct_sequence'].to_list(),
                                   df['label'].to_list())
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=threads,
                            collate_fn=batch_preproccessing_struct,
                            worker_init_fn=seed_worker,
                            generator=g)
    return dataloader


# EnzymeCNNs training and evaluation
## Compute metrics
def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray):
    y_pred = (y_pred_proba >= 0.5).astype(float)
    _acc = (y_true == y_pred).mean().item()
    _f1 = f1_score(y_true, y_pred)
    _mcc = matthews_corrcoef(y_true, y_pred)
    _auc = roc_auc_score(y_true, y_pred_proba)
    return _acc, _f1, _mcc, _auc

## EnzymeCNN
### Training
def enzymecnn_training(model, dataloader, optimizer, criterion, gradient_clip, epoch_idx, device, log_interval=100):
    model.train()
    train_loss, total_loss = 0, 0
    predicted_probability_list, label_list = [], []
    for idx, (encoded_aa_sequences, encoded_strucr_sequences, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        logits = model(encoded_aa_sequences.to(device), encoded_strucr_sequences.to(device)).reshape(-1)
        loss = criterion(logits, labels.to(device))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        
        predicted_probability = torch.sigmoid(logits).cpu()
        predicted_probability_list.append(predicted_probability)
        label_list.append(labels)
        train_loss += loss.item()
        total_loss += loss.item() / len(dataloader)
        
        if (idx+1) % log_interval == 0 and idx > 0:
            print('| epoch {:3d} | {:5d}/{:5d} batches | loss {:8.3f}'.format(epoch_idx, (idx+1), len(dataloader), train_loss))
            train_loss = 0
    
    y_pred_proba = torch.cat(predicted_probability_list).detach().numpy().copy()
    y_true = torch.cat(label_list).detach().numpy().copy()
    train_acc, train_f1, train_mcc,  train_auc = compute_metrics(y_true, y_pred_proba)
    return total_loss, train_acc, train_f1, train_mcc, train_auc

### Evaluation
def enzymecnn_evaluate(model, dataloader, criterion, device, return_output=False):
    model.eval()
    total_loss = 0
    predicted_probability_list, label_list = [], []
    with torch.no_grad():
        for idx, (encoded_aa_sequences, encoded_strucr_sequences, labels) in enumerate(dataloader):
            logits = model(encoded_aa_sequences.to(device), encoded_strucr_sequences.to(device)).reshape(-1)
            loss = criterion(logits, labels.to(device))
            total_loss += loss.item() / len(dataloader)
            predicted_probability = torch.sigmoid(logits).cpu()
            predicted_probability_list.append(predicted_probability)
            label_list.append(labels)
    
    y_pred_proba = torch.cat(predicted_probability_list).detach().numpy().copy()
    y_true = torch.cat(label_list).detach().numpy().copy()
    
    if not return_output:
        eval_acc, eval_f1, eval_mcc,  eval_auc = compute_metrics(y_true, y_pred_proba)
        return total_loss, eval_acc, eval_f1, eval_mcc, eval_auc
    else:
        return total_loss, eval_acc, eval_f1, eval_mcc, eval_auc, y_pred_proba, y_true

### Inference
def enzymecnn_inference(model, dataloader, device):
    predicted_probability_list = []
    with torch.no_grad():
        for idx, (encoded_aa_sequences, encoded_strucr_sequences, labels) in enumerate(dataloader):
            logits = model(encoded_aa_sequences.to(device), encoded_strucr_sequences.to(device)).reshape(-1)
            predicted_probability = torch.sigmoid(logits).cpu()
            predicted_probability_list.append(predicted_probability)
    
    y_pred_proba = torch.cat(predicted_probability_list).detach().numpy().copy()
    return y_pred_proba


## EnzymeCNN ablations
### Training
def enzymecnn_ablation_training(model, dataloader, optimizer, criterion, gradient_clip, epoch_idx, device, log_interval=100):
    model.train()
    train_loss, total_loss = 0, 0
    predicted_probability_list, label_list = [], []
    for idx, (encoded_sequences, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        logits = model(encoded_sequences.to(device)).reshape(-1)
        loss = criterion(logits, labels.to(device))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        
        predicted_probability = torch.sigmoid(logits).cpu()
        predicted_probability_list.append(predicted_probability)
        label_list.append(labels)
        train_loss += loss.item()
        total_loss += loss.item() / len(dataloader)
        
        if (idx+1) % log_interval == 0 and idx > 0:
            print('| epoch {:3d} | {:5d}/{:5d} batches | loss {:8.3f}'.format(epoch_idx, (idx+1), len(dataloader), train_loss))
            train_loss = 0
    
    y_pred_proba = torch.cat(predicted_probability_list).detach().numpy().copy()
    y_true = torch.cat(label_list).detach().numpy().copy()
    train_acc, train_f1, train_mcc,  train_auc = compute_metrics(y_true, y_pred_proba)
    return total_loss, train_acc, train_f1, train_mcc, train_auc

### Evaluation
def enzymecnn_ablation_evaluate(model, dataloader, criterion, device, return_output=False):
    model.eval()
    total_loss = 0
    predicted_probability_list, label_list = [], []
    with torch.no_grad():
        for idx, (encoded_sequences, labels) in enumerate(dataloader):
            logits = model(encoded_sequences.to(device)).reshape(-1)
            loss = criterion(logits, labels.to(device))
            total_loss += loss.item() / len(dataloader)
            predicted_probability = torch.sigmoid(logits).cpu()
            predicted_probability_list.append(predicted_probability)
            label_list.append(labels)
    
    y_pred_proba = torch.cat(predicted_probability_list).detach().numpy().copy()
    y_true = torch.cat(label_list).detach().numpy().copy()
    
    if not return_output:
        eval_acc, eval_f1, eval_mcc,  eval_auc = compute_metrics(y_true, y_pred_proba)
        return total_loss, eval_acc, eval_f1, eval_mcc, eval_auc
    else:
        return total_loss, eval_acc, eval_f1, eval_mcc, eval_auc, y_pred_proba, y_true

### Inference
def enzymecnn_ablation_inference(model, dataloader, device):
    predicted_probability_list = []
    with torch.no_grad():
        for idx, (encoded_sequences, labels) in enumerate(dataloader):
            logits = model(encoded_sequences.to(device)).reshape(-1)
            predicted_probability = torch.sigmoid(logits).cpu()
            predicted_probability_list.append(predicted_probability)
    
    y_pred_proba = torch.cat(predicted_probability_list).detach().numpy().copy()
    return y_pred_proba


# Load trained EnzymeCNN model
## EnzymeCNN
def load_enzymecnn_model(checkpoint_path, num_inputs, num_channels, kernel_size, dilation, dropout, hidden_dims):
    AA_TCN = TemporalConvNet(num_inputs, num_channels, kernel_size, dilation, dropout)
    STRUCT_TCN = TemporalConvNet(num_inputs, num_channels, kernel_size, dilation, dropout)
    classifier = Classifier(num_channels[-1]*2, hidden_dims, 1, dropout)
    model = MultiTCNClassifier(AA_TCN, STRUCT_TCN, classifier)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    return model

## EnzymeCNN ablations
def load_enzymecnn_ablation_model(checkpoint_path, num_inputs, num_channels, kernel_size, dilation, dropout, hidden_dims):
    TCN = TemporalConvNet(num_inputs, num_channels, kernel_size, dilation, dropout)
    classifier = Classifier(num_channels[-1], hidden_dims, 1, dropout)
    model = TCNClassifier(TCN, classifier)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    return model


if __name__ == '__main__':
    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-inputs', type=int)
    parser.add_argument('--num-channels', type=int, nargs='+')
    parser.add_argument('--kernel-size', type=int)
    parser.add_argument('--dilation', type=int)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--hidden-dims', default=[], type=int, nargs='+')
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--gradient-clip', default=1.0, type=float)
    parser.add_argument('--train-dataset', type=str)
    parser.add_argument('--eval-dataset', type=str)
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'inference'], required=True)
    parser.add_argument('--model', default='EnzymeCNN', type=str,
                        choices=['EnzymeCNN', 'EnzymeCNN-AA', 'EnzymeCNN-3Di'])
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--outputdir', type=str, required=True)
    parser.add_argument('--threads', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--debug', action='store_true')
    
    # Parse inputs
    args = parser.parse_args()
    num_inputs = args.num_inputs
    num_channels = args.num_channels
    kernel_size = args.kernel_size
    dilation = args.dilation
    dropout = args.dropout
    hidden_dims = args.hidden_dims
    epoch = args.epoch
    batch_size = args.batch_size
    lr = args.lr
    gradient_clip = args.gradient_clip
    train_dataset_path = args.train_dataset
    eval_dataset_path = args.eval_dataset
    mode = args.mode
    model_type = args.model
    checkpoint_path = args.checkpoint
    outputdir = args.outputdir
    threads = args.threads
    seed = args.seed
    debug = args.debug
    
    # Set seed for reproducibility
    set_seed(seed, debug)
    
    # Set GPU device if GPU is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Make directory to save results
    os.makedirs(outputdir, exist_ok=True)
    
    # Run EnzymeCNN
    if model_type == 'EnzymeCNN':        
        ## EnzymeCNN training
        if mode == 'train':
            if train_dataset_path is None or eval_dataset_path is None:
                raise ValueError('EnzymeCNN training requires training dataset and validation dataset')
            
            ### Construct EnzymeCNN
            AA_TCN = TemporalConvNet(num_inputs, num_channels, kernel_size, dilation, dropout)
            STRUCT_TCN = TemporalConvNet(num_inputs, num_channels, kernel_size, dilation, dropout)
            classifier = Classifier(num_channels[-1]*2, hidden_dims, 1, dropout)
            model = MultiTCNClassifier(AA_TCN, STRUCT_TCN, classifier).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=lr)
            criterion = nn.BCEWithLogitsLoss()

            ### Make DataLoder
            train_dataloader = make_combined_dataloader(dataset_path=train_dataset_path,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        threads=threads)
            eval_dataloader = make_combined_dataloader(dataset_path=eval_dataset_path,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       threads=threads)
            
            ### Training
            training_log = []
            for epoch_idx in range(epoch):
                start_time = time.time()
                train_result = enzymecnn_training(model, train_dataloader, optimizer, criterion, gradient_clip, epoch_idx, device)
                eval_result = enzymecnn_evaluate(model, eval_dataloader, criterion, device)
                
                print('-' * 100)
                print('end of epoch {:3d}   time: {:5.2f}'.format(epoch_idx, time.time() - start_time))
                print('train_loss {:6.3f}   train accuracy {:6.3f}   train f1 {:6.3f}   train MCC {:6.3f}   train AUC {:6.3f}'.format(*train_result))
                print('eval_loss  {:6.3f}   eval accuracy  {:6.3f}   eval f1  {:6.3f}   eval MCC  {:6.3f}   eval AUC  {:6.3f}'.format(*eval_result))
                print('-' * 100)
                
                training_log.append([epoch_idx] + list(train_result) + list(eval_result))
                checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(checkpoint, f'{outputdir}/checkpoint_epoch{epoch_idx}.pt')

            ### Save training log
            df_log = pd.DataFrame(training_log, columns=['epoch', 'train_loss', 'train_accuracy', 'train_f1', 'train_mcc', 'train_auc',
                                                         'eval_loss', 'eval_accuracy', 'eval_f1', 'eval_mcc', 'eval_auc'])
            df_log.to_csv(f'{outputdir}/training_log.csv', header=True, index=False)
        
        ## EnzymeCNN evaluation
        if mode == 'eval':
            if eval_dataset_path is None:
                raise ValueError('EnzymeCNN evaluation requires validation dataset')
            if checkpoint_path is None:
                raise ValueError('EnzymeCNN evaluation requires trained model checkpoint')
            
            ### Load trained EnzymeCNN model
            model = load_enzymecnn_model(checkpoint_path, num_inputs, num_channels, kernel_size, dilation, dropout, hidden_dims).to(device)
            criterion = nn.BCEWithLogitsLoss()
            
            ### Make DataLoder
            eval_dataloader = make_combined_dataloader(dataset_path=eval_dataset_path,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       threads=threads)
            
            ### Evaluation
            eval_loss, eval_acc, eval_f1, eval_mcc, eval_auc, y_pred_proba, y_true = enzymecnn_evaluate(model, eval_dataloader, criterion, device, return_output=True)
            
            ### Save results
            torch.save(y_pred_proba, f'{outputdir}/pred_scores.pt')
            torch.save(y_true, f'{outputdir}/labels.pt')
            
            df_metrics = pd.DataFrame([[eval_loss, eval_acc, eval_f1, eval_mcc, eval_auc]],
                                      columns=['eval_loss', 'eval_accuracy', 'eval_f1', 'eval_mcc', 'eval_auc'])
            df_metrics.to_csv(f'{outputdir}/evaluation_result.csv', header=True, index=False)
        
        ## EnzymeCNN inference
        if mode == 'inference':
            if eval_dataset_path is None:
                raise ValueError('EnzymeCNN inference requires validation dataset')
            if checkpoint_path is None:
                raise ValueError('EnzymeCNN inference requires trained model checkpoint')
            
            ### Load trained EnzymeCNN model
            model = load_enzymecnn_model(checkpoint_path, num_inputs, num_channels, kernel_size, dilation, dropout, hidden_dims).to(device)
            
            ### Make DataLoder
            eval_dataloader = make_combined_dataloader(dataset_path=eval_dataset_path,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       threads=threads)
            
            ### Inference
            y_pred_proba = enzymecnn_inference(model, eval_dataloader, device)
                        
            ### Save results
            torch.save(y_pred_proba, f'{outputdir}/pred_scores.pt')
            
    
    # Run EnzymeCNN-AA
    elif model_type == 'EnzymeCNN-AA':
        ## EnzymeCNN-AA training
        if mode == 'train':
            if train_dataset_path is None or eval_dataset_path is None:
                raise ValueError('EnzymeCNN training requires training dataset and validation dataset')
            
            ### Construct EnzymeCNN-AA
            TCN = TemporalConvNet(num_inputs, num_channels, kernel_size, dilation, dropout)
            classifier = Classifier(num_channels[-1], hidden_dims, 1, dropout)
            model = TCNClassifier(TCN, classifier).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=lr)
            criterion = nn.BCEWithLogitsLoss()

            ### Make DataLoder
            train_dataloader = make_aa_dataloader(dataset_path=train_dataset_path,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  threads=threads)
            eval_dataloader = make_aa_dataloader(dataset_path=eval_dataset_path,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 threads=threads)
            
            ### Training
            training_log = []
            for epoch_idx in range(epoch):
                start_time = time.time()
                train_result = enzymecnn_ablation_training(model, train_dataloader, optimizer, criterion, gradient_clip, epoch_idx, device)
                eval_result = enzymecnn_ablation_evaluate(model, eval_dataloader, criterion, device)
                
                print('-' * 100)
                print('end of epoch {:3d}   time: {:5.2f}'.format(epoch_idx, time.time() - start_time))
                print('train_loss {:6.3f}   train accuracy {:6.3f}   train f1 {:6.3f}   train MCC {:6.3f}   train AUC {:6.3f}'.format(*train_result))
                print('eval_loss  {:6.3f}   eval accuracy  {:6.3f}   eval f1  {:6.3f}   eval MCC  {:6.3f}   eval AUC  {:6.3f}'.format(*eval_result))
                print('-' * 100)
                
                training_log.append([epoch_idx] + list(train_result) + list(eval_result))
                checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(checkpoint, f'{outputdir}/checkpoint_epoch{epoch_idx}.pt')

            ### Save training log
            df_log = pd.DataFrame(training_log, columns=['epoch', 'train_loss', 'train_accuracy', 'train_f1', 'train_mcc', 'train_auc',
                                                         'eval_loss', 'eval_accuracy', 'eval_f1', 'eval_mcc', 'eval_auc'])
            df_log.to_csv(f'{outputdir}/training_log.csv', header=True, index=False)
            
        ## EnzymeCNN-AA evaluation
        if mode == 'eval':
            if eval_dataset_path is None:
                raise ValueError('EnzymeCNN evaluation requires validation dataset')
            if checkpoint_path is None:
                raise ValueError('EnzymeCNN evaluation requires trained model checkpoint')
            
            ### Load trained EnzymeCNN model
            model = load_enzymecnn_ablation_model(checkpoint_path, num_inputs, num_channels, kernel_size, dilation, dropout, hidden_dims).to(device)
            criterion = nn.BCEWithLogitsLoss()
            
            ### Make DataLoder
            eval_dataloader = make_aa_dataloader(dataset_path=eval_dataset_path,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 threads=threads)
            
            ### Evaluation
            eval_loss, eval_acc, eval_f1, eval_mcc, eval_auc, y_pred_proba, y_true = enzymecnn_ablation_inference(model, eval_dataloader, criterion, device, return_output=True)
            
            ### Save results
            torch.save(y_pred_proba, f'{outputdir}/pred_scores.pt')
            torch.save(y_true, f'{outputdir}/labels.pt')
            
            df_metrics = pd.DataFrame([[eval_loss, eval_acc, eval_f1, eval_mcc, eval_auc]],
                                      columns=['eval_loss', 'eval_accuracy', 'eval_f1', 'eval_mcc', 'eval_auc'])
            df_metrics.to_csv(f'{outputdir}/evaluation_result.csv', header=True, index=False)
        
        ## EnzymeCNN-AA inference
        if mode == 'inference':
            if eval_dataset_path is None:
                raise ValueError('EnzymeCNN inference requires validation dataset')
            if checkpoint_path is None:
                raise ValueError('EnzymeCNN inference requires trained model checkpoint')
            
            ### Load trained EnzymeCNN model
            model = load_enzymecnn_ablation_model(checkpoint_path, num_inputs, num_channels, kernel_size, dilation, dropout, hidden_dims).to(device)
            criterion = nn.BCEWithLogitsLoss()
            
            ### Make DataLoder
            eval_dataloader = make_aa_dataloader(dataset_path=eval_dataset_path,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 threads=threads)
            
            ### Inference
            y_pred_proba = enzymecnn_ablation_inference(model, eval_dataloader, device)
            
            ### Save results
            torch.save(y_pred_proba, f'{outputdir}/pred_scores.pt')
    
    # Run EnzymeCNN-3Di
    elif model_type == 'EnzymeCNN-3Di':
        ## EnzymeCNN-3Di training
        if mode == 'train':
            if train_dataset_path is None or eval_dataset_path is None:
                raise ValueError('EnzymeCNN training requires training dataset and validation dataset')
            
            ### Construct EnzymeCNN-3Di
            TCN = TemporalConvNet(num_inputs, num_channels, kernel_size, dilation, dropout)
            classifier = Classifier(num_channels[-1], hidden_dims, 1, dropout)
            model = TCNClassifier(TCN, classifier).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=lr)
            criterion = nn.BCEWithLogitsLoss()
            
            ### Make DataLoder
            train_dataloader = make_struct_dataloader(dataset_path=train_dataset_path,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      threads=threads)
            eval_dataloader = make_struct_dataloader(dataset_path=eval_dataset_path,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     threads=threads)
            
            ### Training
            training_log = []
            for epoch_idx in range(epoch):
                start_time = time.time()
                train_result = enzymecnn_ablation_training(model, train_dataloader, optimizer, criterion, gradient_clip, epoch_idx, device)
                eval_result = enzymecnn_ablation_evaluate(model, eval_dataloader, criterion, device)
                
                print('-' * 100)
                print('end of epoch {:3d}   time: {:5.2f}'.format(epoch_idx, time.time() - start_time))
                print('train_loss {:6.3f}   train accuracy {:6.3f}   train f1 {:6.3f}   train MCC {:6.3f}   train AUC {:6.3f}'.format(*train_result))
                print('eval_loss  {:6.3f}   eval accuracy  {:6.3f}   eval f1  {:6.3f}   eval MCC  {:6.3f}   eval AUC  {:6.3f}'.format(*eval_result))
                print('-' * 100)
                
                training_log.append([epoch_idx] + list(train_result) + list(eval_result))
                checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(checkpoint, f'{outputdir}/checkpoint_epoch{epoch_idx}.pt')

            ### Save training log
            df_log = pd.DataFrame(training_log, columns=['epoch', 'train_loss', 'train_accuracy', 'train_f1', 'train_mcc', 'train_auc',
                                                         'eval_loss', 'eval_accuracy', 'eval_f1', 'eval_mcc', 'eval_auc'])
            df_log.to_csv(f'{outputdir}/training_log.csv', header=True, index=False)
        
        ## EnzymeCNN-3Di evaluation
        if mode == 'eval':
            if eval_dataset_path is None:
                raise ValueError('EnzymeCNN evaluation requires validation dataset')
            if checkpoint_path is None:
                raise ValueError('EnzymeCNN evaluation requires trained model checkpoint')
            
            ### Load trained EnzymeCNN model
            model = load_enzymecnn_ablation_model(checkpoint_path, num_inputs, num_channels, kernel_size, dilation, dropout, hidden_dims).to(device)
            criterion = nn.BCEWithLogitsLoss()
            
            ### Make DataLoder
            eval_dataloader = make_struct_dataloader(dataset_path=eval_dataset_path,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     threads=threads)
            
            ### Evaluation
            eval_loss, eval_acc, eval_f1, eval_mcc, eval_auc, y_pred_proba, y_true = enzymecnn_ablation_inference(model, eval_dataloader, criterion, device, return_output=True)
            
            ### Save results
            torch.save(y_pred_proba, f'{outputdir}/pred_scores.pt')
            torch.save(y_true, f'{outputdir}/labels.pt')
            
            df_metrics = pd.DataFrame([[eval_loss, eval_acc, eval_f1, eval_mcc, eval_auc]],
                                      columns=['eval_loss', 'eval_accuracy', 'eval_f1', 'eval_mcc', 'eval_auc'])
            df_metrics.to_csv(f'{outputdir}/evaluation_result.csv', header=True, index=False)
        
        ## EnzymeCNN-3Di inference
        if mode == 'inference':
            if eval_dataset_path is None:
                raise ValueError('EnzymeCNN inference requires validation dataset')
            if checkpoint_path is None:
                raise ValueError('EnzymeCNN inference requires trained model checkpoint')
            
            ### Load trained EnzymeCNN model
            model = load_enzymecnn_ablation_model(checkpoint_path, num_inputs, num_channels, kernel_size, dilation, dropout, hidden_dims).to(device)
            criterion = nn.BCEWithLogitsLoss()
            
            ### Make DataLoder
            eval_dataloader = make_struct_dataloader(dataset_path=eval_dataset_path,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     threads=threads)
            
            ### Inference
            y_pred_proba = enzymecnn_ablation_inference(model, eval_dataloader, device)
            
            ### Save results
            torch.save(y_pred_proba, f'{outputdir}/pred_scores.pt')

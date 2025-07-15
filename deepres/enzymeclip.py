import argparse
import os
import re
import random
import math
from functools import partial
from datetime import datetime
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers
from transformers import BertModel, BertTokenizer, EsmTokenizer, EsmModel
from peft import get_peft_model, LoraConfig
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


# Pretrained models
## Load SaProt model
def load_saprot_model(checkpoint, lora_config=None):
    """
    The absence of some weights is normal when you initialize SaProt.
    source: https://github.com/westlake-repl/SaProt/issues/12
    """
    tokenizer = EsmTokenizer.from_pretrained(checkpoint)
    model = EsmModel.from_pretrained(checkpoint, add_pooling_layer=False)
    
    if lora_config is None:
        return model, tokenizer
    
    peft_config = LoraConfig(**lora_config)
    model = get_peft_model(model, peft_config)
    
    return model, tokenizer


## Load ESM-2 model
def load_esm_model(checkpoint, lora_config=None):
    tokenizer = EsmTokenizer.from_pretrained(checkpoint)
    model = EsmModel.from_pretrained(checkpoint, add_pooling_layer=False)
    
    if lora_config is None:
        return model, tokenizer
    
    peft_config = LoraConfig(**lora_config)
    model = get_peft_model(model, peft_config)
    
    return model, tokenizer


## RXNFP (Reference: https://github.com/rxn4chemistry/rxnfp)
### Construct reaction tokenizer
SMI_REGEX_PATTERN =  r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"

class SmilesTokenizer(BertTokenizer):
    """
    Constructs a SmilesBertTokenizer.
    Adapted from https://github.com/huggingface/transformers
    and https://github.com/rxn4chemistry/rxnfp.

    Args:
        vocabulary_file: path to a token per line vocabulary file.
    """

    def __init__(
        self,
        vocab_file: str,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        do_lower_case=False,
        **kwargs,
    ) -> None:
        """Constructs an SmilesTokenizer.
        Args:
            vocabulary_file: vocabulary file containing tokens.
            unk_token: unknown token. Defaults to "[UNK]".
            sep_token: separator token. Defaults to "[SEP]".
            pad_token: pad token. Defaults to "[PAD]".
            cls_token: cls token. Defaults to "[CLS]".
            mask_token: mask token. Defaults to "[MASK]".
        """
        super().__init__(
            vocab_file=vocab_file,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            do_lower_case=do_lower_case,
            **kwargs,
        )
        # define tokenization utilities
        self.tokenizer = RegexTokenizer()

    @property
    def vocab_list(self) -> List[str]:
        """List vocabulary tokens.
        Returns:
            a list of vocabulary tokens.
        """
        return list(self.vocab.keys())

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a text representing an enzymatic reaction with AA sequence information.
        Args:
            text: text to tokenize.
        Returns:
            extracted tokens.
        """
        return self.tokenizer.tokenize(text)


class RegexTokenizer:
    """Run regex tokenization"""

    def __init__(self, regex_pattern: str=SMI_REGEX_PATTERN) -> None:
        """Constructs a RegexTokenizer.
        Args:
            regex_pattern: regex pattern used for tokenization.
            suffix: optional suffix for the tokens. Defaults to "".
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)

    def tokenize(self, text: str) -> List[str]:
        """Regex tokenization.
        Args:
            text: text to tokenize.
        Returns:
            extracted tokens separated by spaces.
        """
        tokens = [token for token in self.regex.findall(text)]
        return tokens


### Load RXNFP model
def load_rxnfp_model(checkpoint, lora_config=None):
    tokenizer = SmilesTokenizer.from_pretrained(checkpoint)
    model = BertModel.from_pretrained(checkpoint)
    
    if lora_config is None:
        return model, tokenizer
    
    peft_config = LoraConfig(**lora_config)
    model = get_peft_model(model, peft_config)
    
    return model, tokenizer


# EnzymeCLIP models
## ProjectionHead to align embeddings by pretrained model into a shared space
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(output_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.layer_norm(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


## EnzymeCLIP
class EnzymeCLIP(pl.LightningModule):
    def __init__(self,
                 embedding_dim, lr, batch_size,
                 reaction_model, reaction_tokenizer, reaction_dim, reaction_loss,
                 sequence_model, sequence_tokenizer, sequence_dim, sequence_loss,
                 init_logit_scale=np.log(1 / 0.07), init_logit_bias=None):
        
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.batch_size = batch_size
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias) if init_logit_bias is not None else None
        
        self.reaction_model = reaction_model
        self.reaction_tokenizer = reaction_tokenizer
        self.reaction_dim = reaction_dim
        self.reaction_prjector = ProjectionHead(self.reaction_dim, self.embedding_dim)
        self.reaction_loss = reaction_loss
        
        self.sequence_model = sequence_model
        self.sequence_tokenizer = sequence_tokenizer
        self.sequence_dim = sequence_dim
        self.sequence_prjector = ProjectionHead(self.sequence_dim, self.embedding_dim)
        self.sequence_loss = sequence_loss

    def encode_reaction(self, inputs, normalize=True):
        output = self.reaction_model(**inputs)
        reaction_vec = output['last_hidden_state'][:, 0, :]
        reaction_features = self.reaction_prjector(reaction_vec)
        return F.normalize(reaction_features, dim=-1) if normalize else reaction_features

    def encode_sequence(self, inputs, normalize=True):
        outputs = self.sequence_model(**inputs)
        
        input_ids = inputs["input_ids"]
        eos_id = self.sequence_tokenizer.eos_token_id
        ends = (input_ids == eos_id).int()
        indices = ends.argmax(dim=-1)

        repr_list = []
        hidden_states = outputs["hidden_states"][-1]
        for i, idx in enumerate(indices):
            repr = hidden_states[i][1:idx].mean(dim=0)
            repr_list.append(repr)
        
        sequence_vec = torch.stack(repr_list)
        sequence_features = self.sequence_prjector(sequence_vec)
        return F.normalize(sequence_features, dim=-1) if normalize else sequence_features
    
    def forward(self, reactions, sequences, normalize=True):
        # Encode reaction vector representation and sequence vector representation
        reaction_features = self.encode_reaction(reactions, normalize=normalize)
        sequence_features = self.encode_sequence(sequences, normalize=normalize)
        return reaction_features, sequence_features
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return {'optimizer': optimizer}
    
    def training_step(self, batch, batch_idx):
        reactions, sequences = batch
        reaction_features, sequence_features = self(reactions, sequences)
        loss = self.contrastive_loss(reaction_features, sequence_features)
        self.log('train_loss', loss,
                 batch_size=self.batch_size,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=False,
                 sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        reactions, sequences = batch
        reaction_features, sequence_features = self(reactions, sequences)
        loss = self.contrastive_loss(reaction_features, sequence_features)
        self.log('val_loss', loss,
                 batch_size=self.batch_size,
                 on_epoch=True,
                 prog_bar=False,
                 sync_dist=True)
    
    def contrastive_loss(self, reaction_features, sequence_features):
        # Global batch size aggregation using PyTorch Lightning's all_gather
        gathered_reaction_features = self.all_gather(reaction_features, sync_grads=True).reshape(-1, self.embedding_dim)
        gathered_sequence_features = self.all_gather(sequence_features, sync_grads=True).reshape(-1, self.embedding_dim)
        
        # Compute cosine similarity (logits) for contrastive learning
        logit_scale = self.logit_scale.exp()
        logits_per_reaction = logit_scale * gathered_reaction_features @ gathered_sequence_features.T
        if self.logit_bias is not None:
            logits_per_reaction += self.logit_bias
        logits_per_sequence = logits_per_reaction.T
        
        # Define labels
        labels = torch.arange(gathered_sequence_features.shape[0], device=self.device)
        
        # CrossEntropyLoss
        loss_rxn = self.reaction_loss(logits_per_reaction, labels)
        loss_seq = self.sequence_loss(logits_per_sequence, labels)
        
        return (loss_rxn + loss_seq) / 2
    
    def on_train_epoch_end(self):
        if self.trainer.is_global_zero:
            current_time = datetime.now().strftime('%Y-%m-%d %a %H:%M:%S')
            print(f'{current_time} JST | In progress...')
    
    def on_train_end(self):
        if self.trainer.is_global_zero:
            current_time = datetime.now().strftime('%Y-%m-%d %a %H:%M:%S')
            print(f'{current_time} JST | Training comleted!!!')


## EnzymeCyCLIP
class EnzymeCyCLIP(EnzymeCLIP):
    def __init__(self,
                 embedding_dim, lr, batch_size,
                 reaction_model, reaction_tokenizer, reaction_dim, reaction_loss,
                 sequence_model, sequence_tokenizer, sequence_dim, sequence_loss,
                 init_logit_scale=np.log(1 / 0.07), init_logit_bias=None,
                 cylambda1=0.25, cylambda2=0.25):
        
        super().__init__(embedding_dim, lr, batch_size,
                         reaction_model, reaction_tokenizer, reaction_dim, reaction_loss,
                         sequence_model, sequence_tokenizer, sequence_dim, sequence_loss,
                         init_logit_scale, init_logit_bias)
        
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.batch_size = batch_size
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias) if init_logit_bias is not None else None
        self.cylambda1 = cylambda1
        self.cylambda2 = cylambda2
        
        self.reaction_model = reaction_model
        self.reaction_tokenizer = reaction_tokenizer
        self.reaction_dim = reaction_dim
        self.reaction_prjector = ProjectionHead(self.reaction_dim, self.embedding_dim)
        self.reaction_loss = reaction_loss
        
        self.sequence_model = sequence_model
        self.sequence_tokenizer = sequence_tokenizer
        self.sequence_dim = sequence_dim
        self.sequence_prjector = ProjectionHead(self.sequence_dim, self.embedding_dim)
        self.sequence_loss = sequence_loss
        
    def training_step(self, batch, batch_idx):
        reactions, sequences = batch
        reaction_features, sequence_features = self(reactions, sequences)
        loss, contrastive_loss, cyclic_loss = self.contrastive_loss(reaction_features, sequence_features)
        self.log_dict({'train_loss': loss,
                       'train_contrastive_loss': contrastive_loss,
                       'train_cyclic_loss': cyclic_loss},
                      batch_size=self.batch_size,
                      on_step=True,
                      on_epoch=True,
                      prog_bar=False,
                      sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        reactions, sequences = batch
        reaction_features, sequence_features = self(reactions, sequences)
        loss, contrastive_loss, cyclic_loss = self.contrastive_loss(reaction_features, sequence_features)
        self.log_dict({'val_loss': loss,
                       'val_contrastive_loss': contrastive_loss,
                       'val_cyclic_loss': cyclic_loss},
                      batch_size=self.batch_size,
                      on_epoch=True,
                      prog_bar=False,
                      sync_dist=True)
    
    def contrastive_loss(self, reaction_features, sequence_features):
        # Global batch size aggregation using PyTorch Lightning's all_gather
        gathered_reaction_features = self.all_gather(reaction_features, sync_grads=True).reshape(-1, self.embedding_dim)
        gathered_sequence_features = self.all_gather(sequence_features, sync_grads=True).reshape(-1, self.embedding_dim)
        
        # Compute cosine similarity (logits) for contrastive learning
        logit_scale = self.logit_scale.exp()
        logits_seq_per_rxn = logit_scale * gathered_reaction_features @ gathered_sequence_features.T
        if self.logit_bias is not None:
            logits_seq_per_rxn += self.logit_bias
        logits_rxn_per_seq = logits_seq_per_rxn.T
        
        # Define labels
        labels = torch.arange(gathered_sequence_features.shape[0], device=self.device)
        
        # Cross-modal contrastive loss
        loss_rxn = self.reaction_loss(logits_seq_per_rxn, labels)
        loss_seq = self.sequence_loss(logits_rxn_per_seq, labels)
        contrastive_loss = (loss_rxn + loss_seq) / 2
        
        # In-model cyclic loss
        logits_rxn_per_rxn = logit_scale * gathered_reaction_features @ gathered_reaction_features.T
        logits_seq_per_seq = logit_scale * gathered_sequence_features @ gathered_sequence_features.T
        inmodal_cyclic_loss = (logits_rxn_per_rxn - logits_seq_per_seq).square().mean() / (logit_scale * logit_scale) * self.batch_size
        
        # Cross-modal cyclic loss
        crossmodal_cyclic_loss = (logits_seq_per_rxn - logits_rxn_per_seq).square().mean() / (logit_scale * logit_scale) * self.batch_size
        
        cyclic_loss = self.cylambda1 * inmodal_cyclic_loss + self.cylambda2 * crossmodal_cyclic_loss
        loss = contrastive_loss + cyclic_loss
        
        return loss, contrastive_loss, cyclic_loss


class EnzymeSoftCLIP(EnzymeCLIP):
    def __init__(self,
                 embedding_dim, lr, batch_size,
                 reaction_model, reaction_tokenizer, reaction_dim, reaction_loss,
                 sequence_model, sequence_tokenizer, sequence_dim, sequence_loss,
                 init_logit_scale=np.log(1 / 0.07), init_logit_bias=None,
                 softcoef=0.3):
        
        super().__init__(embedding_dim, lr, batch_size,
                         reaction_model, reaction_tokenizer, reaction_dim, reaction_loss,
                         sequence_model, sequence_tokenizer, sequence_dim, sequence_loss,
                         init_logit_scale, init_logit_bias)
        
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.batch_size = batch_size
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias) if init_logit_bias is not None else None
        self.softcoef = softcoef
        
        self.reaction_model = reaction_model
        self.reaction_tokenizer = reaction_tokenizer
        self.reaction_dim = reaction_dim
        self.reaction_prjector = ProjectionHead(self.reaction_dim, self.embedding_dim)
        self.reaction_loss = reaction_loss
        
        self.sequence_model = sequence_model
        self.sequence_tokenizer = sequence_tokenizer
        self.sequence_dim = sequence_dim
        self.sequence_prjector = ProjectionHead(self.sequence_dim, self.embedding_dim)
        self.sequence_loss = sequence_loss
    
    def training_step(self, batch, batch_idx):
        reactions, sequences = batch
        reaction_features, sequence_features = self(reactions, sequences)
        loss = self.contrastive_loss(reaction_features, sequence_features)
        self.log_dict({'train_loss': loss},
                      batch_size=self.batch_size,
                      on_step=True,
                      on_epoch=True,
                      prog_bar=False,
                      sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        reactions, sequences = batch
        reaction_features, sequence_features = self(reactions, sequences)
        loss = self.contrastive_loss(reaction_features, sequence_features)
        self.log_dict({'val_loss': loss},
                      batch_size=self.batch_size,
                      on_epoch=True,
                      prog_bar=False,
                      sync_dist=True)
    
    def contrastive_loss(self, reaction_features, sequence_features):
        # Global batch size aggregation using PyTorch Lightning's all_gather
        gathered_reaction_features = self.all_gather(reaction_features, sync_grads=True).reshape(-1, self.embedding_dim)
        gathered_sequence_features = self.all_gather(sequence_features, sync_grads=True).reshape(-1, self.embedding_dim)
        
        # Compute cosine similarity (logits) for contrastive learning
        logit_scale = self.logit_scale.exp()
        logits_seq_per_rxn = logit_scale * gathered_reaction_features @ gathered_sequence_features.T
        if self.logit_bias is not None:
            logits_seq_per_rxn += self.logit_bias
        logits_rxn_per_seq = logits_seq_per_rxn.T
        
        # Define labels
        similarity_matrix_seq = gathered_sequence_features @ gathered_sequence_features.T
        similarity_matrix_seq_no_grad = similarity_matrix_seq.detach()
        ground_truth = torch.eye(gathered_sequence_features.shape[0], device=self.device)
        labels = (1 - self.softcoef) * ground_truth + self.softcoef * similarity_matrix_seq_no_grad
        
        # Cross-modal contrastive loss
        loss_rxn = self.reaction_loss(logits_seq_per_rxn, labels)
        loss_seq = self.sequence_loss(logits_rxn_per_seq, labels)
        loss = (loss_rxn + loss_seq) / 2
        
        return loss


class EnzymeSoftCyCLIP(EnzymeCLIP):
    def __init__(self,
                 embedding_dim, lr, batch_size,
                 reaction_model, reaction_tokenizer, reaction_dim, reaction_loss,
                 sequence_model, sequence_tokenizer, sequence_dim, sequence_loss,
                 init_logit_scale=np.log(1 / 0.07), init_logit_bias=None,
                 cylambda1=0.25, cylambda2=0.25, softcoef=0.3):
        
        super().__init__(embedding_dim, lr, batch_size,
                         reaction_model, reaction_tokenizer, reaction_dim, reaction_loss,
                         sequence_model, sequence_tokenizer, sequence_dim, sequence_loss,
                         init_logit_scale, init_logit_bias)
        
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.batch_size = batch_size
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias) if init_logit_bias is not None else None
        self.cylambda1 = cylambda1
        self.cylambda2 = cylambda2
        self.softcoef = softcoef
        
        self.reaction_model = reaction_model
        self.reaction_tokenizer = reaction_tokenizer
        self.reaction_dim = reaction_dim
        self.reaction_prjector = ProjectionHead(self.reaction_dim, self.embedding_dim)
        self.reaction_loss = reaction_loss
        
        self.sequence_model = sequence_model
        self.sequence_tokenizer = sequence_tokenizer
        self.sequence_dim = sequence_dim
        self.sequence_prjector = ProjectionHead(self.sequence_dim, self.embedding_dim)
        self.sequence_loss = sequence_loss

    def training_step(self, batch, batch_idx):
        reactions, sequences = batch
        reaction_features, sequence_features = self(reactions, sequences)
        loss, soft_contrastive_loss, cyclic_loss = self.contrastive_loss(reaction_features, sequence_features)
        self.log_dict({'train_loss': loss,
                       'train_soft_contrastive_loss': soft_contrastive_loss,
                       'train_cyclic_loss': cyclic_loss},
                      batch_size=self.batch_size,
                      on_step=True,
                      on_epoch=True,
                      prog_bar=False,
                      sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        reactions, sequences = batch
        reaction_features, sequence_features = self(reactions, sequences)
        loss, soft_contrastive_loss, cyclic_loss = self.contrastive_loss(reaction_features, sequence_features)
        self.log_dict({'val_loss': loss,
                       'val_soft_contrastive_loss': soft_contrastive_loss,
                       'val_cyclic_loss': cyclic_loss},
                      batch_size=self.batch_size,
                      on_epoch=True,
                      prog_bar=False,
                      sync_dist=True)
    
    def contrastive_loss(self, reaction_features, sequence_features):
        # Global batch size aggregation using PyTorch Lightning's all_gather
        gathered_reaction_features = self.all_gather(reaction_features, sync_grads=True).reshape(-1, self.embedding_dim)
        gathered_sequence_features = self.all_gather(sequence_features, sync_grads=True).reshape(-1, self.embedding_dim)
        
        # Compute cosine similarity (logits) for contrastive learning
        logit_scale = self.logit_scale.exp()
        logits_seq_per_rxn = logit_scale * gathered_reaction_features @ gathered_sequence_features.T
        if self.logit_bias is not None:
            logits_seq_per_rxn += self.logit_bias
        logits_rxn_per_seq = logits_seq_per_rxn.T
        
        # Define labels
        similarity_matrix_seq = gathered_sequence_features @ gathered_sequence_features.T
        similarity_matrix_seq_no_grad = similarity_matrix_seq.detach()
        ground_truth = torch.eye(gathered_sequence_features.shape[0], device=self.device)
        labels = (1 - self.softcoef) * ground_truth + self.softcoef * similarity_matrix_seq_no_grad
        
        # Cross-modal soft contrastive loss
        loss_rxn = self.reaction_loss(logits_seq_per_rxn, labels)
        loss_seq = self.sequence_loss(logits_rxn_per_seq, labels)
        soft_contrastive_loss = (loss_rxn + loss_seq) / 2
        
        # In-model cyclic loss
        logits_rxn_per_rxn = logit_scale * gathered_reaction_features @ gathered_reaction_features.T
        logits_seq_per_seq = logit_scale * gathered_sequence_features @ gathered_sequence_features.T
        inmodal_cyclic_loss = (logits_rxn_per_rxn - logits_seq_per_seq).square().mean() / (logit_scale * logit_scale) * self.batch_size
        
        # Cross-modal cyclic loss
        crossmodal_cyclic_loss = (logits_seq_per_rxn - logits_rxn_per_seq).square().mean() / (logit_scale * logit_scale) * self.batch_size
        
        cyclic_loss = self.cylambda1 * inmodal_cyclic_loss + self.cylambda2 * crossmodal_cyclic_loss
        loss = soft_contrastive_loss + cyclic_loss
        
        return loss, soft_contrastive_loss, cyclic_loss


# Prepare dataloader
## Dataset
class AbstructDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class CLIPDataset(Dataset):
    def __init__(self, reactions, sequences):
        self.rxn = reactions
        self.seq = sequences
    
    def __len__(self):
        return len(self.rxn)
    
    def __getitem__(self, idx):
        return self.rxn[idx], self.seq[idx]

## Preprocessing
def rxn_preprocessing(reactions, tokenizer, device):
    encoded_reactions = tokenizer.batch_encode_plus(reactions,
                                                    max_length=512,
                                                    padding=True,
                                                    truncation=True,
                                                    return_tensors='pt')
    return encoded_reactions.to(device)

def seq_preprocessing(sequences, tokenizer, device):
    encoded_sequences = tokenizer.batch_encode_plus(sequences,
                                                    max_length=2048,
                                                    padding=True,
                                                    truncation=True,
                                                    return_tensors='pt')
    encoded_sequences = {k: v.to(device) for k, v in encoded_sequences.items()}
    encoded_sequences["output_hidden_states"] = True
    return encoded_sequences

def clip_preprocessing(batch, rxn_tokenizer, seq_tokenizer):
    reactions, sequences= list(zip(*batch))
    # reaction
    encoded_reactions = rxn_tokenizer.batch_encode_plus(reactions,
                                                        max_length=512,
                                                        padding=True,
                                                        truncation=True,
                                                        return_tensors='pt')
    # sequence
    encoded_sequences = seq_tokenizer.batch_encode_plus(sequences,
                                                        max_length=2048,
                                                        padding=True,
                                                        truncation=True,
                                                        return_tensors='pt')
    encoded_sequences = {k: v for k, v in encoded_sequences.items()}
    encoded_sequences['output_hidden_states'] = True
    return encoded_reactions, encoded_sequences

## Dataloader creation
def make_clip_dataloader(dataset_path, batch_size, shuffle, threads, reaction_tokenizer, protein_tokenizer):
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(0)
    
    df = pd.read_table(dataset_path)
    dataset = CLIPDataset(df['rxn_smiles'].to_list(), df['combined_sequence'].to_list())
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=threads,
                            collate_fn=partial(clip_preprocessing,
                                               rxn_tokenizer=reaction_tokenizer,
                                               seq_tokenizer=protein_tokenizer),
                            worker_init_fn=seed_worker,
                            generator=g)
    return dataloader

def make_protein_dataloader(protein_dataset_path, batch_size, protein_tokenizer, device):
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(0)
    
    protein_set = pd.read_table(protein_dataset_path)
    seq_dataset = AbstructDataset(protein_set['combined_sequence'].to_list())
    seq_dataloader = DataLoader(seq_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=partial(seq_preprocessing,
                                                   tokenizer=protein_tokenizer,
                                                   device=device),
                                worker_init_fn=seed_worker,
                                generator=g)
    return seq_dataloader

def make_reaction_dataloader(reaction_dataset_path, batch_size, reaction_tokenizer, device):
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(0)
    
    reaction_set = pd.read_table(reaction_dataset_path)
    rxn_dataset = AbstructDataset(reaction_set['rxn_smiles'].to_list())
    rxn_dataloader = DataLoader(rxn_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=partial(rxn_preprocessing,
                                                   tokenizer=reaction_tokenizer,
                                                   device=device),
                                worker_init_fn=seed_worker,
                                generator=g)
    return rxn_dataloader

# EnzymeCLIP inference
## Embed proteins using trained EnzymeCLIP model
def embed_proteins(model, seq_dataloader):
    model.eval()
    protein_features_all = []
    with torch.no_grad():
        for i, proteins in enumerate(seq_dataloader):
            protein_features = model.encode_sequence(proteins, normalize=True)
            protein_features_all.append(protein_features)
    return torch.cat(protein_features_all)

## Embed reactions using trained EnzymeCLIP model
def embed_reactions(model, rxn_dataloader):
    model.eval()
    reaction_features_all = []
    with torch.no_grad():
        for i, reactions in enumerate(rxn_dataloader):
            reaction_features = model.encode_reaction(reactions, normalize=True)
            reaction_features_all.append(reaction_features)
    return torch.cat(reaction_features_all)

# Evaluataion
## Compute enrichment factor
def enrichment_factor(y_true: np.ndarray, y_pred: np.ndarray, chi: float = 0.1):
    assert len(y_true) == len(y_pred), 'Number of inputs do not match'
    n = float(sum(y_true))
    N = float(len(y_true))
    order = np.argsort(-y_pred)
    k = math.floor(chi * N)
    positive_in_topk = (y_true[order] == 1)[:k].sum()
    return float(positive_in_topk) / (chi * n)

## Evaluate EnzymeCLIP performance
def evaluate(ground_truth: np.ndarray, cos_sim_matrix: np.ndarray, chi: float = 0.1):
    assert ground_truth.shape == cos_sim_matrix.shape, 'Shape of inputs do not match'
    ef_scores = []
    for i in range(len(ground_truth)):
        ef_score = enrichment_factor(ground_truth[i], cos_sim_matrix[i], chi)
        ef_scores.append(ef_score)
    return np.mean(ef_scores)


if __name__ == '__main__':
    # Input parameters
    parser = argparse.ArgumentParser(description='EnzymeCLIP')
    # --- Protein encoder ---
    parser.add_argument('--protein-checkpoint', default='westlake-repl/SaProt_650M_AF2', type=str)
    parser.add_argument('--protein-dim', default=1280, type=int)
    parser.add_argument('--protein-use_lora', action='store_true')
    parser.add_argument('--protein-lora_r', default=4, type=int)
    parser.add_argument('--protein-lora_alpha', default=1, type=int)
    parser.add_argument('--protein-lora_bias', default='all', type=str)
    parser.add_argument('--protein-lora_target_modules', nargs='+',
                        default=['query','key','value','dense'], type=list[str])
    # --- Reaction encoder ---
    parser.add_argument('--reaction-checkpoint', default='pretrained/rxnfp', type=str)
    parser.add_argument('--reaction-dim', default=256, type=int)
    parser.add_argument('--reaction-use_lora', action='store_true')
    parser.add_argument('--reaction-lora_r', default=4, type=int)
    parser.add_argument('--reaction-lora_alpha', default=1, type=int)
    parser.add_argument('--reaction-lora_bias', default='all', type=str)
    parser.add_argument('--reaction-lora_target_modules', nargs='+',
                        default=['query','key','value','dense'], type=list[str])
    # --- EnzymeCLIP ---
    parser.add_argument('--embedding-dim', default=256, type=int)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--gradient-clip', default=1.0, type=float)
    parser.add_argument('--init-logit-scale', default=0.07, type=float)
    parser.add_argument('--init-logit-bias', default=None, type=float)
    parser.add_argument('--cylambda1', default=0.25, type=float)
    parser.add_argument('--cylambda2', default=0.25, type=float)
    parser.add_argument('--softcoef', default=0.3, type=float)
    parser.add_argument('--train-dataset', default=None, type=str)
    parser.add_argument('--eval-dataset', default=None, type=str)
    parser.add_argument('--protein-dataset', default=None, type=str)
    parser.add_argument('--reaction-dataset', default=None, type=str)
    parser.add_argument('--ground-truth', default=None, type=str)
    parser.add_argument('--chi', default=[0.05, 0.1], type=float, nargs='+')
    parser.add_argument('--outputdir', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'inference'], required=True)
    parser.add_argument('--model', default='EnzymeCyCLIP', type=str,
                        choices=['EnzymeCLIP', 'EnzymeCyCLIP', 'EnzymeSoftCLIP', 'EnzymeSoftCyCLIP'])
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--threads', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--debug', action='store_true')
    
    # Parse inputs
    args = parser.parse_args()
    protein_checkpoint_path = args.protein_checkpoint
    protein_dim = args.protein_dim
    protein_lora_config = {'r': args.protein_lora_r,
                           'lora_alpha': args.protein_lora_alpha,
                           'bias': args.protein_lora_bias,
                           'target_modules': args.protein_lora_target_modules}
    if not args.protein_use_lora:
        protein_lora_config = None
    
    reaction_checkpoint_path = args.reaction_checkpoint
    reaction_dim = args.reaction_dim
    reaction_lora_config = {'r': args.reaction_lora_r,
                           'lora_alpha': args.reaction_lora_alpha,
                           'bias': args.reaction_lora_bias,
                           'target_modules': args.reaction_lora_target_modules}
    if not args.reaction_use_lora:
        reaction_lora_config = None
    
    embedding_dim = args.embedding_dim
    epoch = args.epoch
    batch_size = args.batch_size
    lr = args.lr
    gradient_clip = args.gradient_clip
    init_logit_scale = np.log(1 / args.init_logit_scale)
    init_logit_bias = args.init_logit_bias
    cylambda1 = args.cylambda1
    cylambda2 = args.cylambda2
    softcoef = args.softcoef
    train_dataset_path = args.train_dataset
    eval_dataset_path = args.eval_dataset
    protein_dataset_path = args.protein_dataset
    reaction_dataset_path = args.reaction_dataset
    ground_truth_path = args.ground_truth
    chi_list = args.chi
    outputdir = args.outputdir
    mode = args.mode
    model_type = args.model
    enzymeclip_checkpoint_path = args.checkpoint
    gpus = args.gpus
    threads = args.threads
    seed = args.seed
    debug = args.debug
    
    # Set seed for reproducibility
    set_seed(seed, debug)
    
    # Construct EnzymeCLIP model
    protein_model, protein_tokenizer = load_saprot_model(protein_checkpoint_path, protein_lora_config)
    protein_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    reaction_model, reaction_tokenizer = load_rxnfp_model(reaction_checkpoint_path, reaction_lora_config)
    reaction_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    global_batch_size = batch_size * gpus
    
    cfg = {'embedding_dim': embedding_dim,
           'lr': lr,
           'batch_size': batch_size,
           'reaction_model': reaction_model,
           'reaction_tokenizer': reaction_tokenizer,
           'reaction_dim': reaction_dim,
           'reaction_loss': reaction_loss,
           'sequence_model': protein_model,
           'sequence_tokenizer': protein_tokenizer,
           'sequence_dim': protein_dim,
           'sequence_loss': protein_loss,
           'init_logit_scale': init_logit_scale}
    
    if model_type == 'EnzymeCLIP':
        model = EnzymeCLIP(**cfg)
    elif model_type == 'EnzymeCyCLIP':
        cfg['cylambda1'] = cylambda1
        cfg['cylambda2'] = cylambda2
        model = EnzymeCyCLIP(**cfg)
    elif model_type == 'EnzymeSoftCLIP':
        cfg['softcoef'] = softcoef
        model = EnzymeSoftCLIP(**cfg)
    elif model_type == 'EnzymeSoftCyCLIP':
        cfg['cylambda1'] = cylambda1
        cfg['cylambda2'] = cylambda2
        cfg['softcoef'] = softcoef
        model = EnzymeSoftCyCLIP(**cfg)
    
    # Run EnzymeCLIP
    ## Training
    if mode == 'train':
        if train_dataset_path is None or eval_dataset_path is None:
            raise ValueError('EnzymeCLIP training requires training dataset and validation dataset')
        
        # Make dataloaders
        train_dataloader = make_clip_dataloader(dataset_path=train_dataset_path,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                threads=threads,
                                                reaction_tokenizer=reaction_tokenizer,
                                                protein_tokenizer=protein_tokenizer)
        eval_dataloader = make_clip_dataloader(dataset_path=eval_dataset_path,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               threads=threads,
                                               reaction_tokenizer=reaction_tokenizer,
                                               protein_tokenizer=protein_tokenizer)
        
        ### Make directory to save checkpoints and training log
        os.makedirs(outputdir, exist_ok=True)
        
        ### Setting to save a checkpoint at the end of every epoch
        checkpoint_callback = ModelCheckpoint(
            filename='{epoch:02d}-{val_loss:.2f}',
            monitor='val_loss',
            mode='min',
            save_top_k=-1,
            every_n_epochs=1,
            save_weights_only=False,
        )
        
        ### Initialize the PyTorch Lightning trainer
        trainer = pl.Trainer(
            accelerator='gpu',                              # Use GPU for training
            devices=gpus,                                   # Number of GPUs to use
            precision='bf16-true',                          # 16-bit bfloat precision (model weights get cast to torch.bfloat16)
            strategy='ddp_find_unused_parameters_true',     # Distributed Data Parallel strategy
            max_epochs=epoch,                               # Number of epochs
            gradient_clip_val=gradient_clip,                # Value at which to clip gradients
            enable_progress_bar=False,                      # Disable progress bar
            callbacks=[checkpoint_callback],
            logger=pl_loggers.CSVLogger(save_dir=outputdir),
            deterministic=True
        )
        ### Run EnzymeCLIP training
        trainer.fit(model, train_dataloader, eval_dataloader)
    
    ## Evaluation
    if mode == 'eval':
        if protein_dataset_path is None or reaction_dataset_path is None or ground_truth_path is None:
            raise ValueError('EnzymeCLIP evaluation requires protein dataset, reaction dataset, and ground truth')
        
        ### Make directory to save embeddings, cosine similarity matrix, and metrics
        os.makedirs(outputdir, exist_ok=True)
        
        ### Load trained EnzymeCLIP model
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = eval(model_type).load_from_checkpoint(checkpoint_path=enzymeclip_checkpoint_path, **cfg).to(device)
        model.eval()
        
        ### Make dataloaders and embed inputs using trained EnzymeCLIP model
        protein_dataloader = make_protein_dataloader(protein_dataset_path=protein_dataset_path,
                                                     batch_size=batch_size,
                                                     protein_tokenizer=protein_tokenizer,
                                                     device=device)
        protein_embeddings = embed_proteins(model, protein_dataloader)
        torch.save(protein_embeddings.to('cpu'), f'{outputdir}/protein_embeddings.pt')
        
        reaction_dataloader = make_reaction_dataloader(reaction_dataset_path=reaction_dataset_path,
                                                       batch_size=batch_size,
                                                       reaction_tokenizer=reaction_tokenizer,
                                                       device=device)
        reaction_embeddings = embed_reactions(model, reaction_dataloader)
        torch.save(reaction_embeddings.to('cpu'), f'{outputdir}/reaction_embeddings.pt')
        
        ### Compute cosine similarity
        cos_sim_matrix = (reaction_embeddings @ protein_embeddings.T).to('cpu')
        torch.save(cos_sim_matrix, f'{outputdir}/cos_sim_matrix.pt')
        
        ### Compute enrichment factor
        ground_truth = pd.read_table(ground_truth_path, index_col=0)
        # ground_truth = np.load(ground_truth_path)
        eval_result = []
        for chi in chi_list:
            ef_score = evaluate(ground_truth.values, cos_sim_matrix, chi)
            eval_result.append([chi, ef_score])
        df_result = pd.DataFrame(eval_result, columns=['Chi', 'Enrichment Factor'])
        df_result.to_csv(f'{outputdir}/enrichment_factor.tsv', sep='\t', header=True, index=False)
    
    ## Inference
    if mode == 'inference':
        if protein_dataset_path is None and reaction_dataset_path is None:
            raise ValueError('EnzymeCLIP inference requires protein dataset or reaction dataset')
        
        ### Make directory to save embeddings and cosine similarity matrix
        os.makedirs(outputdir, exist_ok=True)
        
        ### Load trained EnzymeCLIP model
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = eval(model_type).load_from_checkpoint(checkpoint_path=enzymeclip_checkpoint_path, **cfg).to(device)
        model.eval()
        
        ### Make dataloaders and embed inputs using trained EnzymeCLIP model
        if protein_dataset_path is not None:
            protein_dataloader = make_protein_dataloader(protein_dataset_path=protein_dataset_path,
                                                         batch_size=batch_size,
                                                         protein_tokenizer=protein_tokenizer,
                                                         device=device)
            protein_embeddings = embed_proteins(model, protein_dataloader)
            torch.save(protein_embeddings.to('cpu'), f'{outputdir}/protein_embeddings.pt')
        if reaction_dataset_path is not None:
            reaction_dataloader = make_reaction_dataloader(reaction_dataset_path=reaction_dataset_path,
                                                           batch_size=batch_size,
                                                           reaction_tokenizer=reaction_tokenizer,
                                                           device=device)
            reaction_embeddings = embed_reactions(model, reaction_dataloader)
            torch.save(reaction_embeddings.to('cpu'), f'{outputdir}/reaction_embeddings.pt')
        
        ### Compute cosine similarity if possible
        if protein_dataset_path is not None and reaction_dataset_path is not None:
            cos_sim_matrix = (reaction_embeddings @ protein_embeddings.T).to('cpu')
            torch.save(cos_sim_matrix, f'{outputdir}/cos_sim_matrix.pt')

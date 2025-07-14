# DeepRES
[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Requirements
- python 3.10.14 (with following packages)
  - numpy 1.26.4
  - pandas 2.2.2
  - scikit-learn 1.4.2
  - pytorch 2.2.2
  - pytorch-cuda 12.1
  - pytorch-lightning 2.4.0
  - lightning 2.4.0
  - peft 0.13.2
  - transformers 4.46.3

By using `environment.yml`, you can build an anaconda environment exactly the same as this research.
```
conda env create -f environment.yml
conda activate deepres
```

## Test run
You can test EnzymeCNN and EnzymeCLIP by running following commands.
### EnzymeCNN
```
python enzymecnn.py \
    --num-inputs 20 \
    --num-channels 600 600 600 600 600 \
    --kernel-size 9 \
    --dilation 3 \
    --dropout 0.1 \
    --hidden-dims 256 \
    --eval-dataset data/sample_proteins.tsv \
    --mode eval \
    --checkpoint checkpoint/best_enzymecnn_checkpoint.pt \
    --outputdir result/testrun/enzymecnn/ \
    --seed 123
```
The result file is `result/testrun/enzymecnn/evaluation_result.tsv`:
| eval_loss | eval_accuracy | eval_f1 | eval_mcc | eval_auc |
| ----- | ---------- | ------- | --------------- | ------- |
| 0.026970707811415195 | 0.985 | 0.9852216748768473 | 0.9704367948586523 | 0.9998 |

### EnzymeCLIP
```
python enzymeclip.py \
    --protein-checkpoint westlake-repl/SaProt_650M_AF2 \
    --protein-use_lora \
    --reaction-checkpoint pretrained/rxnfp \
    --reaction-use_lora \
    --protein-dataset data/sample_enzymes.tsv \
    --reaction-dataset data/sample_reactions.tsv \
    --ground-truth data/sample_ground_truth.tsv \
    --mode eval \
    --model EnzymeCyCLIP \
    --checkpoint checkpoint/best_enzymeclip_checkpoint.ckpt \
    --outputdir result/testrun/enzymeclip/ \
    --gpus 1 \
    --seed 123
```
The result file is `result/testrun/enzymeclip/enrichment_factor.tsv`:
| Chi | Enrichment Factor |
| ----- | ---------- |
| 0.05 | 14.703703703703702 |
| 0.1  | 8.407407407407407 |

## License
DeepRES is released under the [MIT License](LICENSE).
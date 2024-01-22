import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import logging


def get_logger(name="movie_recommender_logger"):
    logger_ = logging.getLogger(name)
    logger_.setLevel(logging.INFO)
    return logger_


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    pooled_sent = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                     min=1e-9)
    return F.normalize(pooled_sent, p=2, dim=1)


def trainer(dataset,
            dataset_eval,
            model,
            loss_func,
            batch_size,
            n_epoch,
            optimizer="adam",
            lr=0.001,
            device='cpu',
            save_checkpoint=False,
            log_results_steps=10,
            gradient_accumulation_step=1,
            eval_step=10,
            scheduler=None,
            **training_config):
    pass


class LMParams:
    tokenizer = ''

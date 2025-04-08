import argparse
import os.path

def main(args):
    import json, time, os, sys, glob
    import shutil
    import warnings
    import numpy as np
    import torch
    from torch import optim
    from torch.utils.data import DataLoader
    import copy
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import os.path
    import subprocess
    from utils import worker_init_fn, loader_pdb_fragments, build_training_clusters, PDB_dataset
    from model_utils import featurize_fragments, get_std_opt, ProteinFeatures, FragmentAssembler, PositionalEncodings

    scaler = torch.cuda.amp.GradScaler()

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    base_folder = args.path_for_outputs
    if not base_folder.endswith('/'):
        base_folder += '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ['model_weights']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)

    PATH = args.previous_checkpoint

    logfile = os.path.join(base_folder, 'log.txt')
    if not PATH or not os.path.exists(logfile):
        with open(logfile, 'w') as f:
            f.write('Epoch\tStep\tTime\tTrainLoss\tTrainOrderAcc\tTrainGapMAE\tValidLoss\tValidOrderAcc\tValidGapMAE\n')

    data_path = args.path_for_training_data
    params = {
        "LIST": os.path.join(data_path, "list.csv"),
        "VAL": os.path.join(data_path, "valid_clusters.txt"),
        "TEST": os.path.join(data_path, "test_clusters.txt"),
        "DIR": data_path,
        "DATCUT": "2030-Jan-01",
        "RESCUT": args.rescut,
        "USE_SUBDIRS": True
    }

    LOAD_PARAM = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'pin_memory': False,
        'num_workers': args.num_workers
    }

    protein_features = ProteinFeatures(
        node_features=args.hidden_dim,
        edge_features=args.hidden_dim,
        top_k=args.num_neighbors,
        augment_eps=args.backbone_noise
    ).to(device)

    def collate_fn(batch):
        return featurize_fragments(batch, device, protein_features)

    train, valid, test = build_training_clusters(params, args.debug)

    train_set = PDB_dataset(list(train.keys()), loader_pdb_fragments, train, params)
    valid_set = PDB_dataset(list(valid.keys()), loader_pdb_fragments, valid, params)

    train_loader = DataLoader(train_set, worker_init_fn=worker_init_fn, collate_fn=collate_fn, **LOAD_PARAM)
    valid_loader = DataLoader(valid_set, worker_init_fn=worker_init_fn, collate_fn=collate_fn, **LOAD_PARAM)

    model = FragmentAssembler(
        embed_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        nhead=args.fragment_transformer_heads,
        dim_feedforward=args.fragment_transformer_ff_dim,
        max_fragments=args.max_fragments_in_seq,
        dropout=args.dropout
    ).to(device)

    if PATH:
        try:
            checkpoint = torch.load(PATH, map_location=device)
            total_step = checkpoint['step']
            epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model checkpoint from {PATH} at epoch {epoch}, step {total_step}")
        except Exception as e:
            print(f"Error loading checkpoint {PATH}: {e}. Starting from scratch.")
            total_step = 0
            epoch = 0
    else:
        total_step = 0
        epoch = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    if PATH and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded optimizer state dict.")
        except Exception as e:
            print(f"Error loading optimizer state dict: {e}")

    criterion_order = nn.CrossEntropyLoss(ignore_index=-100)
    criterion_gap = nn.MSELoss(reduction='none')

    for e in range(args.num_epochs):
        current_epoch = epoch + e

        t0 = time.time()
        model.train()
        train_loss_accum = 0.0
        train_order_acc_accum = 0.0
        train_gap_mae_accum = 0.0
        train_valid_batches = 0

        for _, batch_data in enumerate(train_loader):
            padded_embeddings, bool_attention_mask, padded_target_orders, padded_target_gaps, batch_labels = batch_data

            if padded_embeddings.shape[0] == 0:
                continue

            optimizer.zero_grad()

            if args.mixed_precision:
                with torch.cuda.amp.autocast():
                    order_logits, gap_predictions = model(padded_embeddings, bool_attention_mask)

                    loss_order = criterion_order(
                        order_logits.view(-1, args.max_fragments_in_seq),
                        padded_target_orders.view(-1)
                    )

                    gap_preds_flat = gap_predictions.squeeze(-1)
                    loss_gap_raw = criterion_gap(gap_preds_flat, padded_target_gaps)

                    gap_loss_mask = (padded_target_gaps != -1.0)
                    loss_gap_masked = loss_gap_raw * gap_loss_mask
                    loss_gap = loss_gap_masked.sum() / gap_loss_mask.sum().clamp(min=1)

                    total_loss = (args.order_loss_weight * loss_order +
                                  args.gap_loss_weight * loss_gap)

                scaler.scale(total_loss).backward()

                if args.gradient_norm > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                scaler.step(optimizer)
                scaler.update()
            else:
                order_logits, gap_predictions = model(padded_embeddings, bool_attention_mask)
                loss_order = criterion_order(
                    order_logits.view(-1, args.max_fragments_in_seq),
                    padded_target_orders.view(-1)
                )
                gap_preds_flat = gap_predictions.squeeze(-1)
                loss_gap_raw = criterion_gap(gap_preds_flat, padded_target_gaps)
                gap_loss_mask = (padded_target_gaps != -1.0)
                loss_gap_masked = loss_gap_raw * gap_loss_mask
                loss_gap = loss_gap_masked.sum() / gap_loss_mask.sum().clamp(min=1)
                total_loss = (args.order_loss_weight * loss_order +
                              args.gap_loss_weight * loss_gap)

                total_loss.backward()

                if args.gradient_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)
                optimizer.step()

            train_loss_accum += total_loss.item()

            order_preds = torch.argmax(order_logits, dim=-1)
            order_correct_mask = (padded_target_orders != -100)
            order_acc = ((order_preds == padded_target_orders) * order_correct_mask).sum() / order_correct_mask.sum().clamp(min=1)
            train_order_acc_accum += order_acc.item()

            gap_abs_error = torch.abs(gap_preds_flat - padded_target_gaps) * gap_loss_mask
            gap_mae = gap_abs_error.sum() / gap_loss_mask.sum().clamp(min=1)
            train_gap_mae_accum += gap_mae.item()

            train_valid_batches += 1
            total_step += 1

        avg_train_loss = train_loss_accum / train_valid_batches if train_valid_batches > 0 else 0
        avg_train_order_acc = train_order_acc_accum / train_valid_batches if train_valid_batches > 0 else 0
        avg_train_gap_mae = train_gap_mae_accum / train_valid_batches if train_valid_batches > 0 else 0

        model.eval()
        valid_loss_accum = 0.0
        valid_order_acc_accum = 0.0
        valid_gap_mae_accum = 0.0
        valid_batches = 0
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(valid_loader):
                padded_embeddings, bool_attention_mask, padded_target_orders, padded_target_gaps, batch_labels = batch_data
                if padded_embeddings.shape[0] == 0:
                    continue

                order_logits, gap_predictions = model(padded_embeddings, bool_attention_mask)

                loss_order = criterion_order(
                    order_logits.view(-1, args.max_fragments_in_seq),
                    padded_target_orders.view(-1)
                )
                gap_preds_flat = gap_predictions.squeeze(-1)
                loss_gap_raw = criterion_gap(gap_preds_flat, padded_target_gaps)
                gap_loss_mask = (padded_target_gaps != -1.0)
                loss_gap_masked = loss_gap_raw * gap_loss_mask
                loss_gap = loss_gap_masked.sum() / gap_loss_mask.sum().clamp(min=1)
                total_loss = (args.order_loss_weight * loss_order +
                              args.gap_loss_weight * loss_gap)

                valid_loss_accum += total_loss.item()
                order_preds = torch.argmax(order_logits, dim=-1)
                order_correct_mask = (padded_target_orders != -100)
                order_acc = ((order_preds == padded_target_orders) * order_correct_mask).sum() / order_correct_mask.sum().clamp(min=1)
                valid_order_acc_accum += order_acc.item()

                gap_abs_error = torch.abs(gap_preds_flat - padded_target_gaps) * gap_loss_mask
                gap_mae = gap_abs_error.sum() / gap_loss_mask.sum().clamp(min=1)
                valid_gap_mae_accum += gap_mae.item()

                valid_batches += 1

        avg_valid_loss = valid_loss_accum / valid_batches if valid_batches > 0 else 0
        avg_valid_order_acc = valid_order_acc_accum / valid_batches if valid_batches > 0 else 0
        avg_valid_gap_mae = valid_gap_mae_accum / valid_batches if valid_batches > 0 else 0

        t1 = time.time()
        dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=1)

        log_line = (
            f'{current_epoch+1}\t{total_step}\t{dt}\t'
            f'{avg_train_loss:.4f}\t{avg_train_order_acc:.4f}\t{avg_train_gap_mae:.3f}\t'
            f'{avg_valid_loss:.4f}\t{avg_valid_order_acc:.4f}\t{avg_valid_gap_mae:.3f}\n'
        )
        with open(logfile, 'a') as f:
            f.write(log_line)
        print(log_line.replace('\t', ', ').strip())

        checkpoint_filename_last = os.path.join(base_folder, 'model_weights', 'epoch_last.pt')
        torch.save({
            'epoch': current_epoch + 1,
            'step': total_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': vars(args)
        }, checkpoint_filename_last)

    with ProcessPoolExecutor(max_workers=12) as executor:
        q = queue.Queue(maxsize=3)
        p = queue.Queue(maxsize=3)
        for i in range(3):
            q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
            p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
        pdb_dict_train = q.get().result()
        pdb_dict_valid = p.get().result()
       
        dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length) 
        dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
        
        loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
        loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
        
        reload_c = 0 
        for e in range(args.num_epochs):
            t0 = time.time()
            e = epoch + e
            model.train()
            train_sum, train_weights = 0., 0.
            train_acc = 0.
            if e % args.reload_data_every_n_epochs == 0:
                if reload_c != 0:
                    pdb_dict_train = q.get().result()
                    dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length)
                    loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
                    pdb_dict_valid = p.get().result()
                    dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
                    loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
                    q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
                    p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
                reload_c += 1
            for _, batch in enumerate(loader_train):
                start_batch = time.time()
                X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                elapsed_featurize = time.time() - start_batch
                optimizer.zero_grad()
                mask_for_loss = mask*chain_M
                
                if args.mixed_precision:
                    with torch.cuda.amp.autocast():
                        log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                        _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
           
                    scaler.scale(loss_av_smoothed).backward()
                     
                    if args.gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
                    loss_av_smoothed.backward()

                    if args.gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                    optimizer.step()
                
                loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
            
                train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                train_weights += torch.sum(mask_for_loss).cpu().data.numpy()

                total_step += 1

            model.eval()
            with torch.no_grad():
                validation_sum, validation_weights = 0., 0.
                validation_acc = 0.
                for _, batch in enumerate(loader_valid):
                    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    mask_for_loss = mask*chain_M
                    loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
                    
                    validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                    validation_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                    validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()
            
            train_loss = train_sum / train_weights
            train_accuracy = train_acc / train_weights
            train_perplexity = np.exp(train_loss)
            validation_loss = validation_sum / validation_weights
            validation_accuracy = validation_acc / validation_weights
            validation_perplexity = np.exp(validation_loss)
            
            train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=3)     
            validation_perplexity_ = np.format_float_positional(np.float32(validation_perplexity), unique=False, precision=3)
            train_accuracy_ = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=3)
            validation_accuracy_ = np.format_float_positional(np.float32(validation_accuracy), unique=False, precision=3)
    
            t1 = time.time()
            dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=1) 
            with open(logfile, 'a') as f:
                f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}\n')
            print(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}')
            
            checkpoint_filename_last = base_folder+'model_weights/epoch_last.pt'.format(e+1, total_step)
            torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.optimizer.state_dict(),
                        }, checkpoint_filename_last)

            if (e+1) % args.save_model_every_n_epochs == 0:
                checkpoint_filename = base_folder+'model_weights/epoch{}_step{}.pt'.format(e+1, total_step)
                torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise, 
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.optimizer.state_dict(),
                        }, checkpoint_filename)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--path_for_training_data", type=str, default="my_path/pdb_2021aug02", help="path for loading training data") 
    argparser.add_argument("--path_for_outputs", type=str, default="./exp_020", help="path for logs and model weights")
    argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--num_epochs", type=int, default=200, help="number of epochs to train for")
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2, help="reload training data every n epochs")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000, help="number of training example to load for one epoch")
    argparser.add_argument("--batch_size", type=int, default=10000, help="number of tokens for one batch")
    argparser.add_argument("--max_protein_length", type=int, default=10000, help="maximum length of the protein complext")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")   
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.2, help="amount of noise added to backbone during training")   
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", type=bool, default=False, help="minimal data loading for debugging")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", type=bool, default=True, help="train with mixed precision")
    argparser.add_argument("--num_workers", type=int, default=1, help="number of workers for data loading")
    argparser.add_argument("--fragment_transformer_heads", type=int, default=8, help="number of heads for the fragment transformer")
    argparser.add_argument("--fragment_transformer_ff_dim", type=int, default=1024, help="feed-forward dimension for the fragment transformer")
    argparser.add_argument("--max_fragments_in_seq", type=int, default=1000, help="maximum number of fragments in a sequence")
    argparser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
 
    args = argparser.parse_args()    
    main(args)   

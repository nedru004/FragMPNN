import torch
from torch.utils.data import DataLoader
import csv
from dateutil import parser
import numpy as np
import time
import random
import os

class StructureDataset():
    def __init__(self, pdb_dict_list, verbose=True, truncate=None, max_length=100,
        alphabet='ACDEFGHIKLMNPQRSTVWYX'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }

        self.data = []

        start = time.time()
        for i, entry in enumerate(pdb_dict_list):
            seq = entry['seq']
            name = entry['name']

            bad_chars = set([s for s in seq]).difference(alphabet_set)
            if len(bad_chars) == 0:
                if len(entry['seq']) <= max_length:
                    self.data.append(entry)
                else:
                    discard_count['too_long'] += 1
            else:
                #print(name, bad_chars, entry['seq'])
                discard_count['bad_chars'] += 1

            # Truncate early
            if truncate is not None and len(self.data) == truncate:
                return

            if verbose and (i + 1) % 1000 == 0:
                elapsed = time.time() - start
                #print('{} entries ({} loaded) in {:.1f} s'.format(len(self.data), i+1, elapsed))

            #print('Discarded', discard_count)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class StructureLoader():
    def __init__(self, dataset, batch_size=100, shuffle=True,
        collate_fn=lambda x:x, drop_last=False):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
                batch_max = size
            else:
                clusters.append(batch)
                batch, batch_max = [], 0
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch


def worker_init_fn(worker_id):
    np.random.seed()

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

def get_std_opt(parameters, d_model, step):
    return NoamOpt(
        d_model, 2, 4000, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step
    )




def get_pdbs(data_loader, repeat=1, max_length=10000, num_units=1000000):
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet
    c = 0
    c1 = 0
    pdb_dict_list = []
    t0 = time.time()
    for _ in range(repeat):
        for step,t in enumerate(data_loader):
            t = {k:v[0] for k,v in t.items()}
            c1 += 1
            if 'label' in list(t):
                my_dict = {}
                s = 0
                concat_seq = ''
                concat_N = []
                concat_CA = []
                concat_C = []
                concat_O = []
                concat_mask = []
                coords_dict = {}
                mask_list = []
                visible_list = []
                if len(list(np.unique(t['idx']))) < 352:
                    for idx in list(np.unique(t['idx'])):
                        letter = chain_alphabet[idx]
                        res = np.argwhere(t['idx']==idx)
                        initial_sequence= "".join(list(np.array(list(t['seq']))[res][0,]))
                        if initial_sequence[-6:] == "HHHHHH":
                            res = res[:,:-6]
                        if initial_sequence[0:6] == "HHHHHH":
                            res = res[:,6:]
                        if initial_sequence[-7:-1] == "HHHHHH":
                           res = res[:,:-7]
                        if initial_sequence[-8:-2] == "HHHHHH":
                           res = res[:,:-8]
                        if initial_sequence[-9:-3] == "HHHHHH":
                           res = res[:,:-9]
                        if initial_sequence[-10:-4] == "HHHHHH":
                           res = res[:,:-10]
                        if initial_sequence[1:7] == "HHHHHH":
                            res = res[:,7:]
                        if initial_sequence[2:8] == "HHHHHH":
                            res = res[:,8:]
                        if initial_sequence[3:9] == "HHHHHH":
                            res = res[:,9:]
                        if initial_sequence[4:10] == "HHHHHH":
                            res = res[:,10:]
                        if res.shape[1] < 4:
                            pass
                        else:
                            my_dict['seq_chain_'+letter]= "".join(list(np.array(list(t['seq']))[res][0,]))
                            concat_seq += my_dict['seq_chain_'+letter]
                            if idx in t['masked']:
                                mask_list.append(letter)
                            else:
                                visible_list.append(letter)
                            coords_dict_chain = {}
                            all_atoms = np.array(t['xyz'][res,])[0,] #[L, 14, 3]
                            coords_dict_chain['N_chain_'+letter]=all_atoms[:,0,:].tolist()
                            coords_dict_chain['CA_chain_'+letter]=all_atoms[:,1,:].tolist()
                            coords_dict_chain['C_chain_'+letter]=all_atoms[:,2,:].tolist()
                            coords_dict_chain['O_chain_'+letter]=all_atoms[:,3,:].tolist()
                            my_dict['coords_chain_'+letter]=coords_dict_chain
                    my_dict['name']= t['label']
                    my_dict['masked_list']= mask_list
                    my_dict['visible_list']= visible_list
                    my_dict['num_of_chains'] = len(mask_list) + len(visible_list)
                    my_dict['seq'] = concat_seq
                    if len(concat_seq) <= max_length:
                        pdb_dict_list.append(my_dict)
                    if len(pdb_dict_list) >= num_units:
                        break
    return pdb_dict_list



class PDB_dataset(torch.utils.data.Dataset):
    def __init__(self, IDs, loader, train_dict, params):
        self.IDs = IDs
        self.train_dict = train_dict
        self.loader = loader
        self.params = params

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        # Original logic picked a random item if multiple existed for an ID.
        # Assuming train_dict[ID] is a list of [pdbid, chid] pairs, 
        # and we want to process one specific pair.
        # If train_dict[ID] can have multiple chains, we might need to handle that.
        # For now, assume it gives one item or we take the first.
        item = self.train_dict[ID][0] # Or handle multiple items if necessary
        # Add error handling in case loader returns None
        out = None
        attempt = 0
        max_attempts = 5 # Avoid infinite loops if many files are bad
        while out is None and attempt < max_attempts:
            # If train_dict[ID] could have multiple items, select one randomly here
            # sel_idx = np.random.randint(0, len(self.train_dict[ID]))
            # item = self.train_dict[ID][sel_idx]
            out = self.loader(item, self.params)
            if out is None:
                # Try a different ID if the current one fails consistently
                index = (index + 1) % len(self.IDs)
                ID = self.IDs[index]
                item = self.train_dict[ID][0]
            attempt += 1

        if out is None:
            # If still None after attempts, raise an error or return a dummy item
            # This might indicate a larger issue with data availability
            print(f"Warning: Failed to load valid data after {max_attempts} attempts.")
            # Returning a dummy item might hide errors; consider raising Exception
            # For now, let's return a dummy to avoid crashing training loop
            return {'shuffled_xyz': [], 'shuffled_seq': [], 'target_order': torch.tensor([]), 'target_gaps': torch.tensor([]), 'label': 'dummy'}
            # raise RuntimeError(f"Failed to load valid data after {max_attempts} attempts.")
        return out

def loader_pdb_fragments(item, params):
    """Loads fragment data for a given PDB ID and chain ID."""
    pdbid, chid = item[0].split('_')
    # Construct the path to the fragment file
    # Assuming params['DIR'] points to the root directory where parse_cif_noX outputted files
    # e.g., /path/to/output/1a2b_A_frags.pt
    # Adjust the PREFIX logic if the output structure is different
    # Original code used pdbid[1:3] for subdirectory, let's keep that pattern if it exists
    # Check if the directory structure includes subdirs like /pdb/ab/1abc
    subdir = ""
    if params.get('USE_SUBDIRS', True): # Add a param to control this if needed
        subdir = f"{pdbid[1:3]}/{pdbid}"
    else:
        subdir = pdbid

    PREFIX = os.path.join(params['DIR'], subdir) # Base path for the PDB
    FRAG_PATH = f"{PREFIX}_{chid}_frags.pt"

    # Check if fragment file exists
    if not os.path.isfile(FRAG_PATH):
        # print(f"Warning: Fragment file not found: {FRAG_PATH}")
        return None # Signal that loading failed

    # Load fragment data
    try:
        data = torch.load(FRAG_PATH)
    except Exception as e:
        # print(f"Warning: Failed to load {FRAG_PATH}: {e}")
        return None

    fragments = data.get('fragments')
    # order = data.get('order') # Original order [0, 1, 2...]
    gaps = data.get('gaps') # Gaps between fragments in original order

    # Check if data is valid
    if not fragments or gaps is None:
        # print(f"Warning: Invalid or empty data in {FRAG_PATH}")
        return None

    num_fragments = len(fragments)
    if num_fragments <= 1:
        # print(f"Warning: Not enough fragments ({num_fragments}) in {FRAG_PATH}")
        return None # Need at least two fragments to predict order/gaps

    # --- Prepare shuffled data and targets ---

    # Indices for shuffling
    shuffle_indices = list(range(num_fragments))
    random.shuffle(shuffle_indices)

    # Create shuffled data
    shuffled_fragments = [fragments[i] for i in shuffle_indices]
    shuffled_xyz = [f['xyz'] for f in shuffled_fragments] # List of Tensors
    shuffled_seq = [f['seq'] for f in shuffled_fragments] # List of strings

    # Create target order: target_order[i] is the original index of the fragment at shuffled position i
    # Example: fragments = [A, B, C], order = [0, 1, 2]
    #          shuffle_indices = [2, 0, 1] -> shuffled_fragments = [C, A, B]
    #          target_order should indicate C was 2, A was 0, B was 1 -> [2, 0, 1]
    target_order = torch.tensor(shuffle_indices, dtype=torch.long)

    # Create target gaps: Gaps correspond to the original sequence order.
    # gaps[i] is the gap *after* fragments[i] (before fragments[i+1])
    # We need to provide these gaps, maybe associated with the fragment *preceding* the gap.
    target_gaps = torch.tensor(gaps, dtype=torch.float) # Use float for potential regression target

    # The model needs to predict the order and gaps. How the gaps target is used depends
    # on the model architecture (e.g., predict gap after placing a fragment).

    return {
        'shuffled_xyz': shuffled_xyz, # List of coordinate tensors
        'shuffled_seq': shuffled_seq, # List of sequence strings
        'target_order': target_order,   # Tensor mapping shuffled index to original index
        'target_gaps': target_gaps,    # Tensor of gaps between original fragments (len N-1)
        'label': item[0]           # Original PDBID_CHID label
    }

def build_training_clusters(params, debug):
    val_ids = set()
    if params['VAL'] and os.path.exists(params['VAL']):
       try:
           val_ids = set([int(l) for l in open(params['VAL']).readlines()])
       except ValueError: # Handle cases where file might be empty or non-integer
           print(f"Warning: Could not parse integers from {params['VAL']}")
           val_ids = set()

    test_ids = set()
    if params['TEST'] and os.path.exists(params['TEST']):
       try:
           test_ids = set([int(l) for l in open(params['TEST']).readlines()])
       except ValueError:
           print(f"Warning: Could not parse integers from {params['TEST']}")
           test_ids = set()

    if debug:
        val_ids = set()
        test_ids = set()

    rows_all = []
    # read & clean list.csv
    if params['LIST'] and os.path.exists(params['LIST']):
        with open(params['LIST'], 'r') as f:
            reader = csv.reader(f)
            try:
                next(reader) # Skip header
                # Filter based on resolution and date cutoffs
                for r in reader:
                    try:
                        # Check if row has enough columns and values are convertible
                        if len(r) >= 5 and r[2] and r[1]:
                             res = float(r[2])
                             date = parser.parse(r[1])
                             cath_id = int(r[4]) # Assuming CATH ID is needed as integer key
                             if res <= params['RESCUT'] and date <= parser.parse(params['DATCUT']):
                                 rows_all.append([r[0], r[3], cath_id]) # [pdbid_chid, ?, cath_id]
                    except (ValueError, TypeError, parser.ParserError) as e:
                        # print(f"Skipping row due to parsing error: {r} - {e}")
                        pass # Skip rows with bad formatting
            except StopIteration:
                pass # Handle empty file
    else:
        print(f"Warning: List file not found or specified: {params.get('LIST', 'Not Specified')}")

    # compile training and validation sets
    train = {}
    valid = {}
    test = {}

    rows_to_process = rows_all
    if debug:
        # Use a small subset for debugging
        rows_to_process = rows_all[:20]
        print(f"Debug mode: Using {len(rows_to_process)} entries.")

    for r in rows_to_process:
        pdb_chid, chain_letter, cath_id = r # Unpack assumed structure
        item = [pdb_chid, chain_letter] # Store original format needed by loader?
                                        # Loader currently uses item[0] which is pdb_chid

        # Check if the fragment file actually exists before adding to sets
        subdir = f"{pdb_chid[1:3]}/{pdb_chid.split('_')[0]}" if params.get('USE_SUBDIRS', True) else pdb_chid.split('_')[0]
        PREFIX = os.path.join(params['DIR'], subdir)
        FRAG_PATH = f"{PREFIX}_{pdb_chid.split('_')[1]}_frags.pt"

        # Only add if the fragment file exists
        # This pre-filters the dataset items
        if not os.path.isfile(FRAG_PATH):
            # print(f"Skipping {pdb_chid}: Fragment file not found at {FRAG_PATH}")
            continue

        # Assign to train/valid/test based on CATH ID
        if cath_id in val_ids:
            valid.setdefault(cath_id, []).append(item)
        elif cath_id in test_ids:
            test.setdefault(cath_id, []).append(item)
        else:
            train.setdefault(cath_id, []).append(item)

    if debug:
        # If in debug mode, often useful to have valid = train
        # valid = train # Uncomment if needed for debug runs
        if not train:
             print("Warning: No training data collected in debug mode.")
        if not valid:
             print("Warning: No validation data collected in debug mode.")

    print(f"Data loading finished. Train items: {sum(len(v) for v in train.values())}, Valid items: {sum(len(v) for v in valid.values())}, Test items: {sum(len(v) for v in test.values())}")

    # Ensure all sets have at least one entry if not empty, mainly for downstream code
    if not train: train = {0:[]} # Avoid errors if train is empty
    if not valid: valid = {0:[]} # Avoid errors if valid is empty
    if not test: test = {0:[]}   # Avoid errors if test is empty

    return train, valid, test

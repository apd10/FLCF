import torch
import numpy as np
import time
from datasets import *
import torch.utils.data as data
import pdb
from tqdm import tqdm

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import ndcg_score

import concurrent.futures  as futures

from model import NCF, MF, NCFUser
from util import *
from client import *
from FakeRoast.FedOrchestrator import FedOrchestrator
import copy
def get_compression(user_ids, cmp_str):
    '''
        helper to give concise input for compression config
    '''
    compressions = []
    if cmp_str.startswith('2:'):
        x = cmp_str.split(':')[1]
        powers = [int(a) for a in x.split('-')]
        for i in range(len(user_ids)):
            p = int(i * len(powers) / len(user_ids))
            compressions.append(1.0/(2**powers[p]))
    else:
        raise NotImplementedError
    return compressions

def get_server_model(args, total_users, total_items):
    '''
        create a full noncompressed model from args
    '''
    model = None
    if args.model == "NCF":
        model = NCF(total_users, total_items, args.emb_dim, args.ncf_layers, args.ncf_dropout)
    else:
        raise NotImplementedError
    return model

def get_client_model(args, total_users, total_items, compression, seed):
    ''' create a compressable model from args'''
    model = None
    if args.model == "NCF":
        model = NCFUser(total_users, total_items, args.emb_dim, args.ncf_layers, args.ncf_dropout, 
                        compression, seed)
    else:
        raise NotImplementedError
    return model


def setup_and_run_client(user_id, is_compressed, compression, device,
                full_train_dataset, full_test_dataset, args, server_params,
                total_users, total_items):
    # create client (dataset) 
    client = Client(user_id, is_compressed, compression, device,
                full_train_dataset, full_test_dataset, args)
    # create raw model
    if is_compressed:
        raw_model = get_client_model(args, total_users, total_items, 
                          compression, user_id if args.no_consistent_hashing else 2023 ) # hash for consistency
    else:
        raw_model = get_server_model(args, total_users, total_items)
    # set model
    client.download(raw_model, server_params)
    
    # train model
    client.train()
    
    return client

    
def fair(args):
      
    # data 
    DATAFILE="/home/apd10/FLCF/src/data/"+args.dataset+"/"
    train_file = DATAFILE + "train.txt"
    test_file = DATAFILE + "test.txt"

    full_train_dataset = NCFData(train_file, num_neg=args.num_neg)
    full_test_dataset = NCFData(test_file, num_neg=0) # no need for neg samples
        
    print("#data= ", full_train_dataset.__len__())
    print("#test= ", full_test_dataset.__len__())
    # get the assignments user to compression
    user_ids = full_train_dataset.unique_users
    compressions = get_compression(user_ids, args.fair_compressions)

    total_users = max(full_test_dataset.user_max, full_train_dataset.user_max) + 1
    total_items = max(full_test_dataset.item_max, full_train_dataset.item_max) + 1
    
    total_cuda_count = torch.cuda.device_count()

    server_model = get_server_model(args, total_users, total_items)
    full_wts = FedOrchestrator.get_wts_full_single(server_model, is_global=False)

    for t in tqdm(range(args.T)):
        #print("ROUND BEGIN:", t, flush=True)
        sample = np.random.choice(np.arange(len(user_ids)), args.K)
        
        clients = []
        for i in range(args.K):
            s = sample[i]
            user_id = user_ids[s]
            is_compressed = (compressions[s] > 0)
            compression = compressions[s]
            device = i % total_cuda_count
            r = setup_and_run_client(user_id, is_compressed, compression, device,
                                 full_train_dataset, full_test_dataset, args, full_wts,
                                 total_users, total_items)
            clients.append(r)
        torch.cuda.synchronize()

        local_norm_weights = np.array([c.train_dataset.__len__() for c in clients])
        local_norm_weights = local_norm_weights / np.sum(local_norm_weights)
        # issue in user embeddings TODO
        
        full_wts = FedOrchestrator.get_wts([c.model for c in clients], False, False, local_norm_weights)
        param_avg = copy.deepcopy(full_wts)
        FedOrchestrator.set_wts_full(server_model, param_avg)
        if t % args.eval_every == 0:
            server_ndcg = evaluate(server_model, full_train_dataset, full_test_dataset, 
                      total_items, 0, user_batch=10, full=True)
            print("[",t,"/",args.T, "ndcg",server_ndcg, flush=True)


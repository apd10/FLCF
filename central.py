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

def evaluate(model, train_loader, test_loader, total_items, device, user_batch=10, full=False):

    with torch.no_grad():
        all_items = torch.from_numpy(np.arange(total_items)).to(device).long()
        zeros_tensor = torch.zeros(total_items, device=device, dtype=torch.int64)

        users = test_loader.dataset.unique_users
        if not full:
            users = users[:user_batch]
        ndcgs = []
        topk = []

        batches = (len(users) + user_batch - 1) // user_batch

        for i in tqdm(range(batches)):
            users_batch = torch.from_numpy(users[i*user_batch:(i+1)*user_batch]).to(device).long().reshape(-1,1)
            user_tensor = zeros_tensor + users_batch
            items_tensor = all_items.repeat((users_batch.numel(),1))
            fshape = user_tensor.shape

            scores = np.array(model(user_tensor.reshape(-1), items_tensor.reshape(-1)).cpu())
            scores = scores.reshape(*fshape)
            
            mask = np.zeros(scores.shape)
            target = np.zeros(scores.shape)

            for j in range(users_batch.numel()):
                u = users[i*user_batch:(i+1)*user_batch][j]
                train_items = train_loader.dataset.pos_items[u]
                test_items = test_loader.dataset.pos_items[u]
                # mask the train items
                scores[j][train_items] = -10000
                target[j][test_items] = 1
            
            # compute metrics
            ndgc = ndcg_score(target, scores, k=20)
            #meanret = np.mean(target[np.argsort(scores)[-20:]])
            ndcgs.append(ndgc)
            #topk.append(meanret)

        print("[",full,"]Evaluation: ndgc:", np.mean(ndcgs))
        
        
        
    
    
    

def central(args):
    print(args)

    DATAFILE="/home/apd10/FLCF/src/data/"+args.dataset+"/"
    train_file = DATAFILE + "train.txt"
    test_file = DATAFILE + "test.txt"

    full_train_ds = NCFData(train_file, num_neg=args.num_neg)
    full_test_ds = NCFData(test_file, num_neg=0) # no need for neg samples
        
    full_train_loader = data.DataLoader(
                dataset=full_train_ds, batch_size=args.batch_size, shuffle=False
                )
    full_test_loader = data.DataLoader(
                dataset=full_test_ds, batch_size=1024, shuffle=False
                )
    print("#data= ", full_train_ds.__len__())
    print("#test= ", full_test_ds.__len__())

    total_users = max(full_test_ds.user_max, full_train_ds.user_max) + 1
    total_items = max(full_test_ds.item_max, full_train_ds.item_max) + 1
    item_dim = args.emb_dim
    user_dim = args.emb_dim
    total_cuda_count = torch.cuda.device_count()
    
    if args.model == "NCF":
        model = NCF(total_users, total_items, args.emb_dim, args.ncf_layers, args.ncf_dropout)
    elif args.model == "MF":
        model = MF(total_users, total_items, args.emb_dim)
    elif args.model == "NCFU":
        model = NCFUser(total_users, total_items, args.emb_dim, args.ncf_layers, args.ncf_dropout, 0.5, 2023)
    print(model)

    torch.manual_seed(10101)
    np.random.seed(10111)

    device = torch.device("cuda:" + str(0))
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)

    print("epochs", args.T * args.E)
    for epoch in range(args.T*args.E):
        model.train()
        losses = []
        norms = []
        for batch_idx, (users, pos, neg) in tqdm(enumerate(full_train_loader)):
            model.zero_grad()
            loss1, loss2 = model.bpr_loss(torch.LongTensor(users).cuda(), torch.LongTensor(pos).cuda(), torch.LongTensor(neg).cuda())
            loss = loss1 + args.reg_lambda * loss2
            loss.backward()
            optimizer.step()
            losses.append(float(loss1.detach().cpu()))
            norms.append(float(loss2.detach().cpu()))
        if epoch > 0 and epoch % 1 == 0:
            evaluate(model, full_train_loader, full_test_loader, total_items, device)
            if epoch % args.eval_every == 0:
                evaluate(model, full_train_loader, full_test_loader, total_items, device, full=True)
        print(epoch,  "-->", np.mean(losses), np.mean(norms))

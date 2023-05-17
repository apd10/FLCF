Github for FAIR: Federated Averaging in Random Subspaces submission

```
  $ python3 main.py --help
usage: main.py [-h] --algo {fair,central} [--early_stop_thold EARLY_STOP_THOLD] [--early_trend_window EARLY_TREND_WINDOW] [--model {NCFU,NCF,MF}] [--emb_dim EMB_DIM] [--ncf_layers NCF_LAYERS]
               [--ncf_dropout NCF_DROPOUT] --dataset {Gowalla_m1,Gowalla_100_m1,Goodreads_100_m1,AmazonProducts_100_m1,debug_m1} [--fair_compressions FAIR_COMPRESSIONS]
               [--fair_data_loss_compression FAIR_DATA_LOSS_COMPRESSION] [--central_compression CENTRAL_COMPRESSION] [--fair_randomize_user] [--fair_use_fedhm] [--fair_uniform_compression_sample]
               [--reg_lambda REG_LAMBDA] [--K K] [--T T] [--eval_every EVAL_EVERY] [--E E] [--batch_size BATCH_SIZE] [--num_neg NUM_NEG] [--learn_rate LEARN_RATE] [--max_compression MAX_COMPRESSION]
               [--compression COMPRESSION] [--seed SEED] [--hetero_roast] [--no_consistent_hashing]

optional arguments:
  -h, --help            show this help message and exit
  --early_stop_thold EARLY_STOP_THOLD
  --early_trend_window EARLY_TREND_WINDOW
  --model {NCFU,NCF,MF}
  --emb_dim EMB_DIM     embdding dim for user and books (default: 32)
  --ncf_layers NCF_LAYERS
  --ncf_dropout NCF_DROPOUT
  --dataset {Gowalla_m1,Gowalla_100_m1,Goodreads_100_m1,AmazonProducts_100_m1,debug_m1}
  --fair_compressions FAIR_COMPRESSIONS
  --fair_data_loss_compression FAIR_DATA_LOSS_COMPRESSION
  --central_compression CENTRAL_COMPRESSION
  --fair_randomize_user
  --fair_use_fedhm
  --fair_uniform_compression_sample
  --reg_lambda REG_LAMBDA
                        regularization of embeddings (default: 0.000100
  --K K                 number of devices to train each round (default: 10)
  --T T                 number of rounds (default: 500)
  --eval_every EVAL_EVERY
                        number of rounds to eval after (default: 20)
  --E E                 number of epochs for all devices (default: 1)
  --batch_size BATCH_SIZE
                        batch size for all devices (default: 512)
  --num_neg NUM_NEG     num_neg sampled for each positive (default: 10)
  --learn_rate LEARN_RATE
                        learning rate for all devices (default: 0.0010)
  --max_compression MAX_COMPRESSION
                        max_compression only config
  --compression COMPRESSION
                        compression
  --seed SEED           random seed (default: 0)
  --hetero_roast        run roast heterogenous memory models
  --no_consistent_hashing
                        consistent hashing

required arguments:
  --algo {fair,central}
                        algorithm to run: fedavg : Federated Average
```

Example run :
```
CUDA_VISIBLE_DEVICES=0 python3 main.py --algo fair  --dataset Goodreads_100_m1 --learn_rate 1e-3 --reg_lambda 1e-6 --batch_size 512 --num_neg 1 --emb_dim 8 --eval_every 10 --model NCF  --ncf_dropout 0 --fair_compressions 2:0-1  --K 10 --E 5 --T 1000   --early_stop_thold 3:0.1
```

```
output | grep ndcg

[ 10 / 1000 ndcg 0.2619435053580439
[ 20 / 1000 ndcg 0.2629298800576761
[ 30 / 1000 ndcg 0.26537168597843874
[ 40 / 1000 ndcg 0.26832045737333277
[ 50 / 1000 ndcg 0.26696903309057374
[ 60 / 1000 ndcg 0.2655652564392309
[ 70 / 1000 ndcg 0.25866986992498475
[ 80 / 1000 ndcg 0.26506156070516407
[ 90 / 1000 ndcg 0.26681053383227915
[ 100 / 1000 ndcg 0.2705155608901666
[ 110 / 1000 ndcg 0.26496364030598035
[ 120 / 1000 ndcg 0.2714363898170009
[ 130 / 1000 ndcg 0.2708896405285264
[ 140 / 1000 ndcg 0.2744668553929582
[ 150 / 1000 ndcg 0.2734105127499105
[ 160 / 1000 ndcg 0.2743411680723541
[ 170 / 1000 ndcg 0.27683539008907465
[ 180 / 1000 ndcg 0.2748084553293707
[ 190 / 1000 ndcg 0.273878872764404
[ 200 / 1000 ndcg 0.27619014871765024
[ 210 / 1000 ndcg 0.2740663048706515
[ 220 / 1000 ndcg 0.2740047870210224
[ 230 / 1000 ndcg 0.2750593759864052

```

Hyperparms used

|               |           Goodreads-100          |         AmazonProduct-100        |
|:-------------:|:--------------------------------:|:--------------------------------:|
|       E       |                 5                |                 1                |
|    num_neg    |                 1                |                 1                |
|   BPR decay   |               1e-6               |               1e-6               |
|  MLP dropout  |                 0                |                 0                |
|   batchsize   |                512               |                512               |
| learning rate | Best of {1, 1e-1,1e-2,1e-3,1e-4} | Best of {1, 1e-1,1e-2,1e-3,1e-4} |



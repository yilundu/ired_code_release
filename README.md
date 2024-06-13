# Continuous-Space Reasoning Tasks

```
python3 train.py --dataset addition --data-workers 4 --batch_size 2048 --use-innerloop-opt True --supervise-energy-landscape True
python3 train.py --dataset lowrank --data-workers 4 --batch_size 2048 --use-innerloop-opt True --supervise-energy-landscape True
python3 train.py --dataset inverse --data-workers 4 --batch_size 2048 --use-innerloop-opt True --supervise-energy-landscape True
```

# Discrete-Space Reasoning Tasks

```
python3 train.py --dataset sudoku --batch_size 64 --model sudoku   --cond_mask True  --supervise-energy-landscape True   --use-innerloop-opt True
python3 train.py --dataset connectivity-2 --batch_size 512 --model gnn --data-workers 20 --use-innerloop-opt True --supervise-energy-landscape True
```

# Planning Tasks

```
python3 ./gen_planning_dataset.py shortest-path --size 100000   # takes around 10 mins.
python3 ./gen_planning_dataset.py shortest-path-25 --size 10000 # takes around 2mins.
python train.py --dataset shortest-path-1d --model gnn-conv-1d-v2 --data-workers 2 --batch_size 512 --use-innerloop-opt True --supervise-energy-landscape True
```


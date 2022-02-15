import os
from clize import run
from subprocess import call

machine = open("/etc/FZJ/systemname").read().strip()
def launch(
    nodes=1, gpus_per_node=4, local_bs=64, precision="single", 
    model="SLIP_VITB16", expmode="scaling", t=15, run=0, 
    steps=1000, partition="dc-gpu", account="zam", data=None, folder="benchmark",
    ssl_gather_type="all_gather_batch_with_grad",
    clip_gather_type="all_gather_batch",

):
    os.makedirs(folder, exist_ok=True)
    name = f"{nodes}_{gpus_per_node}_{local_bs}_{precision}_{model}_{expmode}_{run}"
    output = f"{folder}/{name}.out"
    precision = "" if precision == "mixed" else "--disable-amp"
    script = f"scripts/run_{machine}_ddp.sh main.py"
    if data is None:
        data = "--dataset coco --root /p/scratch/ccstdl/cherti1/coco --metadata /p/scratch/ccstdl/cherti1/coco/annotations/captions_train2014.json"
    if expmode == "scaling":
        cmd = f"sbatch --gres=gpu:{gpus_per_node} -t {t} --output {output} --error {output} --partition={partition} --account={account} -N {nodes} -n {nodes*gpus_per_node} {script} {data} --model {model} --output-dir benchmark/output/{name} --nb_iter_per_epoch {steps} --batch-size {local_bs} --epochs 1 --only_train --lr 0.001 --clip-gather-type {clip_gather_type} --ssl-gather-type {ssl_gather_type}"
    print(cmd)
    call(cmd,shell=True)
if machine == "jurecadc":
    account = "zam"
    partition = "dc-gpu"
elif machine == "juwelsbooster":
    account = "covidnetx"
    partition = "booster"


def main(
    *, 
    bs=64, 
    data="--dataset synthetic", 
    nodes='1,2,4,8,16,32,64,128,256', 
    precision="mixed", 
    model="SLIP_VITB16", 
    t=120, 
    steps=1000, 
    folder="benchmark",
    ssl_gather_type="all_gather_batch_with_grad",
    clip_gather_type="all_gather_batch",
    gpus_per_node=4,
    run="0",
):
    nodes = nodes.split(",")
    nodes = map(int, nodes)
    runs = run.split(",")
    for run in runs:
        for node in nodes:
            launch(
                nodes=node, local_bs=bs, precision=precision, model=model, t=t, run=run, 
                steps=steps, partition=partition, account=account, data=data, folder=folder,
                ssl_gather_type=ssl_gather_type,
                clip_gather_type=clip_gather_type,
                gpus_per_node=gpus_per_node,
            )

if __name__ == "__main__":
    run(main)

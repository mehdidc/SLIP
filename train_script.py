from glob import glob
import os
from clize import run
from subprocess import call

machine = open("/etc/FZJ/systemname").read().strip()
datasets = {
    "coco": "--dataset coco --root /p/scratch/ccstdl/cherti1/coco --metadata /p/scratch/ccstdl/cherti1/coco/annotations/captions_train2014.json",
    "laion400M": '--dataset wds --root "/p/scratch/ccstdl/katta1/LAION-400M/laion400m-dat-release/{00000..41455}.tar"',
    "cc12m": ' --root "/p/scratch/ccstdl/cherti1/conceptual-captions-12m-lmdb" --dataset lmdb_multiple'
}
def main(
    *,
    nodes=16, 
    gpus_per_node=4, 
    local_bs=64, 
    precision="mixed", 
    model="SLIP_VITB16", 
    t=60*6, 
    nb_iter_per_epoch=0,
    partition:str=None, 
    account="zam", 
    data="coco", 
    folder="results/test",
    ssl_gather_type="all_gather_batch_with_grad",
    clip_gather_type="all_gather_batch",
    lr=3e-3,
    wd=0.1,
    epochs=25,
    warmup_epochs=1,
    val_root="datasets/imagenet-1K-lmdb/val",
    val_dataset="lmdb",
    val_template="imagenet",
    only_evaluate=False,
):
    os.makedirs(folder, exist_ok=True)
    nb_runs = len(glob(os.path.join(folder, "out_*")))
    run_id = nb_runs + 1
    output = f"{folder}/out_{run_id}"
    error = f"{folder}/err_{run_id}"
    precision = "" if precision == "mixed" else "--disable-amp"
    script = f"scripts/run_{machine}_ddp.sh main.py"
    evaluate = "--evaluate" if only_evaluate else ""
    iters = f"--nb_iter_per_epoch {nb_iter_per_epoch}" if nb_iter_per_epoch else ""
    data = datasets[data]
    if partition is None:
        partition = ""
    else:
        partition = f"--partition {partition}"
    cmd = f"sbatch {partition} --gres=gpu:{gpus_per_node} -t {t} --output {output} --error {error} -N {nodes} -n {nodes*gpus_per_node} {script} {data} --model {model} --output-dir {folder} {iters} --batch-size {local_bs} --epochs {epochs} --lr {lr} --wd {wd} --clip-gather-type {clip_gather_type} --ssl-gather-type {ssl_gather_type} --warmup-epochs={warmup_epochs} --val-dataset {val_dataset} --val-root {val_root} --val-template {val_template} {evaluate}"
    print(cmd)
    call(cmd,shell=True)

if __name__ == "__main__":
    run(main)


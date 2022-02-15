import os
from clize import run
from subprocess import call

machine = open("/etc/FZJ/systemname").read().strip()
datasets = {
    "coco": "--dataset coco --root /p/scratch/ccstdl/cherti1/coco --metadata /p/scratch/ccstdl/cherti1/coco/annotations/captions_train2014.json",
    "laion400M": '--dataset wds --root "/p/scratch/ccstdl/katta1/LAION-400M/laion400m-dat-release/{00000..41455}.tar"'
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
    partition="dc-gpu", 
    account="zam", 
    data="coco", 
    folder="results/test",
    ssl_gather_type="all_gather_batch_with_grad",
    clip_gather_type="all_gather_batch",
    lr=3e-3,
    wd=0.1,
    epochs=25,
):
    os.makedirs(folder, exist_ok=True)
    output = f"{folder}/out"
    error = f"{folder}/err"
    precision = "" if precision == "mixed" else "--disable-amp"
    script = f"scripts/run_{machine}_ddp.sh main.py"
    iters = f"--nb_iter_per_epoch {nb_iter_per_epoch}" if nb_iter_per_epoch else ""
    data = datasets[data]
    cmd = f"sbatch --gres=gpu:{gpus_per_node} -t {t} --output {output} --error {error} -N {nodes} -n {nodes*gpus_per_node} {script} {data} --model {model} --output-dir {folder} {iters} --batch-size {local_bs} --epochs {epochs} --lr {lr} --wd {wd} --clip-gather-type {clip_gather_type} --ssl-gather-type {ssl_gather_type}"
    print(cmd)
    call(cmd,shell=True)

if __name__ == "__main__":
    run(main)


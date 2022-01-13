machine=$(cat /etc/FZJ/systemname)
if [[ "$machine" == jurecadc ]]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    ml use $OTHERSTAGES
    ml Stages/2020
    ml GCC/9.3.0
    ml OpenMPI/4.1.0rc1
    ml CUDA/11.0
    ml cuDNN/8.0.2.39-CUDA-11.0
    ml NCCL/2.8.3-1-CUDA-11.0
    ml PyTorch
    ml Horovod/0.20.3-Python-3.8.5
    ml scikit
    source /p/project/covidnetx/environments/jureca/bin/activate 
fi
if [[ "$machine" == juwelsbooster ]]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    ml purge
    ml use $OTHERSTAGES
    ml Stages/2020
    ml GCC/10.3.0
    ml OpenMPI/4.1.1
    ml Python
    source /p/project/covidnetx/environments/juwels_booster/bin/activate 
    ml PyTorch
    ml Horovod/0.23.0-Python-3.8.5
    ml scikit
    export PYTHONPATH=/p/project/covidnetx/environments/juwels_booster/lib/python3.8/site-packages:$PYTHONPATH
fi
if [[ "$machine" == jusuf ]]; then
    ml purge
    ml use $OTHERSTAGES
    ml Stages/2020
    ml GCC/9.3.0
    ml OpenMPI/4.1.0rc1
    ml CUDA/11.0
    ml cuDNN/8.0.2.39-CUDA-11.0
    ml NCCL/2.8.3-1-CUDA-11.0
    ml PyTorch
    ml Horovod/0.20.3-Python-3.8.5
    #ml scikit
    source /p/project/covidnetx/environments/jusuf/bin/activate
fi

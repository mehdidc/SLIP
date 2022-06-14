RUN="python eval_clip_benchmark.py --resume results/laion_01apr2022/checkpoint.pt"
for mod in SLIP_ViT-B/32,laion400m_e32 ViT-B-32,openai;do
    model=$(echo $mod|cut -d, -f1)
    pretrained=$(echo $mod|cut -d, -f2)

    ds=imagenet1k
    echo $ds $model $pretrained
    $RUN --dataset_root=/p/scratch/ccstdl/cherti1/imagenet-raw --dataset=imagenet1k --task=zeroshot_classification --pretrained=$pretrained --model=$model --output="${ds}_${pretrained}_${model}.json"  --batch_size=512 --num_workers=24

    for ds in flickr8k flickr30k;do
        echo $ds $model $pretrained
        $RUN --dataset_root=/p/scratch/ccstdl/cherti1/$ds/Images --annotation_file=/p/scratch/ccstdl/cherti1/$ds/captions.txt --dataset=$ds --task=zeroshot_retrieval --recall_k 1 5 10 --pretrained=$pretrained --model=$model --output="${ds}_${pretrained}_${model}.json"  --batch_size=512 --num_workers=24
    done
    
    ds=mscoco_captions
    echo $ds $model $pretrained

    $RUN --dataset_root=/p/scratch/ccstdl/cherti1/coco/val2017 --annotation_file=/p/scratch/ccstdl/cherti1/coco/annotations/captions_val2017.json --dataset=$ds --task=zeroshot_retrieval --recall_k 1 5 10 --pretrained=$pretrained --model=$model --output="${ds}_${pretrained}_${model}.json"  --batch_size=512 --num_workers=24

    ds=fer2013
    echo $ds $model $pretrained
    $RUN --dataset_root=/p/scratch/ccstdl/cherti1/fer-2013 --dataset=fer2013 --task=zeroshot_classification --pretrained=$pretrained --model=$model --output="${ds}_${pretrained}_${model}.json"  --batch_size=512 --num_workers=24


    for ds in cifar10 cifar100 voc2007 food101 sun397 cars fgvc_aircraft dtd pets caltech101 flowers mnist stl10 eurosat gtsrb country211 pcam renderedsst2;do
        echo $ds $model $pretrained
        $RUN --dataset=$ds --task=zeroshot_classification --pretrained=$pretrained --model=$model --output="${ds}_${pretrained}_${model}.json"  --batch_size=512 --num_workers=24
    done
done

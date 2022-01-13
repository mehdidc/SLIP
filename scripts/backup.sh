for v in results/*;do
    cd $v
    echo $v
    if [ ! -f archive.tar ]; then
        tar cf archive.tar *.jpg
        rm *.jpg
    fi
    cd -
done

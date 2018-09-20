if [[ $# -ne 6 ]]; then
  echo "$0 lang subword model_type lr bs ud"
  exit 1
fi

lang=$1
subword=$2
model_type=$3
lr=$4
bs=$5
ud=$6

emb_base=/mnt/hdd/yz568/model/hyp
emb_path=$emb_base/$subword/$lang/$lang.$subword.$model_type.ep5.lr$lr.bs$bs.vec.txt

python3 -u crop_embeds.py --in_embeds $emb_path \
  --input_files /mnt/hdd/yz568/data/ud/$lang/$ud \
  --out_emb $lang.$subword.$model_type.ep5.lr$lr.bs$bs.txt

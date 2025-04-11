IN=$1

BASE=`basename $1`
#POS_MODEL=pos_tagging_model_19_finetuned
POS_MODEL=pos_tagging_model_19_gysbert
TAGGED_FILES=`dirname $1`


TAGGED=$TAGGED_FILES/$BASE.pos
LEMMATIZED=$TAGGED_FILES/$BASE.lem

echo "$TAGGED $LEMMATIZED"


python ../tagging/pos_tagging.py $1 $TAGGED ../data/tagging/tagging_models/$POS_MODEL
echo "POS tagging completed; lemmatize to $LEMMATIZED"
python lemmatize-tsv.py $TAGGED ../data/byt5-lem-hilex-19/checkpoint-53500/ ../data/byt5-lem-hilex-19 > $LEMMATIZED

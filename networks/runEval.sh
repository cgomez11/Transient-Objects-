DATA_PATH='/media/user_home2/cgomez11/Astronomy/Networks/Data/Dataset_3channels_npy_original_MORE//'
SAVE='/media/SSD3/Astronomy/models/densenetOwn/5classes_problem/TnT/DN_32_70_original_FT_halfTrain_pre6casses_7/'
CONFIG='3channels_original_TnT_halfTrain_halfEval'
RESUME='/media/SSD3/Astronomy/models/densenetOwn/5classes_problem/TnT/DN_32_70_original_FT_halfTrain_pre6casses_7'
BS=6
VAL='validation' #validation or train
GR=32
DEPTH=70
GPU='2'
for value in {0..49}
do
	MODEL_PATH="$RESUME/complete_model_$value.pth"
	echo $MODEL_PATH
	python evaluateModel.py --data_path $DATA_PATH --path_to_save $SAVE --gpuNum $GPU --config $CONFIG --resume $MODEL_PATH --batch_size $BS --nClasses 6 --nChannels 3 --validate $VAL --GR $GR --depth $DEPTH
done

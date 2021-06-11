config_file=L2AC_train.yaml
file=config/$config_file
var1=name
var2=top_n
path=$pwd
python_script=train_l2ac.py

var_1=name	
array_1=(0006 0007)
var_2=top_n
array_2=(1 2)
len=${#array_1[@]}
for ((i=0;i<$len; i++))
do
	echo "$var_1 = ${array_1[$i]}"
	echo "$var_2 = ${array_2[$i]}"
	sed -i "s/$var_2:.*/$var_2: ${array_2[$i]}/" $file
	sed -i "s/$var_1:.*/$var_1: '${array_1[$i]}'/"  $file
	python $python_script $config_file

done
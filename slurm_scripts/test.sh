config_file=L2AC_train.yaml
file=config/$config_file
var1=name
var2=top_n
path=$pwd
python_script=train_l2ac.py

var_1=name	
array_1=(0006 0007 0008 0009 0010 0011 0012 0013 0014 0015)
var_2=top_n
array_2=(1 2 3 4 5 6 7 8 9 10)
len=${#array_1[@]}
for ((i=0;i<$len; i++))
do
	echo "$var_1 = ${array_1[$i]}"
	echo "$var_2 = ${array_2[$i]}"
	sed -i "s/$var2:.*/$var2: ${array_2[$i]}/" $file
	sed -i "s/$var1:.*/$var1: ${array_1[$i]}/"  $file
	python $python_script $config_file

done
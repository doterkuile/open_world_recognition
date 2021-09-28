## TODO

- Find why my L2AC results on amazon dataset are too good to be true, check influence of training in two steps

- Set extended similarity to default

- Does not finetuning dataset make a significant difference in performance 

- Create a test model script: for real deployment -> input image retrieve nine similar classes compare and conclude

- Check if influence on ImageNet pretraining is not too much of a deal for TinyImageNet by running on CIFAR100 as well
	Now start runs on tinyimagenet without finetuning but with a lot of l2ac_train=160


- Make run of L2AC with all default settings and run with best settings to check difference in performance on tinyimagenet and CIFAR100

# Evaluate new object

-input_dataset -> extract features and get labels
-memory_dataset -> extract features and get labels

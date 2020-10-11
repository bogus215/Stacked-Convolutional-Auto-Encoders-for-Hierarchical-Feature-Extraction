import sys
import gc

#%% cifar
for exp in [1,2,3,4]:
    script_descriptor = open("train_autoencoder.py", encoding ='utf-8')
    a_script = script_descriptor.read()
    sys.argv = ["train_autoencoder.py", "--epoch", "5000",'--printevery','100','--experiment',f'{exp}_CIFAR', '--dataset','CIFAR',
                '--kernel-num','100','--kernel-sizes','5','--input_dim','32','--input_dim_channel','3']
    try:
        print(sys.argv)
        print('start')
        exec(a_script)
        gc.collect()
    except:
        print('failed')


#%% mnist
for exp in [1,2,3,4]:
    script_descriptor = open("train_autoencoder.py", encoding ='utf-8')
    a_script = script_descriptor.read()
    sys.argv = ["train_autoencoder.py", "--epoch", "1000",'--printevery','100','--experiment',f'{exp}_MINIST','--dataset','MNIST',
                '--kernel-num','20','--kernel-sizes','7','--input_dim','28','--input_dim_channel','1']
    try:
        print(sys.argv)
        print('start')
        exec(a_script)
        gc.collect()
    except:
        print('failed')


#%% classification
batch_size = [128,64,32]
for data in ["CIFAR","MNIST"]:
# for data in ["CIFAR"]:
    for ind, data_size in enumerate(['50000','10000','1000']):
        script_descriptor = open("train_classifier.py", encoding ='utf-8')
        a_script = script_descriptor.read()
        if data == "MNIST":
            sys.argv = ["train_classifier.py", "--epoch", "1000",'--printevery','100','--experiment',f'{data}_classification_datasize_{data_size}','--dataset',f'{data}',
                       '--input_dim','28','--input_dim_channel','1','--batch_size',f'{batch_size[ind]}']
        else:

            sys.argv = ["train_classifier.py", "--epoch", "1000",'--printevery','100','--experiment',f'{data}_classification_datasize_{data_size}','--dataset',f'{data}',
                       '--input_dim','32','--input_dim_channel','3' , '--batch_size' ,f'{batch_size[ind]}']


        try:
            print(sys.argv)
            print('start')
            exec(a_script)
            gc.collect()
        except:
            print('failed')


#%% for ready fine tune
batch_size = [128,64,32]
for data in ["CIFAR","MNIST"]:
    for ind, data_size in enumerate(['50000','10000','1000']):

        script_descriptor = open("train_autoencoder_fine.py", encoding ='utf-8')
        a_script = script_descriptor.read()
        if data == "MNIST":
            sys.argv = ["train_autoencoder_fine.py", "--epoch", "1000",'--printevery','100','--experiment',f'{data}_ready_fine_tune_datasize_{data_size}','--dataset',f'{data}',
                       '--input_dim','28','--input_dim_channel','1','--batch_size',f'{batch_size[ind]}']
        else:

            sys.argv = ["train_autoencoder_fine.py", "--epoch", "1000",'--printevery','100','--experiment',f'{data}_ready_fine_tune_datasize_{data_size}','--dataset',f'{data}',
                       '--input_dim','32','--input_dim_channel','3' , '--batch_size' , f"{batch_size[ind]}"]


        try:
            print(sys.argv)
            print('start')
            exec(a_script)
            gc.collect()
        except:
            print('failed')


#%% classification using pretrained filter
batch_size = [128,64,32]
for data in ["CIFAR","MNIST"]:
# for data in ["CIFAR",]:
    for ind, data_size in enumerate(['50000','10000','1000']):

    # for ind, data_size in enumerate(['50000']):

        script_descriptor = open("train_classifier_fine.py", encoding ='utf-8')
        a_script = script_descriptor.read()
        if data == "MNIST":
            sys.argv = ["train_classifier_fine.py", "--epoch", "1000",'--printevery','100','--experiment',f'{data}_fine_classification_datasize_{data_size}','--dataset',f'{data}',
                       '--input_dim','28','--input_dim_channel','1','--batch_size',f'{batch_size[ind]}']
        else:

            sys.argv = ["train_classifier_fine.py", "--epoch", "1000",'--printevery','100','--experiment',f'{data}_fine_classification_datasize_{data_size}','--dataset',f'{data}',
                       '--input_dim','32','--input_dim_channel','3' , '--batch_size' , f"{batch_size[ind]}"]

        try:
            print(sys.argv)
            print('start')
            exec(a_script)
            gc.collect()
        except:
            print('failed')
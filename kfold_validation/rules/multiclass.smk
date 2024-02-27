import os

rule run_cv_model:
    input: 
        df_train = os.path.join(config['output_dir'], f"{config['srate']}Hz/trial_{config['trial']}/", "fold_{kfold}/df_train.csv"),
        df_val = os.path.join(config['output_dir'], f"{config['srate']}Hz/trial_{config['trial']}/", "fold_{kfold}/df_val.csv")
    log:
        os.path.join(config['output_dir'], f"{config['srate']}Hz/trial_{config['trial']}/logs/", "cv_model_{kfold}.log")
    params:
        python_file = config['python_file'],
        architecture = config['architecture'],
        out_dir = os.path.join(config['output_dir'], f"{config['srate']}Hz/trial_{config['trial']}/", "fold_{kfold}/"),
        feature = config['feature'],
        srate =  config['srate'],
    resources:
        mem_mb = 128000,
        time = 1440,
    output:
        out_results = os.path.join(config['output_dir'], f"{config['srate']}Hz/trial_{config['trial']}/", "fold_{kfold}/results_val.txt")
    threads: 32
    shell:
        "python {params.python_file} -m {params.architecture} -p {params.out_dir} -c {threads} -f {params.feature} -s {params.srate} -df_val {input.df_val} -df_train {input.df_train}"
import os

rule run_cv_model:
    input: 
        df_train = os.path.join(config['output_dir'], f"{config['srate']}Hz/trial_{config['trial']}/", "fold_{kfold}/df_train.csv"),
        df_val = os.path.join(config['output_dir'], f"{config['srate']}Hz/trial_{config['trial']}/", "fold_{kfold}/df_val.csv")
    log:
        os.path.join(config['output_dir'], f"{config['srate']}Hz/trial_{config['trial']}/logs/", "cv_model_{kfold}_{category}.log")
    params:
        python_file = config['python_file'],
        architecture = config['architecture'],
        out_dir = os.path.join(config['output_dir'], f"{config['srate']}Hz/trial_{config['trial']}/", "fold_{kfold}/{category}/"),
        feature = config['feature'],
        srate =  config['srate'],
        category = '{category}'
    resources:
        mem_mb = 128000,
        time = 720,
    output:
        out_model = os.path.join(config['output_dir'], f"{config['srate']}Hz/trial_{config['trial']}/", "fold_{kfold}/{category}/best_model.pt"),
        out_results = os.path.join(config['output_dir'], f"{config['srate']}Hz/trial_{config['trial']}/", "fold_{kfold}/{category}/results_val.txt")
    threads: 32
    shell:
        "python {params.python_file} -m {params.architecture} -p {params.out_dir} -c {threads} -f {params.feature} -s {params.srate} -cat {params.category} -df_val {input.df_val} -df_train {input.df_train}"

rule merge_binary_models:
    input: 
        df_train = os.path.join(config['output_dir'], f"{config['srate']}Hz/trial_{config['trial']}/", "fold_{kfold}/df_train.csv"),
        df_val = os.path.join(config['output_dir'], f"{config['srate']}Hz/trial_{config['trial']}/", "fold_{kfold}/df_val.csv"),
        model_paths = expand(os.path.join(config['output_dir'], f"{config['srate']}Hz/trial_{config['trial']}/", "fold_{kfold}/{category}/best_model.pt"), category=config['category'], allow_missing=True),
        results_paths = expand(os.path.join(config['output_dir'], f"{config['srate']}Hz/trial_{config['trial']}/", "fold_{kfold}/{category}/results_val.txt"), category=config['category'], allow_missing=True)
    resources:
        mem_mb = 16000,#128000,
        time = 720,
    threads: 4,#32
    log:
        os.path.join(config['output_dir'], f"{config['srate']}Hz/trial_{config['trial']}/logs/", "merge_models_{kfold}.log")
    output:
        out_results = os.path.join(config['output_dir'], f"{config['srate']}Hz/trial_{config['trial']}/", "fold_{kfold}/results_val.txt")
    script: os.path.join(workflow.basedir,'scripts/merge_models.py')


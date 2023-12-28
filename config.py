import sys
from utils.utils_mytorch import parse_args

BASIC_CONFIG = {
    'BATCH_SIZE': 128,
    'DATASET': 'wd50k',
    'DEVICE': 'cpu',
    'EMBEDDING_DIM': 200,
    'ENT_POS_FILTERED': True,
    'EPOCHS': 500,
    'EVAL_EVERY': 5,
    'LEARNING_RATE': 0.0001,

    'MAX_QPAIRS': 15,
    'MODEL_NAME': 'ReSaE_transformer',
    'CORRUPTION_POSITIONS': [0, 2],

    # important args
    'SAVE': False,
    'STATEMENT_LEN': -1,
    'USE_TEST': True,
    'LABEL_SMOOTHING': 0.1,
    'SAMPLER_W_QUALIFIERS': True,
    'OPTIMIZER': 'adam',
    'CLEANED_DATASET': True,  # should be false for WikiPeople and JF17K for their original data

    'GRAD_CLIPPING': True,
    'LR_SCHEDULER': True,
    'RANDOMSEED':42,
    'TORCH_SEED':132
}

RESAEARGS = {
    'LAYERS': 2,
    'N_BASES': 0,
    'GCN_DIM': 200,
    'GCN_DROP': 0.1,
    'HID_DROP': 0.3,
    'BIAS': False,
    'OPN': 'other',
    'TRIPLE_QUAL_WEIGHT': 0.8,
    'QUAL_AGGREGATE': 'sum',  # or concat or mul
    'QUAL_OPN': 'other',
    'QUAL_N': 'sum',  # or mean
    'SUBBATCH': 0,
    'QUAL_REPR': 'sparse',  # sparse or full. Warning: full is 10x slower
    'ATTENTION': False,
    'ATTENTION_HEADS': 4,
    'ATTENTION_SLOPE': 0.2,
    'ATTENTION_DROP': 0.1,
    'HID_DROP2': 0.1,
    'MSE': 0.0,
    'KL': 0.0,
    # For ConvE Only
    'FEAT_DROP': 0.3,
    'N_FILTERS': 200,
    'KERNEL_SZ': 7,
    'K_W': 10,
    'K_H': 20,
    'READOUT_DIM':50, #50
    # For Transformer
    'T_LAYERS': 2,
    'T_N_HEADS': 4,
    'T_HIDDEN': 512,
    'POSITIONAL': True,
    'POS_OPTION': 'default',
    'TOKEN_TYPE': 'default',
    'TIME': False,
    'POOLING': 'avg',
    'DECODER': 'transformer', # or GAT
    'READOUT': 'typewise' # or mean
}




def get_config(basic_config, model_config):
    config = basic_config.copy()
    gcnconfig = model_config.copy()
    parsed_args = parse_args(sys.argv[1:])
    print(parsed_args)

    # Superimpose this on default config
    for k, v in parsed_args.items():
        # If its a generic arg
        if k in config.keys():
            default_val = config[k.upper()]
            if default_val is not None:
                needed_type = type(default_val)
                config[k.upper()] = needed_type(v)
            else:
                config[k.upper()] = v
        # If its a ReSaEarg
        elif k.lower().startswith('gcn_') and k[4:] in gcnconfig:
            default_val = gcnconfig[k[4:].upper()]
            if default_val is not None:
                needed_type = type(default_val)
                gcnconfig[k[4:].upper()] = needed_type(v)
            else:
                gcnconfig[k[4:].upper()] = v

        else:
            config[k.upper()] = v

    config['RESAEARGS'] = gcnconfig
    return config
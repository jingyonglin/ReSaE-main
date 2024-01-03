import os

os.environ['MKL_NUM_THREADS'] = '1'
from functools import partial
import random
import sys
import collections
from data_loaders.data_manager import DataManager
from utils.utils import *
from utils.utils_mytorch import mt_save_dir
from loops.evaluation import EvaluationBenchGNNMultiClass, evaluate_pointwise
from loops.evaluation import acc, mrr, mr, hits_at
from models.models_statements import ReSaE_Transformer
from loops.corruption import Corruption
from loops.sampler import MultiClassSampler
from loops.loops import training_loop_gcn
from config import *


if __name__ == "__main__":

    config = get_config(BASIC_CONFIG,RESAEARGS)
    np.random.seed(config['RANDOMSEED'])
    random.seed(config['RANDOMSEED'])
    torch.manual_seed(config['TORCH_SEED'])

    data = DataManager.load(config=config)()

    # Break down the data
    try:
        train_data, valid_data, test_data, n_entities, n_relations, _, _ = data.values()
    except ValueError:
        raise ValueError(f"Honey I broke the loader for {config['DATASET']}")

    config['NUM_ENTITIES'] = n_entities
    config['NUM_RELATIONS'] = n_relations

    # Exclude entities which don't appear in the dataset. E.g. entity nr. 455 may never appear.
    # always off for wikipeople and jf17k
    if config['DATASET'] == 'jf17k' or config['DATASET'] == 'wikipeople':
        config['ENT_POS_FILTERED'] = False

    if config['ENT_POS_FILTERED']:
        ent_excluded_from_corr = DataManager.gather_missing_entities(
            data=train_data + valid_data + test_data,
            positions=config['CORRUPTION_POSITIONS'],
            n_ents=n_entities)
    else:
        ent_excluded_from_corr = [0]

    """
     However, when we want to run a GCN based model, we also work with
            COO representations of triples and qualifiers.

            In this case, for each split: [train, valid, test], we return
            -> edge_index (2 x n) matrix with [subject_ent, object_ent] as each row.
            -> edge_type (n) array with [relation] corresponding to sub, obj above
            -> quals (3 x nQ) matrix where columns represent quals [qr, qv, k] for each k-th edge that has quals

        So here, train_data_gcn will be a dict containing these ndarrays.
    """

    if config['MODEL_NAME'].lower().startswith('ReSaE'):
        # Replace the data with their graph repr formats
        if config['RESAEARGS']['QUAL_REPR'] == 'full':
            if config['USE_TEST']:
                train_data_gcn = DataManager.get_graph_repr(train_data + valid_data, config)
            else:
                train_data_gcn = DataManager.get_graph_repr(train_data, config)
        elif config['RESAEARGS']['QUAL_REPR'] == 'sparse':
            if config['USE_TEST']:
                train_data_gcn = DataManager.get_alternative_graph_repr(train_data + valid_data, config)
            else:
                train_data_gcn = DataManager.get_alternative_graph_repr(train_data, config)
        else:
            print("Supported QUAL_REPR are `full` or `sparse`")
            raise NotImplementedError

        # add reciprocals to the train data
        reci = DataManager.add_reciprocals(train_data, config)
        train_data.extend(reci)
        reci_valid = DataManager.add_reciprocals(valid_data, config)
        valid_data.extend(reci_valid)
        reci_test = DataManager.add_reciprocals(test_data, config)
        test_data.extend(reci_test)
    else:
        train_data_gcn, valid_data_gcn, test_data_gcn = None, None, None

    print(f"Training on {n_entities} entities")

    """
        Make the model.
    """
    config['DEVICE'] = torch.device(config['DEVICE'])
    model = ReSaE_Transformer(train_data_gcn, config)


    model.to(config['DEVICE'])
    print("Model params: ", sum([param.nelement() for param in model.parameters()]))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])
    """
        Prepare test benches.

            When computing train accuracy (`ev_tr_data`), we wish to use all the other data 
                to avoid generating true triples during corruption. 
            Similarly, when computing test accuracy, we index train and valid splits 
                to avoid generating negative triples.
    """
    if config['USE_TEST']:
        ev_vl_data = {'index': combine(train_data, valid_data), 'eval': combine(test_data)}
        ev_tr_data = {'index': combine(valid_data, test_data), 'eval': combine(train_data)}
        tr_data = {'train': combine(train_data, valid_data), 'valid': ev_vl_data['eval']}
    else:
        ev_vl_data = {'index': combine(train_data, test_data), 'eval': combine(valid_data)}
        ev_tr_data = {'index': combine(valid_data, test_data), 'eval': combine(train_data)}
        tr_data = {'train': combine(train_data), 'valid': ev_vl_data['eval']}

    eval_metrics = [acc, mrr, mr, partial(hits_at, k=3),
                    partial(hits_at, k=5), partial(hits_at, k=10)]

    evaluation_valid = None
    evaluation_train = None

    # Saving stuff
    if config['SAVE']:
        savedir = Path(f"./models/{config['DATASET']}/{config['MODEL_NAME']}")
        if not savedir.exists(): savedir.mkdir(parents=True)
        savedir = mt_save_dir(savedir, _newdir=True)
        save_content = {'model': model, 'config': config}
    else:
        savedir, save_content = None, None

    # The args to use if we're training w default stuff
    args = {
        "epochs": config['EPOCHS'],
        "data": tr_data,
        "opt": optimizer,
        "train_fn": model,
        "neg_generator": Corruption(n=n_entities, excluding=[0],
                                    position=list(range(0, config['MAX_QPAIRS'], 2))),
        "device": config['DEVICE'],
        "data_fn": None,
        "eval_fn_trn": evaluate_pointwise,
        "val_testbench": evaluation_valid.run if evaluation_valid else None,
        "trn_testbench": evaluation_train.run if evaluation_train else None,
        "eval_every": config['EVAL_EVERY'],
        "log_wandb": False,
        "run_trn_testbench": False,
        "savedir": savedir,
        "save_content": save_content,
        "qualifier_aware": config['SAMPLER_W_QUALIFIERS'],
        "grad_clipping": config['GRAD_CLIPPING'],
        "scheduler": None
    }

    if config['MODEL_NAME'].lower().startswith('ReSaE'):
        training_loop = training_loop_gcn
        sampler = MultiClassSampler(data=args['data']['train'],
                                    n_entities=config['NUM_ENTITIES'],
                                    lbl_smooth=config['LABEL_SMOOTHING'],
                                    bs=config['BATCH_SIZE'],
                                    with_q=config['SAMPLER_W_QUALIFIERS'])
        evaluation_valid = EvaluationBenchGNNMultiClass(ev_vl_data, model, bs=config['BATCH_SIZE'],
                                                        metrics=eval_metrics,
                                                        filtered=True, n_ents=n_entities,
                                                        excluding_entities=ent_excluded_from_corr,
                                                        positions=config.get('CORRUPTION_POSITIONS', None),
                                                        config=config)
        args['data_fn'] = sampler.reset
        args['val_testbench'] = evaluation_valid.run
        args['trn_testbench'] = None
        if config['LR_SCHEDULER']:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95)
            args['scheduler'] = scheduler

    traces = training_loop(**args)

    with open('traces.pkl', 'wb+') as f:
        pickle.dump(traces, f)






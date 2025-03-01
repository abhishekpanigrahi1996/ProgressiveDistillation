import torch
from torch.utils.data import Dataset
import os
import numpy
from tqdm import tqdm

from utils import prepare_model, get_logits_gpt, accuracy, loss_fn, get_features, boolean_data, set_seed


try:
    import wandb
    USE_WANDB = True
except:
    USE_WANDB = False
    os.environ["WANDB_DISABLED"] = "true"
    print("\n\n\nNot using wandb\n\n\n")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

 
class HierarchicalData(Dataset):
  def __init__(self, data, labels):
      self.inputs = data
      self.labels = labels
      self.n_examples = len(data)
  
  def __getitem__(self, idx):
      return self.inputs[idx], self.labels[idx]

  def __len__(self):
      return self.n_examples


def main(args):
    hidden_size = args.hidden_size
    model_type = args.model_type
    head_dim = args.head_dim
    if model_type == 'gpt2':
       if head_dim != hidden_size // args.n_heads:
          print("Warning: head_dim is not equal to hidden_size // n_heads")
          print(f"head_dim: {head_dim}, hidden_size: {hidden_size}, n_heads: {args.n_heads}")
          print(f"Setting head_dim to be hidden_size // n_heads = {hidden_size // args.n_heads}")
          print("Note -- head_dim is for logging purpose only")
          args.head_dim = hidden_size // args.n_heads
    data_dimension = args.data_dimension
    num_labels = args.num_labels
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    adam_eps = args.adam_eps
    logging_path = args.logging_path
    output_path = args.output_path
    seed = args.seed
    num_layers = args.num_layers 
    
    feature_complexity = args.feature_complexity
    randomize_features = args.randomize_features
    feature_coordinates = args.feature_coordinates
    if feature_coordinates != '':
      feature_coordinates = [int(x) for x in feature_coordinates.split('_')]
    else:
      feature_coordinates = None
    
    n_examples = args.n_examples
    n_steps = args.n_steps
    n_epochs = numpy.ceil(n_steps / n_examples).astype(int)

    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    log_intvl = args.log_intvl
    save_intvl = args.save_intvl

    add_cls_token = args.add_cls_token

    save_suffix = f'_{model_type}_hid{hidden_size}' + f'_n{data_dimension}_k{feature_complexity}' + f'_num_labels{num_labels}' + f'_lr{learning_rate}' + f'warm{args.warmup_ratio}' + f'_wd{weight_decay}' + f'_seed{seed}' + f'_num_layers{num_layers}' + f'_e{args.n_epochs}'
    if 'gpt2' in args.model_type:
        save_suffix += '_nheads' + str(args.n_heads)
    if args.subsample != -1:
        save_suffix += '_subsample' + str(args.subsample)
    if args.tie_word_embeddings == 0:
        save_suffix += '_untied'
    if args.add_cls_token:
        save_suffix += '_clsToken'
    if args.feature_coordinates != '':
        save_suffix += '_feats' + args.feature_coordinates
    if args.skip_causal_mask:
        save_suffix += '_skipCausal'
    logging_dir = logging_path + save_suffix
    output_dir = output_path + save_suffix


    """
    Get data
    """
    set_seed(seed)
    all_features = get_features(num_labels, data_dimension, feature_complexity,
                                random=randomize_features,
                                feature_coordinates=feature_coordinates,)
    all_data, all_y = boolean_data(n_examples, data_dimension, num_labels, all_features)

    if 'gpt2' in args.model_type:
       # convert {-1, 1} to {1, 0}
        all_data = (1 - all_data) // 2
        if add_cls_token:
          cls_tokens = args.vocab_size * numpy.ones((len(all_data), 1))
          args.vocab_size = args.vocab_size + 1
          all_data = numpy.concatenate([cls_tokens, all_data], axis=1)
        all_data = all_data.astype(numpy.int64)
  
    eval_split = num_labels * 125
    train_split = len(all_data) - eval_split
    eval_batch_size = min(1000, eval_split)
    
    train_data, train_y = all_data[:train_split], all_y[:train_split]
    train_dataset = HierarchicalData(train_data, train_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=args.num_workers)
    eval_data, eval_y = all_data[train_split:], all_y[train_split:]
    eval_dataset = HierarchicalData(eval_data, eval_y)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size,
                                              shuffle=False, num_workers=args.num_workers)

    dtype = torch.float32 if args.model_type == 'mlp' else torch.long


    """
    Get model
    """
    all_layers, all_parameters = prepare_model(data_dimension, num_labels,
                                               hidden_size, num_layers,
                                               model_type=args.model_type,
                                               n_heads=args.n_heads,
                                               vocab_size=args.vocab_size,
                                               tie_word_embeddings=args.tie_word_embeddings,
                                               linear_mlp=args.linear_mlp, skip_mlp=args.skip_mlp,
                                               skip_causal_mask=args.skip_causal_mask,)

    if 'gpt2' in args.model_type:
      optimizer = torch.optim.Adam(all_parameters, lr=learning_rate, weight_decay=weight_decay, eps=adam_eps)
    else:
      optimizer = torch.optim.SGD(all_parameters, lr=learning_rate, weight_decay=weight_decay)


    if USE_WANDB:
        # TODO: set your wandb project name & entity here.
        project = ''
        entity = ''
        wandb.init(project=project, name=logging_dir, config=args, entity=entity)
        wandb.config.update({
            'output_dir': output_dir,
            'task': 'classification',
            "pid": os.getpid(),
        })

    wandb.watch(all_layers, log='all')
    

    global_step_cnt = 0
    eval_accs = []
    for epoch in tqdm(range(n_epochs)):
        print("Current Epoch:", epoch)
        for bt, batch in tqdm(enumerate(train_loader)):
            batch_data, batch_y = batch

            cuda_batch_data, cuda_batch_y = batch_data.to(DEVICE).type(dtype), batch_y.to(DEVICE).long()
            
            if args.model_type == 'mlp':
                predicted_y = all_layers(cuda_batch_data)
            elif 'gpt' in args.model_type:
              predicted_y = get_logits_gpt(cuda_batch_data, all_layers)

            loss = loss_fn(predicted_y, cuda_batch_y)
            loss.backward()

            # set learning rate
            curr_learning_rate = 0
            if global_step_cnt < n_steps * args.warmup_ratio:
                # linear warmup
                curr_learning_rate = learning_rate * (global_step_cnt+1) / (n_steps * args.warmup_ratio)
            else:
                if args.anneal_type == 'cosine':
                    curr_learning_rate = learning_rate * 0.5 * (1 + numpy.cos(numpy.pi * (global_step_cnt - n_steps * args.warmup_ratio) / (n_steps * (1-args.warmup_ratio))))
                elif args.anneal_type == 'linear':
                    curr_learning_rate = learning_rate * (1 - (global_step_cnt - n_steps * args.warmup_ratio) / (n_steps * (1-args.warmup_ratio)))
                elif args.anneal_type == 'constant':
                    curr_learning_rate = learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_learning_rate 


            if global_step_cnt % 5000 == 0:
               print(f"Loss at step {global_step_cnt}: {loss.item()}")

            if global_step_cnt % log_intvl == 0: 
                if USE_WANDB:
                    wandb_results = {'Train Loss': loss.item()}
                    wandb_results['Learning Rate'] = curr_learning_rate
                    wandb.log(wandb_results, step=global_step_cnt)
                    
            optimizer.step()
            optimizer.zero_grad()

            if global_step_cnt % min(log_intvl, len(train_loader)) == 0:
                eval_loss = 0.
                predicted_y_gap = 0
                for bt, batch in tqdm(enumerate(eval_loader)):
                    batch_data, batch_y = batch
                    cuda_batch_data, cuda_batch_y = batch_data.to(DEVICE).type(dtype), batch_y.to(DEVICE).long()
                    with torch.no_grad():
                        if args.model_type == 'mlp':
                            eval_logits = all_layers(cuda_batch_data)
                        elif 'gpt' in args.model_type:
                          eval_logits= get_logits_gpt(cuda_batch_data, all_layers)
                        predicted_y = torch.nn.functional.softmax(eval_logits, dim=-1)
                        predicted_y_gap += (predicted_y[:, 0] -  predicted_y[:, 1]).mean().item()
                        loss = accuracy(predicted_y, cuda_batch_y)
                        eval_loss += loss.item()
                eval_acc = eval_loss / len(eval_loader)
                eval_accs += eval_acc,
                predicted_y_gap /= len(eval_loader)
                if USE_WANDB:
                    wandb.log({'Eval Acc': eval_acc, 'y_gap': predicted_y_gap}, 
                              step=global_step_cnt)
                
                # check for early stopping
                if len(eval_accs) > 10 and sum(eval_accs[-10:])/10 > 0.9999 and 0:
                    print("Early Stopping at:", global_step_cnt)
                    return
                
            if global_step_cnt % save_intvl == 0:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                print("Save at:", global_step_cnt)
                fout = output_dir + '/checkpoint-'+str(global_step_cnt)
                if args.model_type == 'mlp':
                  if type(all_layers) == list:
                    model_state_dict = [layer.state_dict() for layer in all_layers]
                  else:
                    model_state_dict = all_layers.state_dict()
                  try:
                    torch.save(
                        {
                        'epoch': epoch, 
                        'step': global_step_cnt,
                        'model_state_dict': model_state_dict,
                        'optimizer_state_dict': optimizer.state_dict()
                      },
                      fout)
                  except:
                     print("Failed to save model")

                elif 'gpt' in args.model_type:
                  all_layers.save_pretrained(fout)

            global_step_cnt += 1



if __name__ == "__main__":
    # build an argument parser
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--model_type', type=str, default='mlp',
                        choices=['mlp', 'gpt2', 'gpt2_avg_head', 'gpt2_log_scale'])
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--head_dim', type=int, default=8)
    ## GPT specific
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--vocab_size', type=int, default=50257)
    parser.add_argument('--skip_causal_mask', type=int, default=0)
    ###
    parser.add_argument('--data_dimension', type=int)
    parser.add_argument('--use_cls_head', type=int, default=0)
    parser.add_argument('--add_cls_token', type=int, default=0)
    parser.add_argument('--tie_word_embeddings', type=int, default=1)
    parser.add_argument('--linear_mlp', type=int, default=0)
    parser.add_argument('--skip_mlp', type=int, default=0)
    parser.add_argument('--num_labels', type=int)
    parser.add_argument('--feature_complexity', type=int)
    parser.add_argument('--randomize_features', type=int)
    parser.add_argument('--feature_coordinates', type=str, default='')
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--anneal_type', type=str, default='cosine')
    parser.add_argument('--warmup_ratio', type=float, default=0.06)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--logging_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--n_steps', type=int)
    parser.add_argument('--n_examples', type=int)
    parser.add_argument('--subsample', type=int, default=-1)
    parser.add_argument('--fdata_train', type=str)
    parser.add_argument('--fdata_eval', type=str)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--eval_batch_size', type=int)
    parser.add_argument('--log_intvl', type=int, default=50000)
    parser.add_argument('--save_intvl', type=int, default=100000)
    parser.add_argument("--pcfg_name", type=str, default='')
    args = parser.parse_args()
    
    main(args)







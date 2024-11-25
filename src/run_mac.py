import torch
import pandas as pd
import os
from src.utils import read_table, cost, auc_score
from src.env import EDCopilotEnv
from sb3_contrib.ppo_mask import MaskablePPO
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
import pickle

def train(args):
    policy_kwargs = {
        "model_name_path": args.output_path,
        "weight_decay": args.weight_decay,
        "optimizer_class": torch.optim.AdamW,
        "optimizer_kwargs": {
            "eps": args.adam_epsilon
        }
    }  
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    df_train = pd.read_csv(os.path.join(args.data_input_path, 'train.csv'))
    df_valid = pd.read_csv(os.path.join(args.data_input_path, 'valid.csv'))
    header, train_table, train_label = read_table(df_train, args.outcome)
    header, valid_table, valid_label = read_table(df_valid, args.outcome)
    train_env = EDCopilotEnv((train_table, train_label), header, args, True)
    valid_env = EDCopilotEnv((valid_table, valid_label), header, args)

    model = MaskablePPO(
        MaskableLMActorCriticPolicy,
        train_env,
        learning_rate=args.learning_rate,
        n_steps=args.buffer_steps,
        n_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log="run",
        device=device,
        max_grad_norm=args.max_grad_norm
    )
    
    ckpt_path = os.path.join(args.output_path, f'{args.penalty_ratio}_{args.wrong_prediction_penalty}')
    os.makedirs(ckpt_path, exist_ok=True)

    best_f1 = 0
    for _ in tqdm(range(args.epochs), desc="Training"):
        model.learn(
            args.total_timesteps,
            tb_log_name=f"lab_{args.penalty_ratio}_{args.wrong_prediction_penalty}_{args.outcome}",
            progress_bar=True
        )
        results = validate(model, valid_env)
        for key, value in results.items():
            print(f"{key:25}: {value}")
        if best_f1 < results["F1"]:
            best_f1 = results["F1"]
            model.policy.save(ckpt_path)
            print(f"Model saved to {ckpt_path}")

    test(args)

@torch.no_grad()
def validate(model, env):
    model.policy.set_training_mode(False)
    max_episodes = env.num_patients

    cumulative_reward = 0
    n_tested_ave = 0
    cost_tested_ave = 0
    n_healthy = 0
    n_ill = 0
    n_healthy_acc_predict = 0
    n_ill_acc_predict = 0

    predict_probs = np.zeros(max_episodes)
    true_labels = np.zeros(max_episodes)
    pred_labels = np.zeros(max_episodes)
    action_dist = {}
    total_action_preds = []
    total_action_targets = []

    for episode in tqdm(range(max_episodes), desc="Evaluating"):
        obs = env.reset()
        done = False
        true_labels[episode] = env.current_patient_y
        total_action_targets.append(env.current_patient_lab_y)
        
        action_preds = []
        if not env.current_patient_y:
            n_healthy += 1
        else:
            n_ill += 1

        while not done:
            action_masks = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)

            obs, reward, done, info = env.step(action)
            cumulative_reward += reward
            if action < env.num_lab:
                n_tested_ave += 1

                cost_pair = list(cost.items())[action]
                feature_name = cost_pair[0]
                cost_tested_ave += cost_pair[1]
                                     
                action_dist[feature_name] = action_dist.get(feature_name, 0) + 1
                
                action_preds.append(int(action))
            else:
                obs_tensor = model.policy.obs_to_tensor(obs)[0]
                action_list = torch.tensor(list(range(env.num_lab + 2)), device=obs_tensor.device)
                prob = torch.exp(model.policy.evaluate_actions(obs_tensor, action_list, action_masks)[1])[-2:]
                predict_probs[episode] = (prob[-1] / torch.sum(prob)).item()

                pred_labels[episode] = (action == (env.num_lab + 1))
                
                if not env.current_patient_y:
                    n_healthy_acc_predict += (action == env.num_lab)
                else:
                    n_ill_acc_predict += (action == (1 + env.num_lab))
                    
                total_action_preds.append(action_preds)

    f1 = f1_score(true_labels, pred_labels)
    roc_auc, average_precision, sensitivity, specificity, threshold = auc_score(true_labels, predict_probs)
    results = {
        "F1": f1,
        "AUC": roc_auc,
        "AUPRC": average_precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Threshold": threshold,
        "Cumulative reward": cumulative_reward / max_episodes,
        "Cost average": cost_tested_ave / max_episodes,
        "Average number of test": n_tested_ave / max_episodes,
    }
    return results

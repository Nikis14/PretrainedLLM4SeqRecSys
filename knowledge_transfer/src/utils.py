# src/utils.py

import random
import numpy as np
import torch
import json

from torch import nn


def set_seed(seed):
    """Устанавливает зерно для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_user_profile_embeddings(file_path, user_id_mapping):
    """
    Старый метод: загружает эмбеддинги профилей пользователей из ОДНОГО JSON-файла,
    результат: [num_users, emb_dim], [num_users]
    """
    with open(file_path, 'r') as f:
        user_profiles_data = json.load(f)

    embedding_dim = len(next(iter(user_profiles_data.values())))
    # num_users = len(user_id_mapping)
    max_idx = max(user_id_mapping.values()) + 1  # Ensure list can accommodate the highest index
    print('Seq Stats:', max_idx, len(user_id_mapping))
    user_profiles_list = [[0.0] * embedding_dim for _ in range(max_idx)]
    null_profile_binary_mask = [False for _ in range(max_idx)]

    not_found_profiles_cnt = 0
    for original_id, idx in user_id_mapping.items():
        embedding = user_profiles_data.get(str(original_id))
        if embedding is not None:
            user_profiles_list[idx] = embedding
        else:
            # Если эмбеддинг не найден, инициализируем нулями
            # user_profiles_list[idx] = [0.0] * embedding_dim
            null_profile_binary_mask[idx] = True
    print(f"Number of users without profiles: {not_found_profiles_cnt}")

    user_profiles_tensor = torch.tensor(user_profiles_list, dtype=torch.float)
    null_profile_binary_mask_tensor = torch.BoolTensor(null_profile_binary_mask)
    return user_profiles_tensor, null_profile_binary_mask_tensor


def load_user_profiles_multi(files_list, user_id_mapping):
    """
    Новый метод: загружает несколько JSON-файлов => [num_users, K, emb_dim], [num_users, K]
    """
    all_tensors = []
    all_masks = []

    for file_path in files_list:
        with open(file_path, 'r') as f:
            user_profiles_data = json.load(f)

        embedding_dim = len(next(iter(user_profiles_data.values())))
        num_users = len(user_id_mapping)

        user_profiles_list = [None] * num_users
        null_profile_binary_mask = [False] * num_users

        for original_id, idx in user_id_mapping.items():
            emb = user_profiles_data.get(str(original_id))
            if emb is not None:
                user_profiles_list[idx] = emb
            else:
                user_profiles_list[idx] = [0.0] * embedding_dim
                null_profile_binary_mask[idx] = True

        user_profiles_tensor = torch.tensor(user_profiles_list, dtype=torch.float32)
        null_mask_tensor = torch.tensor(null_profile_binary_mask, dtype=torch.bool)

        all_tensors.append(user_profiles_tensor)
        all_masks.append(null_mask_tensor)

    # stack => [num_users, K, emb_dim], [num_users, K]
    user_profiles_tensor_3d = torch.stack(all_tensors, dim=1)
    null_profile_binary_mask_2d = torch.stack(all_masks, dim=1)

    # check that all profiles either exist or not
    inconsistent_profiles = ~(~torch.any(null_profile_binary_mask_2d, dim=1) | torch.all(null_profile_binary_mask_2d, dim=1))
    null_profile_binary_mask_2d[inconsistent_profiles] = False  # set False for all profiles that are inconsistent
    print(f'Number of inconcsistent profiles: {inconsistent_profiles.sum().item()}')

    null_profile_binary_mask_1d = null_profile_binary_mask_2d[:, 0]

    return user_profiles_tensor_3d, null_profile_binary_mask_1d


def load_user_profile_embeddings_any(config, user_id_mapping):
    """
    Универсальная функция:
    Если config['model']['multi_profile'] == True => грузим список файлов
    Иначе => грузим одиночный файл.
    """
    multi_profile = config['model'].get('multi_profile', False)
    files_list = config['data']['user_profile_embeddings_files']
    if multi_profile:
        return load_user_profiles_multi(files_list, user_id_mapping)
    if isinstance(files_list, str):
        files_list = [files_list]
    return load_user_profile_embeddings(files_list[0], user_id_mapping)

    # if multi_profile:
    #     files_list = config['data']['user_profile_embeddings_files']
    #     return load_user_profiles_multi(files_list, user_id_mapping)
    # else:
    #     path = config['data']['user_profile_embeddings_path']
    #     return load_user_profile_embeddings(path, user_id_mapping)


def init_criterion_reconstruct(criterion_name):
    if criterion_name == 'MSE':
        return lambda x,y: nn.MSELoss()(x,y)
    if criterion_name == 'RMSE':
        return lambda x,y: torch.sqrt(nn.MSELoss()(x,y))
    if criterion_name == 'CosSim':
        return lambda x,y: 1 - torch.mean(nn.CosineSimilarity(dim=1, eps=1e-6)(x,y))
    raise Exception('Not existing reconstruction loss')


def calculate_recsys_loss(target_seq, outputs, criterion):
    # Проверяем, если outputs является кортежем (на всякий случай)
    if isinstance(outputs, tuple):
        outputs = outputs[0]

    logits = outputs.view(-1, outputs.size(-1))
    targets = target_seq.view(-1)

    loss = criterion(logits, targets)
    return loss


def calculate_guide_loss(model,
                         user_profile_emb,
                         hidden_for_reconstruction,
                         null_profile_binary_mask_batch,
                         criterion_reconstruct_fn):
    # if model.use_down_scale:
    #     user_profile_emb_transformed = model.profile_transform(user_profile_emb)
    # else:
    #     user_profile_emb_transformed = user_profile_emb.detach().clone().to(device)
    # if model.use_upscale:
    #     hidden_for_reconstruction = model.hidden_layer_transform(hidden_for_reconstruction)
    # user_profile_emb_transformed[null_profile_binary_mask_batch] = hidden_for_reconstruction[
    #     null_profile_binary_mask_batch]
    #
    # loss_guide = criterion_reconstruct_fn(hidden_for_reconstruction, user_profile_emb_transformed)
    # return loss_guide

    # pass
    user_profile_emb_transformed = model.aggregate_profile(user_profile_emb)
    try:
        if model.use_upscale:
            hidden_for_reconstruction = model.hidden_layer_transform(hidden_for_reconstruction)
    except:
        pass

    user_profile_emb_transformed[null_profile_binary_mask_batch] = \
        hidden_for_reconstruction[null_profile_binary_mask_batch]

    loss_guide = criterion_reconstruct_fn(hidden_for_reconstruction, user_profile_emb_transformed)
    return loss_guide

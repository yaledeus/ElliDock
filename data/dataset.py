#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import pickle
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from .bio_parse import Complex
import sys
sys.path.append('..')
from utils.logger import print_log


# use this class to splice the dataset and maintain only one part of it in RAM
# SabDab Antibody-Antigen Complex dataset
class SabDabDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, save_dir=None, num_entry_per_file=-1, random=False):
        '''
        file_path: path to the dataset
        save_dir: directory to save the processed data
        num_entry_per_file: number of entries in a single file. -1 to save all data into one file
                            (In-memory dataset)
        '''
        super().__init__()
        if save_dir is None:
            if not os.path.isdir(file_path):
                save_dir = os.path.split(file_path)[0]
            else:
                save_dir = file_path
            prefix = os.path.split(file_path)[1]
            if '.' in prefix:
                prefix = prefix.split('.')[0]
            save_dir = os.path.join(save_dir, f'{prefix}_processed')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        metainfo_file = os.path.join(save_dir, '_metainfo')
        self.data: List[Complex] = []

        # try loading preprocessed files
        need_process = False
        try:
            with open(metainfo_file, 'r') as fin:
                metainfo = json.load(fin)
                self.num_entry = metainfo['num_entry']
                self.file_names = metainfo['file_names']
                self.file_num_entries = metainfo['file_num_entries']
        except Exception as e:
            print_log(f'Faild to load file {metainfo_file}, error: {e}', level='WARN')
            need_process = True

        if need_process:
            # preprocess
            self.file_names, self.file_num_entries = [], []
            self.preprocess(file_path, save_dir, num_entry_per_file)
            self.num_entry = sum(self.file_num_entries)

            metainfo = {
                'num_entry': self.num_entry,
                'file_names': self.file_names,
                'file_num_entries': self.file_num_entries
            }
            with open(metainfo_file, 'w') as fout:
                json.dump(metainfo, fout)

        self.random = random
        self.cur_file_idx, self.cur_idx_range = 0, (0, self.file_num_entries[0])  # left close, right open
        self._load_part()

        # user defined variables
        self.idx_mapping = [i for i in range(self.num_entry)]
        self.mode = '1*1'  # H/L/Antigen, 1 for include, 0 for exclude, * for either

    def _save_part(self, save_dir, num_entry):
        file_name = os.path.join(save_dir, f'part_{len(self.file_names)}.pkl')
        print_log(f'Saving {file_name} ...')
        file_name = os.path.abspath(file_name)
        if num_entry == -1:
            end = len(self.data)
        else:
            end = min(num_entry, len(self.data))
        with open(file_name, 'wb') as fout:
            pickle.dump(self.data[:end], fout)
        self.file_names.append(file_name)
        self.file_num_entries.append(end)
        self.data = self.data[end:]

    def _load_part(self):
        f = self.file_names[self.cur_file_idx]
        print_log(f'Loading preprocessed file {f}, {self.cur_file_idx + 1}/{len(self.file_names)}')
        with open(f, 'rb') as fin:
            del self.data
            self.data = pickle.load(fin)
        self.access_idx = [i for i in range(len(self.data))]
        if self.random:
            np.random.shuffle(self.access_idx)

    def _check_load_part(self, idx):
        if idx < self.cur_idx_range[0]:
            while idx < self.cur_idx_range[0]:
                end = self.cur_idx_range[0]
                self.cur_file_idx -= 1
                start = end - self.file_num_entries[self.cur_file_idx]
                self.cur_idx_range = (start, end)
            self._load_part()
        elif idx >= self.cur_idx_range[1]:
            while idx >= self.cur_idx_range[1]:
                start = self.cur_idx_range[1]
                self.cur_file_idx += 1
                end = start + self.file_num_entries[self.cur_file_idx]
                self.cur_idx_range = (start, end)
            self._load_part()
        idx = self.access_idx[idx - self.cur_idx_range[0]]
        return idx

    def __len__(self):
        return self.num_entry

    ########### load data from file_path and add to self.data ##########
    def preprocess(self, file_path, save_dir, num_entry_per_file):
        '''
        Load data from file_path and add processed data entries to self.data.
        Remember to call self._save_data(num_entry_per_file) to control the number
        of items in self.data (this function will save the first num_entry_per_file
        data and release them from self.data) e.g. call it when len(self.data) reaches
        num_entry_per_file.
        '''
        with open(file_path, 'r') as fin:
            lines = fin.read().strip().split('\n')
        for line in tqdm(lines):
            item = json.loads(line)
            try:
                complex = Complex.from_pdb(
                    item['pdb_data_path'], item['heavy_chain'], item['light_chain'],
                    item['antigen_chains']
                )
            except AssertionError as e:
                print_log(e, level='ERROR')
                print_log(f'parse {item["pdb"]} pdb failed, skip', level='ERROR')
                continue

            self.data.append(complex)
            if num_entry_per_file > 0 and len(self.data) >= num_entry_per_file:
                self._save_part(save_dir, num_entry_per_file)
        if len(self.data):
            self._save_part(save_dir, num_entry_per_file)

    ########## override get item ##########
    def __getitem__(self, idx):
        idx = self.idx_mapping[idx]
        idx = self._check_load_part(idx)
        item = self.data[idx]

        try:
            # antibody
            ab_seq = item.antibody_seq()
            ab_bb_coord = item.antibody_coord()     # backbone atoms, [N_ab, 4, 3]
            ab_rp = item.antibody_relative_pos()
            ab_id = item.antibody_identify()

            # antigen
            ag_seq = item.antigen_seq()
            ag_bb_coord = item.antigen_coord()      # backbone atoms, [N_ag, 4, 3]
            ag_rp = item.antigen_relative_pos()
            ag_id = item.antigen_identity()

            # center
            center = np.mean(ag_bb_coord.reshape(-1, 3), axis=0)

            # keypoint pairs
            keypoints = item.find_keypoint(threshold=10)

            assert ab_bb_coord.ndim == 3, f'invalid antibody coordinate dimension: {ab_bb_coord.ndim}'
            assert ag_bb_coord.ndim == 3, f'invalid antigen coordinate dimension: {ag_bb_coord.ndim}'
            assert len(ab_seq) == len(ab_rp) and len(ab_seq) == len(ab_id) and len(ab_seq) == ab_bb_coord.shape[0], \
                'antibody seq/coord/rp/id dimension mismatch'
            assert len(ag_seq) == len(ag_rp) and len(ag_seq) == len(ag_id) and len(ag_seq) == ag_bb_coord.shape[0], \
                'antigen seq/coord/rp/id dimension mismatch'
            assert ab_bb_coord.shape[1] == ag_bb_coord.shape[1] and ab_bb_coord.shape[-1] == ag_bb_coord.shape[-1], \
                'antibody and antigen coordinates mismatch'
            assert keypoints.shape[0] >= 1, 'keypoints not found'

            data = {
                'S': np.array(ab_seq + ag_seq),
                'X': np.concatenate((ab_bb_coord, ag_bb_coord), axis=0),
                'RP': np.array(ab_rp + ag_rp),
                'ID': np.array(ab_id + ag_id),
                ### segment, 0 for antibody and 1 for antigen
                'Seg': np.array([0 for _ in range(len(ab_seq))] + [1 for _ in range(len(ag_seq))]),
                'center': center,
                'keypoints': keypoints,
            }
        except Exception as e:
            print_log(f"{e}", level='ERROR')
            data = {
                'Error': True
            }


        return data

    @classmethod
    def collate_fn(cls, batch):
        keys = ['X', 'S', 'RP', 'ID', 'Seg']
        types = [torch.float] + [torch.long for _ in range(4)]
        res = {}
        # in case batch is null
        blank_batch = True
        for item in batch:
            if not 'Error' in item:
                blank_batch = False
                break
        if blank_batch:
            return res
        # collate batch elements
        for key, _type in zip(keys, types):
            val = []
            for item in batch:
                if 'Error' in item:
                    continue
                val.append(torch.tensor(item[key], dtype=_type))
            res[key] = torch.cat(val, dim=0)
        centers, keypoints, bid, k_bid = [], [], [], []
        i = 0
        for item in batch:
            if 'Error' in item:
                continue
            centers.append(item['center'])
            keypoints.append(torch.tensor(item['keypoints']))
            bid.extend([i] * len(item['S']))
            k_bid.extend([i] * len(item['keypoints']))
            i += 1
        res['center'] = torch.tensor(np.array(centers), dtype=torch.float)
        res['keypoints'] = torch.cat(keypoints, dim=0)
        res['bid'] = torch.tensor(bid, dtype=torch.long)
        res['k_bid'] = torch.tensor(k_bid, dtype=torch.long)
        return res

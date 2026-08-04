## IMPORTS

import pandas as pd
import os
import numpy as np
import tqdm
import torch
import rsa_utils

## FUNCTIONS
def load_correct_form_standard(data_loc='data'):
    """ Loads the correct form for standard RSA experiment"""
    df = pd.read_excel('{}/correct_form_standard_v2.xlsx'.format(data_loc))
    return list(zip(df.verb_match.tolist(), df.noun_match.tolist()))

def load_correct_form_standard_and(data_loc='data'):
    """ Loads the correct form for standard RSA experiment with 'and'"""
    df = pd.read_excel('{}/correct_form_standard_and_v2.xlsx'.format(data_loc))
    return list(zip(df.noun_match.tolist(), df.and_match.tolist()))


def get_standard_sentences(data_loc='data'):
    """ Loads the sentences for standard RSA experiment"""
    df = pd.read_excel('{}/standard_sentences_v2.xlsx'.format(data_loc))
    sentences = np.array(df['sentence'].tolist())
    return sentences

def get_idiom_modifier_head_words_per_sentence_standard(data_loc='data'):
    """ Loads the idiom noun and verb words for standard RSA experiment"""
    df = pd.read_excel('{}/standard_sentences_v2.xlsx'.format(data_loc))
    mod_head_tuples_per_sentence = np.array(list(zip(df['verb'].tolist(), df['noun'].tolist())))
    return mod_head_tuples_per_sentence

def get_idiom_modifier_head_words_per_sentence_standard_and(data_loc='data'):
    """ Loads the idiom noun and 'and' words for standard RSA experiment with """
    df = pd.read_excel('{}/standard_sentences_v2.xlsx'.format(data_loc))
    mod_head_tuples_per_sentence = np.array(list(zip(df['noun'].tolist(), df['and'].tolist())))
    return mod_head_tuples_per_sentence



def load_correct_form_context(data_loc='data'):
    """ Loads the correct form for figurative context RSA experiment"""
    df = pd.read_excel('{}/correct_form_context_v2.xlsx'.format(data_loc))
    return list(zip(df.verb_match.tolist(), df.noun_match.tolist()))

def load_correct_form_context_and(data_loc='data'):
    """ Loads the correct form for figurative context RSA experiment with 'and'"""
    df = pd.read_excel('{}/correct_form_context_v2.xlsx'.format(data_loc))
    return list(zip(df.noun_match.tolist(), df.and_match.tolist()))

def get_context_sentences(data_loc='data'):
    """ Loads the sentences for figurative context RSA experiment"""
    df = pd.read_excel('{}/standard_sentences_context_v2.xlsx'.format(data_loc))
    sentences = np.array(df['sentence'].tolist())
    return sentences

def get_idiom_modifier_head_words_per_sentence_context(data_loc='data'):
    """ Loads the idiom noun and verb words for figurative context RSA experiment"""
    df = pd.read_excel('{}/standard_sentences_context_v2.xlsx'.format(data_loc))
    mod_head_tuples_per_sentence = np.array(list(zip(df['verb'].tolist(), df['noun'].tolist())))
    return mod_head_tuples_per_sentence

def get_idiom_modifier_head_words_per_sentence_context_and(data_loc='data'):
    """ Loads the idiom noun and 'and' words for figurative context RSA experiment"""
    df = pd.read_excel('{}/standard_sentences_context_v2.xlsx'.format(data_loc))
    mod_head_tuples_per_sentence = np.array(list(zip(df['noun'].tolist(), df['and'].tolist())))
    return mod_head_tuples_per_sentence




def load_correct_form_no_context(data_loc='data'):
    """ Loads the correct form for literal context RSA experiment"""
    df = pd.read_excel('{}/correct_form_no_context_v2.xlsx'.format(data_loc))
    return list(zip(df.verb_match.tolist(), df.noun_match.tolist()))

def load_correct_form_no_context_and(data_loc='data'):
    """ Loads the correct form for literal context RSA experiment with 'and'"""
    df = pd.read_excel('{}/correct_form_no_context_v2.xlsx'.format(data_loc))
    return list(zip(df.noun_match.tolist(), df.and_match.tolist()))
    
def get_no_context_sentences(data_loc='data'):
    """ Loads the sentences for literal context RSA experiment"""
    df = pd.read_excel('{}/standard_sentences_no_context_v2.xlsx'.format(data_loc))
    sentences = np.array(df['sentence'].tolist())
    return sentences

def get_idiom_modifier_head_words_per_sentence_no_context(data_loc='data'):
    """ Loads the idiom noun and verb words for literal context RSA experiment"""
    df = pd.read_excel('{}/standard_sentences_no_context_v2.xlsx'.format(data_loc))
    mod_head_tuples_per_sentence = np.array(list(zip(df['verb'].tolist(), df['noun'].tolist())))
    return mod_head_tuples_per_sentence

def get_idiom_modifier_head_words_per_sentence_no_context_and(data_loc='data'):
    """ Loads the idiom noun and 'and' words for literal context RSA experiment"""
    df = pd.read_excel('{}/standard_sentences_no_context_v2.xlsx'.format(data_loc))
    mod_head_tuples_per_sentence = np.array(list(zip(df['noun'].tolist(), df['and'].tolist())))
    return mod_head_tuples_per_sentence




def get_hidden_state_file(model_name, layer=11, rep_type='sentence_pair_cls', data_loc='data'):
    """ Gets the file path for each model to save the representations """
    hidden_state_folder = '{}/representations/{}/layer_{}/{}'.format(data_loc, model_name.split('-')[0], layer, rep_type)
    return '{}/{}_layer_{}_{}.npy'.format(hidden_state_folder, model_name, layer, rep_type)



def select_within_compound_groups(rdm, group_i):
    """ Selects the within idiom groups for the RDM"""
    to_keep_inds = []
    
    get_lower = lambda x: x[np.where(np.triu(np.ones(x.shape[:1])) == 0)]

    for start in range(0, 480, 8):  # 60 groups of 8
        block_inds = [[(i, j) for i in range(start, start + 8)] for j in range(start, start + 8)]
        to_keep_inds.append(get_lower(np.array(block_inds)))

    return np.array([rdm[i[0]][i[1]] for i in to_keep_inds[group_i]])

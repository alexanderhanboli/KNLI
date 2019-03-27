from __future__ import  print_function, division
import os
import json
import argparse
import sys
import ntpath
import glob

parser = argparse.ArgumentParser()
# Input data
parser.add_argument('--check_point_dir', default='/home/ubuntu/projects/KNLI/check_points/')
parser.add_argument('--model_name',  default='')
parser.add_argument('--cp',  default=False, action='store_true')
# Test data
parser.add_argument('--ftest', default='snli_data.json')

def get_fname_from_path(f):
    '''
     input:
           '/Users/user/logs/check_points/mmmxm_dummy_B32_H5_D1_best.pt'
     output:
           'mmmxm_dummy_B32_H5_D1_best.pt'
    '''
    return ntpath.basename(f)

def get_list_dir(path, ext = '.json'):
    '''
     This function returns the list dir
    '''
    return [f for f in os.listdir(path) if f.endswith(ext) ]

def read_json_file(input_json):
    ## load the json file
    try:
        file_info = json.load(open(input_json, 'r'))

    except ValueError:
        print('An error occured.%s'% input_json)
        file_info = []

    return file_info

def find_best_model(check_point_dir, finput):

  best_val = -1
  best_recall = -1
  best_precision = -1
  for i in finput:
    vals = read_json_file(os.path.join(check_point_dir,i))
    curr_v = vals['best_val_accuracy']
    if curr_v > best_val:
          best_val = curr_v
          best_args = vals['args']
          best_recall = vals['best_recall']
          best_precision = vals['best_precision']
          best_file = i
  return best_precision, best_recall, best_val, best_args, os.path.join(check_point_dir,best_file.replace('json', 'pt'))

def move_to_mfiles(mfname_best):
    '''
       copy model and json and log file to the mfiles
    '''
    jfname_best = mfname_best.replace('.pt', '.json')
    jfname = jfname_best.replace('_best.json', '.json')
    mfname = mfname_best.replace('_best.pt', '.pt')

    print('Move best models to mfiles/ ...')
    cmd1 = 'cp ' + mfname_best + ' ./mfiles/' + get_fname_from_path(mfname_best)
    print('model --> ' + mfname_best)
    os.system(cmd1)

if __name__ == "__main__":

    args = parser.parse_args()
    print(args.__dict__)

    # list all the file  in a directory
    jsonFiles = get_list_dir(args.check_point_dir)

    ####################
    # First group the models
    ####################
    list_models = {}
    for j in jsonFiles:
        # the files has the following format
        # emrssiamese_npp_EMR_BETA_0656416_S2_B32_H1624_D6_best.json
        if (j.split('_')[-1]) != 'best.json':
            continue

        model_type = j.split('_')[0]
        model_description = j.split('_')[1]

        if args.model_name != '' and model_type != args.model_name:
            continue

        if model_type not in list_models:
            list_models[model_type] = []

        list_models[model_type].append(os.path.join(args.check_point_dir,j))

    ####################
    # Find the best model per groups
    ####################
    print("-----------------------------------")
    print("-----------------------------------")
    #print("Cosine" )
    #command = "python3.4 -u test_in_stream.py --fp_test " + args.ftest + " --fp_embd ~/wiki.en.bin "
    #print(command)
    print("-----------------------------------")
    cp_list = []

    for gid, model in enumerate(list_models.keys()):
        pre, recall, acc, bargs ,ptf = find_best_model(args.check_point_dir, list_models[model])
        print(" %s ==> precison: %.4f, recall: %.4f, accuracy: %.4f, file %s" %(model, pre, recall, acc, ptf))
        print("-----------------------------------")
        print(" the best args are: {}".format(bargs))
        gcmd = 'CUDA_VISIBLE_DEVICES=%d  ' % (gid, )
        command = "python -u test.py --fp_test " + args.ftest + " --best_model "+ ptf  + " --split test --batch_size 16"
        print(gcmd + command)
        print("-----------------------------------")
        cp_list.append(ptf)

    if args.cp == True:
      print("-----------------------------------")
      for i in cp_list:
          move_to_mfiles(i)
          print('----------------')

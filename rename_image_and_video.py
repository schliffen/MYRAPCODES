#
# automatic Standard data labelling - for Rapsodo
#

import glob, sys, os
import argparse
import logging
dir = os.getcwd()

# defining required directories
parser = argparse.ArgumentParser(description='Short sample app')

parser.add_argument('-a', '--data_dir',  type=str, help="defining the root", default="/home/yizhuo/Documents/s1/HomeCourt_Data_816_tmp/")
parser.add_argument('-a1', '--dest_dir',  type=str, help="defining the root", default="/home/yizhuo/Documents/")
parser.add_argument('-n1', "--name_prj", type=str, help="project name prefix for the name", default="ba")
parser.add_argument('-n2', "--name_dt",  type=str, help="date prefix for the name video", default="2018")
parser.add_argument('-n3', "--name_prp", type=str, help="purpose prefix for the name video", default="hc-outdoor-practice")
parser.add_argument('-n4', "--name_src", type=str, help="source prefix for the name video", default="hc")
parser.add_argument('-n5', "--name_loc", type=str, help="location prefix for the name video", default="sgsg")
parser.add_argument('-n6', "--dic_key", type=str, help="the list of the video (the folder)"
                                                       "this should provided as list", default="")
parser.add_argument('-n7', "--dic_val", type=str, help="the list of date of the video creation yearmmddhhmm"
                                                       "this should provided as list", default="")
parser.add_argument('-f1', "--frame_name", type=int, help=" prefix for the name the frames", default=0000)

arg = parser.parse_args()

# Creating dictionary for naming
logging.basicConfig(filename= arg.dest_dir + "log_" + arg.name_prp + ".log", level=logging.INFO)
# name_date = dict(zip(arg.dic_key, arg.dic_val))
#
# sorting files by modification date ----

# skimming over files

# for root,folder,_ in os.walk(arg.data_dir):
#     for sfolder in folder:
#         frame_count = 0

tesdir = '/home/yizhuo/Documents/Renamed_data/'
dest_dir = '/home/yizhuo/Documents/Untitled Folder/'
# vid_file = glob.glob(tesdir + '')
file_lst = os.listdir(tesdir)
for file in file_lst:
    fdate = file.split('_')[1]
    if len(fdate)<14:
        for i in range(14-len(fdate)):
            fdate = fdate + '0'
    try:
        if int(fdate[4:6]) > int(fdate[6:8]):

            new_name = file
            new_name = list(new_name)
            file = list(file)
            new_name[7:9] = file[9:11]
            new_name[9:11] = file[7:9]
            new_name = ''.join(v for v in new_name)
            file = ''.join(v for v in file)

            os.renames(tesdir + file, dest_dir + new_name)
            logging.info('the file:' + str(file) + 'has changed into' + str(new_name))
        else:
            logging.info('The file ' + str(file) + 'is alright - no change!')
    except:
            print('this file has not the desired structure:', file)
            logging.info('this file has not the desired structure:' + str(file))
    logging.info('***********************************************************')
    # print len(vid_file)
logging.info('This is part of data organization for basketball project puposes which is done at 20180917 by ALI for RAPSODO')
"""
for _ , ssfolder, files  in os.walk(arg.data_dir):
            if ssfolder == []:
                # video_lst = glob.glob(arg.data_dir + '*.MP4')
                video_lst  = [arg.data_dir + item for item in files if item.endswith('.MP4')]
                video_lst.sort(key=os.path.getatime)
                # img_files = glob.glob(arg.data_dir + '*.PNG')
                img_files  = [arg.data_dir + item for item in files if item.endswith('.PNG')]
                img_files.sort(key=os.path.getatime)
            #
            #
            # else:
            #
            #     for i in ssfolder:
            #         img_dir  = root + sfolder + '/' + i + '/'
            #         # sorting wrt modofocation date
            #         files = glob.glob(img_dir)
            #         sorted_files = files.sort(key=os.path.getatime)
            #         for fil in sorted_files:
            #             print fil


                    # for item in sorted_files:
                    #     new_name = arg.name_prj +'_' + name_date[sfolder] + '_' + arg.name_prp +'_' + arg.name_src +'_' + arg.name_loc  +'_' + '0913_0' + str(frame_count) + '.png'
                    #     frame_count +=1
                    #     os.renames(img_dir + item, img_dir + new_name)

print('Naming is finished with success!')
"""







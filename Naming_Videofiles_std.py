#
# automatic Standard data labelling - for Rapsodo
#

import glob, sys, os
import argparse
import logging

dir = os.getcwd()

# defining required directories
parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('-a', '--data_dir',  type=str, help="defining the root", default="/home/yizhuo/Documents/s3/")
parser.add_argument('-a1', '--dest_dir',  type=str, help="destination directory", default="/home/yizhuo/Documents/Untitled Folder/")
parser.add_argument('-n1', "--name_prj", type=str, help="project name prefix for the name", default="ba")
parser.add_argument('-n2', "--name_dt",  type=str, help="date prefix for the name video", default="20180917")
parser.add_argument('-n3', "--name_prp", type=str, help="purpose prefix for the name video", default="hc-in-out-door-self-train") # chenge the purpose -> imortant
parser.add_argument('-n4', "--name_src", type=str, help="source prefix for the name video", default="hc")
parser.add_argument('-n5', "--name_loc", type=str, help="location prefix for the name video", default="sgsg")
parser.add_argument('-n6', "--dic_key", type=str, help="the list of the video (the folder)"
                                                       "this should provided as list", default="")
parser.add_argument('-n7', "--dic_val", type=str, help="the list of date of the video creation yearmmddhhmm"
                                                       "this should provided as list", default="201809171612")
parser.add_argument('-f1', "--frame_name", type=int, help=" prefix for the name the frames", default=0)

arg = parser.parse_args()

# Creating dictionary for naming

name_date = dict(zip(arg.dic_key, arg.dic_val))


# creating logging from all of the process
logging.basicConfig(filename=arg.dest_dir + "log_" + arg.name_prp + ".log", level=logging.INFO)

# skimming over files
video_count = 0

for root,folder,_ in os.walk(arg.data_dir):
    for sfolder in folder:
        frame_count = 0
        flist = os.listdir(root + sfolder)
        for file in flist:
            if file.endswith('.MP4') and file.split('_')[0] != '.':
                # in case that we have date on the name of the videos:
                try:
                    date = file.split('_')[1].split('.')[0].split(' ')[0].split('-')
                    hour = file.split('_')[1].split('.')[0].split(' ')[1].split('-')
                    new_name_video = arg.name_prj +'_' + date[2] + date[1] + date[0] + "".join(hour[0:3]) + '_' + arg.name_prp +'_' + arg.name_src +'_' + arg.name_loc  +  '.mp4'
                    new_name_image = arg.name_prj +'_' + date[2] + date[1] + date[0]+  "".join(hour[0:3]) + '_' + arg.name_prp +'_' + arg.name_src +'_' + arg.name_loc  +  '.png'
                    log_msg = 'The video file: ' + str(file) + ' is successfully changed to: ' + str(new_name_video) + 'at:' + arg.name_dt
                    logging.info(log_msg)


                # in the case that we do not have the date on the name of the videos:
                except:
                    new_name_video = arg.name_prj +'_' + arg.dic_val + str(video_count) + '_' + arg.name_prp +'_' + arg.name_src +'_' + arg.name_loc   +  '.mp4'
                    new_name_image = arg.name_prj +'_' + arg.dic_val + str(video_count) + '_' + arg.name_prp +'_' + arg.name_src +'_' + arg.name_loc  +  '.png'
                    logging.error('There was no date in the file name so we used a default date for:' + str(file))
                    log_msg = 'The video file: ' + str(file) + ' is successfully changed to: ' + str(new_name_video) + 'at:' + arg.name_dt
                    logging.info(log_msg)
                # renaming the video
                os.renames(root + sfolder + '/' + file, arg.dest_dir + new_name_video)
                #
                # in some folders we do not have image, so we should check if image file exists
                try:
                    for pngfile in glob.glob(root + sfolder + '/' +'*.PNG'):
                        os.renames(pngfile, arg.dest_dir + new_name_image)
                        log_msg = 'The image file' + str(pngfile) + 'is successfully changed to: ' + str(new_name_image)  + 'at:' + arg.name_dt
                        logging.info(log_msg)
                except:
                    print('For image there is no image!!!!')
                    logging.error('The image file corresponding to video: ' + str(new_name_video) + 'was not found')
                video_count +=1
                logging.info('***************************************************')

logging.info('** The images of this videos are stored in different folder (they are not organized!)')
print('Naming is finished with success!')







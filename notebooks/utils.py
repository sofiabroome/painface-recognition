# coding: utf-8

import sys
sys.path.append('../')

from data_handler import get_video_id_stem_from_path
from data_handler import get_video_id_from_path
from data_handler import get_video_id_from_frame_path
import visualization
import visualization.visualize_gradcam
import visualization.gradcam as gradcam
from helpers import process_image
from keras.utils import np_utils
from keras.layers import Activation
import skvideo.io
import pandas as pd
import data_handler
import numpy as np
import keras
import cv2
import os


class ArgsProxy:
    def __init__(self, data_path, of_path, input_height, input_width,
                 seq_length, seq_stride, batch_size, nb_labels,
                 aug_flip, aug_crop, aug_light, nb_input_dims):
        self.data_path = data_path
        self.of_path = of_path
        self.input_height = input_height
        self.input_width = input_width
        self.seq_length = seq_length
        self.seq_stride = seq_stride
        self.batch_size = batch_size
        self.nb_labels = nb_labels
        self.aug_flip = aug_flip
        self.aug_crop = aug_crop
        self.aug_light = aug_light
        self.nb_input_dims = nb_input_dims


def read_or_create_subject_dfs(dh, args, subject_ids):
    """
    Read or create the per-subject dataframes listing
    all the frame paths and corresponding labels and metadata.
    :param dh: DataHandler
    :return: [pd.Dataframe]
    """
    subject_dfs = []
    for subject_id in subject_ids:
        print(args.data_path)
        subject_csv_path = args.data_path + subject_id + '.csv'
        if os.path.isfile(subject_csv_path):
            sdf = pd.read_csv(subject_csv_path)
        else:
            print('Making a DataFrame for subject id: ', subject_id)
            sdf = dh.subject_to_df(subject_id)
            sdf.to_csv(path_or_buf=subject_csv_path)
        subject_dfs.append(sdf)
    return subject_dfs

def read_or_create_subject_rgb_and_OF_dfs(dh,
                                          args,
                                          subject_ids,
                                          subject_dfs):
    """
    Read or create the per-subject optical flow files listing
    all the frame paths and labels.
    :param dh: DataHandler object
    :param subject_dfs: [pd.DataFrame]
    :return: [pd.DataFrame]
    """
    subject_rgb_OF_dfs = []
    for ind, subject_id in enumerate(subject_ids):
        subject_of_csv_path = dh.of_path + str(subject_id) + '.csv'
        if os.path.isfile(subject_of_csv_path):
            sdf = pd.read_csv(subject_of_csv_path)
        else:
            print('Making a DataFrame for subject id: ', subject_id)
            sdf = dh.save_OF_paths_to_df(subject_id,
                                         subject_dfs[ind])
            sdf.to_csv(path_or_buf=subject_of_csv_path)
        subject_rgb_OF_dfs.append(sdf)
    return subject_rgb_OF_dfs


def get_sequence(args, subject_dfs, subject=-1, video=None, start_index=None):
    """
    :param subject: int [0,5]
    "param video: str '5_5b'"
    """
    if subject==-1 and video:
        print('Need to provide both subject and video ID to get sequence function.')
    if subject == -1:
        random_subject = np.random.randint(0,6)
        df = subject_dfs[random_subject]
    else:
        print('Chose subject ', subject)
        df = subject_dfs[subject]
    if video:
        df = df[df['Video_ID'] == video]
    if start_index is None:
        random_start_index = np.random.randint(0, len(df))
        start_index = random_start_index
    print('Start index in subject dataframe: ', start_index)
    end_index = start_index + args.seq_length
    
    sequence_df = df.iloc[start_index:end_index]
    sequence_df.reset_index(drop=True, inplace=True)
    
    vid_id_first = get_video_id_from_frame_path(sequence_df.loc[0]['Path'])
    vid_id_last = get_video_id_from_frame_path(sequence_df.loc[args.seq_length-1]['Path'])

    assert(vid_id_first == vid_id_last)
    
    return sequence_df

def make_video_from_frames(frames, path):
    """
    :param frames: [np.array]
    :param path: str, f ex 'output.avi'
    """
    height, width, channels = frames[0,0].shape
    seq_length = frames.shape[1]

    # video = cv2.VideoWriter(path, -1, 1, (width, height))
    # for i in range(seq_length):
    #     video.write(frames[0][i])
    # 
    # cv2.destroyAllWindows()
    # video.release()

    frames = frames.astype(np.uint8)
    inputdict = {'-r': '2'}
    outputdict = {'-r': '2'}
    writer = skvideo.io.FFmpegWriter(path, inputdict=inputdict, outputdict=outputdict)
    print(frames.shape)
    for i in range(seq_length):
            writer.writeFrame(frames[0][i])
    writer.close()


def data_for_one_random_sequence_two_stream(args, subject_dfs, computer, subject=None, start_index=None):
    sequence_df = get_sequence(args, subject_dfs, subject=subject, start_index=start_index)
    print(sequence_df[['Path', 'Pain']])

    image_paths = sequence_df['Path'].values
    of_paths = sequence_df['OF_Path'].values
    y = sequence_df['Pain'].values
    
    label_onehot = np_utils.to_categorical(y, num_classes=args.nb_labels)
    batch_label = label_onehot.reshape(args.batch_size, args.seq_length, -1)

    batch_img = np.concatenate(read_images_and_return_list(args, image_paths, computer), axis=1)
    batch_flow = np.concatenate(read_images_and_return_list(args, of_paths, computer), axis=1)
    return batch_img, batch_flow, batch_label


def data_for_one_random_sequence_4D(args, subject_dfs, computer, subject=None, start_index=None):
    sequence_df = get_sequence(args, subject_dfs, subject=subject, start_index=start_index)
    print(sequence_df[['Path', 'Pain']])

    image_paths = sequence_df['Path'].values
    y = sequence_df['Pain'].values
    
    label_onehot = np_utils.to_categorical(y, num_classes=args.nb_labels)
    batch_label = label_onehot.reshape(args.batch_size, -1)

    batch_img = np.concatenate(read_images_and_return_list(args, image_paths, computer), axis=1)
    batch_img = np.reshape(batch_img, (args.batch_size, args.input_width, args.input_height, 3))
    return batch_img, batch_label

def data_for_one_random_sequence_5D(args, subject_dfs, computer, subject=None, start_index=None):
    sequence_df = get_sequence(args, subject_dfs, subject=subject, start_index=start_index)
    print(sequence_df[['Path', 'Pain']])
    image_paths = sequence_df['Path'].values
    y = sequence_df['Pain'].values
    
    label_onehot = np_utils.to_categorical(y, num_classes=args.nb_labels)
    batch_label = label_onehot.reshape(args.batch_size, args.seq_length, -1)

    batch_img = np.concatenate(read_images_and_return_list(args, image_paths, computer), axis=1)
    batch_img = np.reshape(batch_img, (args.batch_size, args.seq_length, args.input_width, args.input_height, 3))
    return batch_img, batch_label

def read_images_and_return_list(args, paths, computer='hg'):
    list_to_return = []
    for p in paths:
        if computer == 'hg':
            p = '/home/sofia/Documents/painface-recognition/' + p
        elif computer == 'local':
            p = '/Users/sbroome/Documents/EquineML/painface-recognition/' + p
        else:
            p = '/home/sbroome/dev/painface-recognition/' + p
        img = process_image(p, (args.input_width, args.input_height, 3))
        img = img.reshape((1,1,args.input_width, args.input_height, 3))
        list_to_return.append(img)
    return list_to_return


class RodriguezNetwork:
    def __init__(self, rgb, path=None):
        self.rgb = rgb
        self.path = path
        self.m = self.build_from_saved_weights()

    def build_from_saved_weights(self):
        m = keras.models.load_model(self.path)
        x = m.layers[0](self.rgb)
        self.timedist_vgg = m.layers[1](self.rgb)
        x = m.layers[2](self.timedist_vgg)
        self.dense_1 = m.layers[3](x)
        self.lstm = m.layers[4](self.dense_1)
        self.preds = m.layers[5](self.lstm)
        return m


class InceptionNetwork:
    def __init__(self, rgb, from_scratch, path=None):
        self.rgb = rgb
        self.path = path
        self.m = self.build_from_saved_weights()

    def build_from_saved_weights(self):
        print('Loading model...')
        m = keras.models.load_model(self.path)

        print('Finished loading model. Building layers...')
        # RGB-stream
        x = m.layers[0](self.rgb)
        for i in range(1,17):
            x = m.layers[i](x)
        mp_2 = m.layers[17](x)

        c_9 = m.layers[18](mp_2)
        bn_9 = m.layers[19](c_9)
        a_9 = m.layers[20](bn_9)  # output 64

        c_7 = m.layers[21](mp_2)  # output 48
        c_10 = m.layers[22](a_9) # output 96
        bn_7 = m.layers[23](c_7)  # output 48

        bn_10 = m.layers[24](c_10)
        a_7 = m.layers[25](bn_7)
        a_10 = m.layers[26](bn_10)

        ap_1 = m.layers[27](mp_2)
        c_6 = m.layers[28](mp_2)
        c_8 = m.layers[29](a_7)

        c_11 = m.layers[30](a_10)
        c_12 = m.layers[31](ap_1)
        bn_6 = m.layers[32](c_6)
        bn_8 = m.layers[33](c_8)
        bn_11 = m.layers[34](c_11)

        bn_12 = m.layers[35](c_12)
        a_6 = m.layers[36](bn_6)
        a_8 = m.layers[37](bn_8)
        a_11 = m.layers[38](bn_11)
        a_12 = m.layers[39](bn_12)

        # mixed0
        mixed0 = m.layers[40]([a_6, a_8, a_11, a_12])

        c_16 = m.layers[41](mixed0)
        bn_16 = m.layers[42](c_16)
        a_16 = m.layers[43](bn_16)

        c_14 = m.layers[44](mixed0)
        c_17 = m.layers[45](a_16)
        bn_14 = m.layers[46](c_14)
        bn_17 = m.layers[47](c_17)
        a_14 = m.layers[48](bn_14)
        a_17 = m.layers[49](bn_17)
        ap_2 = m.layers[50](mixed0)
        c_13 = m.layers[51](mixed0)
        c_15 = m.layers[52](a_14)
        c_18 = m.layers[53](a_17)
        c_19 = m.layers[54](ap_2)
        bn_13 = m.layers[55](c_13)
        bn_15 = m.layers[56](c_15)
        bn_18 = m.layers[57](c_18)
        bn_19 = m.layers[58](c_19)
        a_13 = m.layers[59](bn_13)
        a_15 = m.layers[60](bn_15)
        a_18 = m.layers[61](bn_18)
        a_19 = m.layers[62](bn_19)
        mixed1 = m.layers[63]([a_13, a_15, a_18, a_19])

        c_23 = m.layers[64](mixed1)
        bn_23 = m.layers[65](c_23)
        a_23 = m.layers[66](bn_23)

        c_21 = m.layers[67](mixed1)
        c_24 = m.layers[68](a_23)
        bn_21 = m.layers[69](c_21)
        bn_24 = m.layers[70](c_24)
        a_21 = m.layers[71](bn_21)
        a_24 = m.layers[72](bn_24)
        ap_3 = m.layers[73](mixed1)
        c_20 = m.layers[74](mixed1)
        c_22 = m.layers[75](a_21)
        c_25 = m.layers[76](a_24)
        c_26 = m.layers[77](ap_3)
        bn_20 = m.layers[78](c_20)
        bn_22 = m.layers[79](c_22)
        bn_25 = m.layers[80](c_25)
        bn_26 = m.layers[81](c_26)
        a_20 = m.layers[82](bn_20)
        a_22 = m.layers[83](bn_22)
        a_25 = m.layers[84](bn_25)
        a_26 = m.layers[85](bn_26)
        mixed2 = m.layers[86]([a_20, a_22, a_25, a_26])
        c_28 = m.layers[87](mixed2)
        bn_28 = m.layers[88](c_28)
        a_28 = m.layers[89](bn_28)

        c_29 = m.layers[90](a_28)
        bn_29 = m.layers[91](c_29)
        a_29 = m.layers[92](bn_29)

        c_27 = m.layers[93](mixed2)
        c_30 = m.layers[94](a_29)
        bn_27 = m.layers[95](c_27)
        bn_30 = m.layers[96](c_30)
        a_27 = m.layers[97](bn_27)
        a_30 = m.layers[98](bn_30)
        mp_3 = m.layers[99](mixed2)

        mixed3 = m.layers[100]([a_27, a_30, mp_3])
        x = m.layers[101](mixed3)
        for i in range(102,107):
            x = m.layers[i](x)

        c_32 = m.layers[107](mixed3)
        c_37 = m.layers[108](x) # a_36
        bn_32 = m.layers[109](c_32) # a_36
        bn_37 = m.layers[110](c_37)
        a_32 = m.layers[111](bn_32)
        a_37 = m.layers[112](bn_37)
        c_33 = m.layers[113](a_32)
        c_38 = m.layers[114](a_37)
        bn_33 = m.layers[115](c_33)
        bn_38 = m.layers[116](c_38)
        a_33 = m.layers[117](bn_33)
        a_38 = m.layers[118](bn_38)
        ap_4 = m.layers[119](mixed3)
        c_31 = m.layers[120](mixed3)
        c_34 = m.layers[121](a_33)
        c_39 = m.layers[122](a_38)
        c_40 = m.layers[123](ap_4)
        bn_31 = m.layers[124](c_31)
        bn_34 = m.layers[125](c_34)
        bn_39 = m.layers[126](c_39)
        bn_40 = m.layers[127](c_40)
        a_31 = m.layers[128](bn_31)
        a_34 = m.layers[129](bn_34)
        a_39 = m.layers[130](bn_39)
        a_40 = m.layers[131](bn_40)
        mixed4 = m.layers[132]([a_31, a_34, a_39, a_40])
        x = m.layers[133](mixed4)
        for i in range(134,139):
            x = m.layers[i](x)  # a_46
        
        c_42 = m.layers[139](mixed4)
        c_47 = m.layers[140](x)
        bn_42 = m.layers[141](c_42)
        bn_47 = m.layers[142](c_47)
        a_42 = m.layers[143](bn_42)
        a_47 = m.layers[144](bn_47)
        c_43 = m.layers[145](a_42)
        c_48 = m.layers[146](a_47)
        bn_43 = m.layers[147](c_43)
        bn_48 = m.layers[148](c_48)
        a_43 = m.layers[149](bn_43)
        a_48 = m.layers[150](bn_48)
        ap_5 = m.layers[151](mixed4)
        c_41 = m.layers[152](mixed4)
        c_44 = m.layers[153](a_43)
        c_49 = m.layers[154](a_48)
        c_50 = m.layers[155](ap_5)
        bn_41 = m.layers[156](c_41)
        bn_44 = m.layers[157](c_44)
        bn_49 = m.layers[158](c_49)
        bn_50 = m.layers[159](c_50)
        a_41 = m.layers[160](bn_41)
        a_44 = m.layers[161](bn_44)
        a_49 = m.layers[162](bn_49)
        a_50 = m.layers[163](bn_50)
        mixed5 = m.layers[164]([a_41, a_44, a_49, a_50])

        x = m.layers[165](mixed5)
        for i in range(166,171):
            x = m.layers[i](x)  # a_46
        
        c_52 = m.layers[171](mixed5)
        c_57 = m.layers[172](x)
        bn_52 = m.layers[173](c_52)
        bn_57 = m.layers[174](c_57)
        a_52 = m.layers[175](bn_52)
        a_57 = m.layers[176](bn_57)
        c_53 = m.layers[177](a_52)
        c_58 = m.layers[178](a_57)
        bn_53 = m.layers[179](c_53)
        bn_58 = m.layers[180](c_58)
        a_53 = m.layers[181](bn_53)
        a_58 = m.layers[182](bn_58)
        ap_6 = m.layers[183](mixed5)
        c_51 = m.layers[184](mixed5)
        c_54 = m.layers[185](a_53)
        c_59 = m.layers[186](a_58)
        c_60 = m.layers[187](ap_6)
        bn_51 = m.layers[188](c_51)
        bn_54 = m.layers[189](c_54)
        bn_59 = m.layers[190](c_59)
        bn_60 = m.layers[191](c_60)
        a_51 = m.layers[192](bn_51)
        a_54 = m.layers[193](bn_54)
        a_59 = m.layers[194](bn_59)
        a_60 = m.layers[195](bn_60)
        mixed6 = m.layers[196]([a_51, a_54, a_59, a_60])

        x = m.layers[197](mixed6)
        for i in range(198,203):
            x = m.layers[i](x)  # a_46
        
        c_62 = m.layers[203](mixed6)
        c_67 = m.layers[204](x)
        bn_62 = m.layers[205](c_62)
        bn_67 = m.layers[206](c_67)
        a_62 = m.layers[207](bn_62)
        a_67 = m.layers[208](bn_67)
        c_63 = m.layers[209](a_62)
        c_68 = m.layers[210](a_67)
        bn_63 = m.layers[211](c_63)
        bn_68 = m.layers[212](c_68)
        a_63 = m.layers[213](bn_63)
        a_68 = m.layers[214](bn_68)
        ap_7 = m.layers[215](mixed6)
        c_61 = m.layers[216](mixed6)
        c_64 = m.layers[217](a_63)
        c_69 = m.layers[218](a_68)
        c_70 = m.layers[219](ap_7)
        bn_61 = m.layers[220](c_61)
        bn_64 = m.layers[221](c_64)
        bn_69 = m.layers[222](c_69)
        bn_70 = m.layers[223](c_70)
        a_61 = m.layers[224](bn_61)
        a_64 = m.layers[225](bn_64)
        a_69 = m.layers[226](bn_69)
        a_70 = m.layers[227](bn_70)
        mixed7 = m.layers[228]([a_61, a_64, a_69, a_70])

        x = m.layers[229](mixed7)
        for i in range(230,235):
            x = m.layers[i](x)  # a_46
        
        c_71 = m.layers[235](mixed7)
        c_75 = m.layers[236](x)
        bn_71 = m.layers[237](c_71)
        bn_75 = m.layers[238](c_75)
        a_71 = m.layers[239](bn_71)
        a_75 = m.layers[240](bn_75)
        c_72 = m.layers[241](a_71)
        c_76 = m.layers[242](a_75)
        bn_72 = m.layers[243](c_72)
        bn_76 = m.layers[244](c_76)
        a_72 = m.layers[245](bn_72)
        a_76 = m.layers[246](bn_76)
        mp_4 = m.layers[247](mixed7)
        mixed8 = m.layers[248]([a_72, a_76, mp_4])
        c_81 = m.layers[249](mixed8)
        bn_81 = m.layers[250](c_81)
        a_81 = m.layers[251](bn_81)

        c_78 = m.layers[252](mixed8)
        c_82 = m.layers[253](a_81)
        bn_78 = m.layers[254](c_78)
        bn_82 = m.layers[255](c_82)
        a_78 = m.layers[256](bn_78)
        a_82 = m.layers[257](bn_82)
        c_79 = m.layers[258](a_78)
        c_80 = m.layers[259](a_78)
        c_83 = m.layers[260](a_82)
        c_84 = m.layers[261](a_82)
        ap_8 = m.layers[262](mixed8)
        c_77 = m.layers[263](mixed8)
        bn_79 = m.layers[264](c_79)
        bn_80 = m.layers[265](c_80)
        bn_83 = m.layers[266](c_83)
        bn_84 = m.layers[267](c_84)
        c_85 = m.layers[268](ap_8)
        bn_77 = m.layers[269](c_77)
        a_79 = m.layers[270](bn_79)
        a_80 = m.layers[271](bn_80)
        a_83 = m.layers[272](bn_83)
        a_84 = m.layers[273](bn_84)
        bn_85 = m.layers[274](c_85)
        a_77 = m.layers[275](bn_77)
        mixed9_0 = m.layers[276]([a_79, a_80])
        concat1 = m.layers[277]([a_83, a_84])
        a_85 = m.layers[278](bn_85)
        mixed9 = m.layers[279]([a_77, mixed9_0, concat1, a_85])
        c_90 = m.layers[280](mixed9)
        bn_90 = m.layers[281](c_90)
        a_90 = m.layers[282](bn_90)
        c_87 = m.layers[283](mixed9)
        c_91 = m.layers[284](a_90)
        bn_87 = m.layers[285](c_87)
        bn_91 = m.layers[286](c_91)
        a_87 = m.layers[287](bn_87)
        a_91 = m.layers[288](bn_91)
        c_88 = m.layers[289](a_87)
        c_89 = m.layers[290](a_87)
        c_92 = m.layers[291](a_91)
        c_93 = m.layers[292](a_91)
        ap_9 = m.layers[293](mixed9)
        self.c_86 = m.layers[294](mixed9)
        bn_88 = m.layers[295](c_88)
        bn_89 = m.layers[296](c_89)
        bn_92 = m.layers[297](c_92)
        bn_93 = m.layers[298](c_93)
        self.c_94 = m.layers[299](ap_9)
        bn_86 = m.layers[300](self.c_86)
        a_88 = m.layers[301](bn_88)
        a_89 = m.layers[302](bn_89)
        a_92 = m.layers[303](bn_92)
        a_93 = m.layers[304](bn_93)
        bn_94 = m.layers[305](self.c_94)
        a_86 = m.layers[306](bn_86)
        mixed9_1 = m.layers[307]([a_88, a_89])
        concat2 = m.layers[308]([a_92, a_93])
        a_94 = m.layers[309](bn_94)
        self.mixed10 = m.layers[310]([a_86, mixed9_1, concat2, a_94])
        gap_1 = m.layers[311](self.mixed10)
        self.dense_1 = m.layers[312](gap_1)
        self.dense_2 = m.layers[313](self.dense_1)
        self.preds = m.layers[314](self.dense_2)
        return m

class TwoStreamCLSTMNetwork:
    def __init__(self, rgb, optical_flow, from_scratch, path=None):
        self.rgb = rgb
        self.optical_flow = optical_flow
        self.path = path
        self.m = self.build_from_saved_weights()

    def build_from_saved_weights(self):
        m = keras.models.load_model(self.path)
        
        # RGB-stream
        x = m.layers[0](self.rgb)
        self.clstm1_rgb = m.layers[2].layers[0](x)
        x = m.layers[2].layers[1](self.clstm1_rgb)
        x = m.layers[2].layers[2](x)
        x = m.layers[2].layers[3](x)
        x = m.layers[2].layers[4](x)
        x = m.layers[2].layers[5](x)
        x = m.layers[2].layers[6](x)
        x = m.layers[2].layers[7](x)
        x = m.layers[2].layers[8](x)
        self.clstm4_rgb = m.layers[2].layers[9](x)
        x = m.layers[2].layers[10](self.clstm4_rgb)
        x = m.layers[2].layers[11](x)
        self.rgb_stream = m.layers[2].layers[12](x)
        
        # FLOW-stream
        y = m.layers[1](self.optical_flow)
        y = m.layers[3].layers[0](y)
        y = m.layers[3].layers[1](y)
        y = m.layers[3].layers[2](y)
        y = m.layers[3].layers[3](y)
        y = m.layers[3].layers[4](y)
        y = m.layers[3].layers[5](y)
        y = m.layers[3].layers[6](y)
        y = m.layers[3].layers[7](y)
        y = m.layers[3].layers[8](y)
        self.clstm4_of = m.layers[3].layers[9](y)
        y = m.layers[3].layers[10](self.clstm4_of)
        y = m.layers[3].layers[11](y)
        self.of_stream = m.layers[3].layers[12](y)
        
        self.merge = m.layers[4]([self.rgb_stream, self.of_stream])
        x = m.layers[5](self.merge)
        self.dense = m.layers[6](x)
        self.preds = Activation('sigmoid')(self.dense)
        return m


def create_graph_for_clstm(batch_size, args, channels, best_model_path, two_stream=True):    
    from keras import backend as K
    import tensorflow as tf
        
    images = tf.placeholder(tf.float32, [batch_size, args.seq_length, args.input_width, args.input_height, channels], name='images')
    flows = tf.placeholder(tf.float32, [batch_size, args.seq_length, args.input_width, args.input_height, channels], name='flows')
    labels = tf.placeholder(tf.float32, [batch_size, args.seq_length, 2], name='labels')
    
    if two_stream:
        clstm_model = TwoStreamCLSTMNetwork(images, flows, from_scratch=0, path=best_model_path)
    else:
        clstm_model = CLSTMNetwork(images, from_scratch=0, path=best_model_path)
        
    sess = K.get_session()  # Grab the Keras session where the weights are initialized.
    
    cost = (-1) * tf.reduce_sum(tf.multiply(labels, tf.log(clstm_model.preds)), axis=1)
        
    y_c = tf.reduce_sum(tf.multiply(clstm_model.dense, labels), axis=1)
        
    if two_stream:
        # target_conv_layer = clstm_model.clstm1_rgb # Choose which CLSTM-layer to study
        target_conv_layer = clstm_model.merge # Choose which CLSTM-layer to study
    else:
        target_conv_layer = clstm_model.clstm4 # Choose which CLSTM-layer to study
        
    target_conv_layer_grad = tf.gradients(y_c, target_conv_layer)[0]
        
    gb_grad = tf.gradients(cost, [images, flows])[0]  # Guided backpropagation back to input layer
    return sess, clstm_model


def plot_sequences(rgb, X_seq_list, flipped, cropped, shaded,
                      seq_index, batch_index, window_index):
    rows = 4
    cols = 10
    f, axarr = plt.subplots(rows, cols, figsize=(20,10))
    for i in range(0, rows):
        for j in range(0, cols):
            axarr[i, j].set_xticks([])
            axarr[i, j].set_yticks([])
            # axarr[i, j].set_aspect('equal')
            if i == 0:
                im = X_seq_list[j]
                im /= 255
                axarr[i, j].imshow(im)
            elif i == 1:
                im = flipped[j]
                im /= 255
                axarr[i, j].imshow(im)
            elif i == 2:
                im = cropped[j]
                im /= 255
                axarr[i, j].imshow(im)
            else:
                im = shaded[j]
                im /= 255
                axarr[i, j].imshow(im)
    plt.tick_params(axis='both', which='both', bottom='off', left='off')
    f.subplots_adjust(wspace=0, hspace=0)
    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.axis('off')
    # plt.tight_layout()
    plt.savefig('seq_{}_batch_{}_wi_{}_part_{}_rgb_{}.png'.format(seq_index,
                                                           batch_index,
                                                           window_index,
                                                           partition,
                                                           rgb))
    plt.close()


def visualize_overlays_4D(images, conv_outputs, conv_grads, flows=None):

    from skimage.transform import resize
    from matplotlib import pyplot as plt

    if flows is not None:
        nb_rows = 3
        fig_width = 23
        fig_height = 7
    else:
        nb_rows = 1
        fig_width = 22
        fig_height = 4.5

    image = images[0,:,:,:]
    if flows is not None:
        flow = flows[0,:,:,:]
        flow = flow.astype(float)
        flow -= np.min(flow)
        flow /= flow.max()

    output = conv_outputs[0,:,:,:]           # [7,7,512]
    grads_val = conv_grads[0,:,:,:]          # [7,7,512]
    weights = np.mean(grads_val, axis = (0,1)) # alpha_k, [512]
    cam = np.zeros(output.shape[0 : 2], dtype = np.float32) # [7,7]

    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # Passing through ReLU
    image = np.reshape(image, (180,320,3))
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam) # scale 0 to 1.0
    cam = resize(cam, (180,320), preserve_range=True)

    img = image.astype(float)
    img -= np.min(img)
    img /= img.max()

    fig = plt.figure(figsize=(fig_width, fig_height))    
    cam_heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
    ax = fig.add_subplot(nb_rows,2,1)
    ax.set_xticks([])
    ax.set_yticks([])
    imgplot = plt.imshow(img)
    
    ax.set_title('Input Image')

    from PIL import Image
    bg = Image.fromarray((255*img).astype('uint8'))
    overlay = Image.fromarray(cam_heatmap.astype('uint8'))
    blend = Image.blend(bg, overlay, 0.2)
    ax = fig.add_subplot(nb_rows,2,2)
    ax.set_xticks([])
    ax.set_yticks([])
    imgplot = plt.imshow(blend)

    ax.set_title('Input Image with GradCAM Overlay')
    plt.tick_params(axis='both', which='both', bottom='off', left='off')
    plt.show()


def visualize_overlays(images, conv_outputs, conv_grads, args, flows=None):

    from skimage.transform import resize
    from matplotlib import pyplot as plt

    if args.nb_input_dims == 5:
        images = images[0]

    if flows is not None:
        nb_rows = 3
        fig_width = 23
        fig_height = 7
    else:
        nb_rows = 2
        fig_width = 22
        fig_height = 4.5

    for im in range(images.shape[0]):
        # print(im)
        image = images[im,:,:,:]
        if flows is not None:
            flow = flows[0,im,:,:]
            flow = flow.astype(float)
            flow -= np.min(flow)
            flow /= flow.max()

        output = conv_outputs[im,:,:,:]           # [7,7,512]
        grads_val = conv_grads[im,:,:,:]          # [7,7,512]
        weights = np.mean(grads_val, axis = (0, 1)) # alpha_k, [512]
        cam = np.zeros(output.shape[0 : 2], dtype = np.float32) # [7,7]

        # Taking a weighted average
        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        # Passing through ReLU
        if args.nb_input_dims == 4:
            image = np.reshape(image, (args.input_height, args.input_width, 3))
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam) # scale 0 to 1.0
        cam = resize(cam, (args.input_height, args.input_width), preserve_range=True)

        img = image.astype(float)
        img -= np.min(img)
        img /= img.max()

        cam_heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
        if im == 0:
            fig = plt.figure(figsize=(fig_width, fig_height))    
        ax = fig.add_subplot(nb_rows,10,im+1)
        ax.set_xticks([])
        ax.set_yticks([])
        imgplot = plt.imshow(img)
        
        if flows is not None:
            ax = fig.add_subplot(nb_rows,10,im+11)
            ax.set_xticks([])
            ax.set_yticks([])
            imgplot = plt.imshow(flow)
        #ax.set_title('Input Image')
        #plt.show()
        

        from PIL import Image
        if flows is not None:
            ax = fig.add_subplot(nb_rows,10,im+21)
        else:
            ax = fig.add_subplot(nb_rows,10,im+11)
            
        ax.set_xticks([])
        ax.set_yticks([])
        bg = Image.fromarray((255*img).astype('uint8'))
        overlay = Image.fromarray(cam_heatmap.astype('uint8'))
        blend = Image.blend(bg, overlay, 0.2)
        imgplot = plt.imshow(blend)
        #ax.set_title('Input Image with GradCAM Overlay')
    plt.tick_params(axis='both', which='both', bottom='off', left='off')
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

def process_frames():
    return frames

def compute_manual_gradcam(images, conv_outputs, conv_grads, args):
    from skimage.transform import resize

    if args.nb_input_dims == 5:
        images = images[0]

    for im in range(images.shape[0]):
        image = images[im,:,:,:]

        output = conv_outputs[im,:,:,:]           # [7,7,512]
        grads_val = conv_grads[im,:,:,:]          # [7,7,512]
        weights = np.mean(grads_val, axis = (0, 1)) # alpha_k, [512]
        cam = np.zeros(output.shape[0 : 2], dtype = np.float32) # [7,7]

        # Taking a weighted average
        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        # Passing through ReLU
        if args.nb_input_dims == 4:
            image = np.reshape(image, (args.input_height, args.input_width, 3))
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam) # scale 0 to 1.0
        cam = resize(cam, (args.input_height, args.input_width), preserve_range=True)

        img = image.astype(float)
        img -= np.min(img)
        img /= img.max()

        cam_heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
        if im == 0:
            fig = plt.figure(figsize=(fig_width, fig_height))    
        
        from PIL import Image
        bg = Image.fromarray((255*img).astype('uint8'))
        overlay = Image.fromarray(cam_heatmap.astype('uint8'))
        blend = Image.blend(bg, overlay, 0.2)
        imgplot = plt.imshow(blend)
    return heatmaps


def visualize_sequence_and_gradcam(frames, heatmaps, flows=None):
    from matplotlib import pyplot as plt
    from PIL import Image

    fig = plt.figure(figsize=(fig_width, fig_height))    

    if args.nb_input_dims == 5:
        images = images[0]

    if flows is not None:
        nb_rows = 3
        fig_width = 23
        fig_height = 7
    else:
        nb_rows = 2
        fig_width = 22
        fig_height = 4.5

    for im in range(frames.shape[0]):
        image = frames[im,:,:,:]
        if flows is not None:
            flow = flows[0,im,:,:]
            flow = flow.astype(float)
            flow -= np.min(flow)
            flow /= flow.max()

        img = image.astype(float)
        img -= np.min(img)
        img /= img.max()

        ax = fig.add_subplot(nb_rows,10,im+1)
        ax.set_xticks([])
        ax.set_yticks([])
        imgplot = plt.imshow(img)
        
        if flows is not None:
            ax = fig.add_subplot(nb_rows,10,im+11)
            ax.set_xticks([])
            ax.set_yticks([])
            imgplot = plt.imshow(flow)

        if flows is not None:
            ax = fig.add_subplot(nb_rows,10,im+21)
        else:
            ax = fig.add_subplot(nb_rows,10,im+11)
            
        ax.set_xticks([])
        ax.set_yticks([])
        bg = Image.fromarray((255*img).astype('uint8'))
        overlay = Image.fromarray(cam_heatmap.astype('uint8'))
        blend = Image.blend(bg, overlay, 0.2)
        imgplot = plt.imshow(blend)

    plt.tick_params(axis='both', which='both', bottom='off', left='off')
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def run_sess_for_clstm_networks():
    with sess.as_default():
    
        # 2-stream
        if two_stream:
            prob = sess.run(clstm_model.preds,
                            feed_dict={images: batch_img,
                                       flows: batch_flow,
                                       K.learning_phase(): 0})
    
            print(prob)
    
            gb_grad_value, target_conv_layer_value, target_conv_layer_grad_value =             sess.run([gb_grad, target_conv_layer, target_conv_layer_grad],
                          feed_dict={images: batch_img,
                                     flows: batch_flow,
                                     labels: batch_label,
                                     K.learning_phase(): 0})
    
            target_conv_layer_value = np.reshape(target_conv_layer_value,
                                             (1, 10, 8, 8, 32))
            target_conv_layer_grad_value = np.reshape(target_conv_layer_grad_value,
                                             (1, 10, 8, 8, 32))
        else:
            prob = sess.run(clstm_model.preds,
                            feed_dict={images: batch_img,
                                       K.learning_phase(): 0})
    
            print(prob)
            gb_grad_value, target_conv_layer_value, target_conv_layer_grad_value =             sess.run([gb_grad, target_conv_layer, target_conv_layer_grad],
                          feed_dict={images: batch_img,
                          labels: batch_label,
                          K.learning_phase(): 0})
            
            target_conv_layer_value = np.reshape(target_conv_layer_value,
                                             (1, 10, 16, 16, 32))
            target_conv_layer_grad_value = np.reshape(target_conv_layer_grad_value,
                                             (1, 10, 16, 16, 32))
    
    
        visualize_overlays(batch_img, target_conv_layer_value,
                           target_conv_layer_grad_value)
    
        for i in range(seq_length):
            print(prob[0,i,:])
        print(batch_label)


def run_on_one_sequence(sess, clstm_model, batch_img, batch_flow, two_stream=True):
    from keras import backend as K
    with sess.as_default():
        g = sess.graph
        images = g.get_tensor_by_name("images")
        flows = g.get_tensor_by_name("flows")
        labels = g.get_tensor_by_name("labels")

        # 2-stream
        if two_stream:
            prob = sess.run(clstm_model.preds,
                            feed_dict={images: batch_img,
                                       flows: batch_flow,
                                       K.learning_phase(): 0})

            print(prob)

            gb_grad_value, target_conv_layer_value, target_conv_layer_grad_value = sess.run([gb_grad, target_conv_layer, target_conv_layer_grad],
                          feed_dict={images: batch_img,
                                     flows: batch_flow,
                                     labels: batch_label,
                                     K.learning_phase(): 0})

            target_conv_layer_value = np.reshape(target_conv_layer_value,
                                             (1, 10, 8, 8, 32))
            target_conv_layer_grad_value = np.reshape(target_conv_layer_grad_value,
                                             (1, 10, 8, 8, 32))
        else:
            prob = sess.run(clstm_model.preds,
                            feed_dict={images: batch_img,
                                       K.learning_phase(): 0})

            print(prob)
            gb_grad_value, target_conv_layer_value, target_conv_layer_grad_value = sess.run([gb_grad, target_conv_layer, target_conv_layer_grad],
                          feed_dict={images: batch_img,
                          labels: batch_label,
                          K.learning_phase(): 0})
            target_conv_layer_value = np.reshape(target_conv_layer_value,
                                         (1, 10, 16, 16, 32))
            target_conv_layer_grad_value = np.reshape(target_conv_layer_grad_value,
                                         (1, 10, 16, 16, 32))


        visualize_overlays(batch_img, target_conv_layer_value,
                           target_conv_layer_grad_value)

        for i in range(seq_length):
            print(prob[0,i,:])
        print(batch_label)


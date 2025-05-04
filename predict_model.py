import pysam
import numpy as np
import math
import os
import torch
# from train_insnet import TTB_Insnet
from train_svt import TTB_Insnet
# from train_cat import TTB_Insnet
# from train_plus import TTB_Insnet
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import time

def decode_flag(Flag):

    signal = {1 << 2: 0, 1 >> 1: 1, 1 << 4: 2, 1 << 11: 3, 1 << 4 | 1 << 11: 4}

    return signal[Flag] if(Flag in signal) else 0

def c_pos(cigar, refstart):

    number = ''
    numlist = [str(i) for i in range(10)]
    readstart = False
    readend = False
    refend = False
    readloc = 0
    refloc = refstart
    for c in cigar:
        if(c in numlist):
            number += c
        else:
            number = int(number)
            if(readstart == False and c in ['M', 'I', '=', 'X']):
                readstart = readloc
            if(readstart != False and c in ['H', 'S']):
                readend = readloc
                refend = refloc
                break
            if(c in ['M', 'I', 'S', '=', 'X']):
                readloc += number
            if(c in ['M', 'D', 'N', '=', 'X']):
                refloc += number
            number = ''
    if(readend == False):
        readend = readloc
        refend = refloc

    return refstart, refend, readstart, readend 

def ins_signature(pre, bamfile):

    data = []
    for chr_name,start,end in pre:

        for read in bamfile.fetch(chr_name,start,end):
            aligned_length = read.reference_length
            if aligned_length == None:
                aligned_length= 0
            if (read.mapping_quality >= 0) and aligned_length >= 0:
                cigar = []
                sta = read.reference_start
                for ci  in read.cigartuples:
                    if ci[0] in [0, 2, 3, 7, 8]:
                        sta += ci[1]
                    elif ci[0] == 1 :
                        if ci[1] >=50 and (abs(sta-start) < 200):
                            cigar.append([sta,sta,ci[1]])
                if len(cigar) !=0:
                    cigar = np.array(cigar)
                    cigar = cigar[np.argsort(cigar[:,0])]
                    a = mergecigar(cigar)
                    data.extend(a)
            if(read.has_tag('SA') == True):
                code = decode_flag(read.flag)
                rawsalist = read.get_tag('SA').split(';')
                for sa in rawsalist[:-1]:
                    sainfo = sa.split(',')
                    tmpcontig, tmprefstart, strand, cigar = sainfo[0], int(sainfo[1]), sainfo[2], sainfo[3]
                    if(tmpcontig != chr_name):
                        continue 
                    if((strand == '-' and (code %2) ==0) or (strand == '+' and (code %2) ==1)):
                        refstart_1, refend_1, readstart_1, readend_1 =  read.reference_start, read.reference_end,read.query_alignment_start,read.query_alignment_end
                        refstart_2, refend_2, readstart_2, readend_2 = c_pos(cigar, tmprefstart)
                        a = readend_1 - readstart_2
                        b = refend_1 - refstart_2
                        if(abs(b-a)<30):
                            continue
                        if((b-a)>=50 and ((b-a)>0)):
                            data22 = []                          
                            if(refend_1<=end and refend_1>=start):
                                data22.append([refend_1,refend_1,abs((b-a))])
                            if(refstart_2<=end and refstart_2>=start):
                                data22.append([refstart_2,refstart_2,abs((b-a))])
                            data22 = np.array(data22)
                            if len(data22)==0:
                                continue
                            data.extend(data22)    
    data = np.array(data)
    if len(data) == 0:
        return data
    data = data[np.argsort(data[:,0])]
                      
    return data

def mergecigar(infor):

    data = []
    i = 0
    while i>=0:
        count = 0
        if i >(len(infor)-1):
            break
        lenth = infor[i][2]
        for j in range(i+1,len(infor)):
            if abs(infor[j][1] - infor[i][1]) <= 40: 
                count = count + 1
                infor [i][1] = infor[j][0]#改[0]0
                lenth = lenth +  infor[j][2] #+ abs(infor[j][0] - infor[i][0])
        data.append([infor[i][0],infor[i][0]+1, lenth])
        if count == 0:
            i += 1
        else :
            i += (count+1)

    return data

def merge(infor):

    data = []
    i = 0
    while i>=0:
        dat = []     
        count = 0
        if i >(len(infor)-1):
            break
        dat.append(infor[i])
        for j in range(i+1,len(infor)):
            if( (abs(infor[i][0] -infor[j][0]) <= 1500) and (abs(infor[i][1] - infor[j][1])<= 1500)):
                count = count + 1
                dat.append(infor[j])
        dat = np.array(dat)
        data.append(dat)
        if count == 0:
            i += 1
        else :
            i += (count+1)

    return data

def merge_insnet_long(pre, index, chr_name, bamfile):

    data = []
    insertion = []
    for i in range(len(pre)):
        if pre[i] > 0.5:
            data.append([chr_name, index[i], index[i] + 200])
    signature = ins_signature(data, bamfile)
    ins_sigs = merge(signature)
    for sig in ins_sigs:
            pp = np.array(sig)
            start = math.ceil(np.median(pp[:, 0]))
            kk = int(len(pp)/2)
            svle = np.sort(pp[:,2])
            length = math.ceil(np.median(svle[kk:]))
            insertion.append([chr_name, start, length, len(pp), 'INS'])

    return insertion 

def predict_step(base, predict):

    for i in range(len(predict)):
        if predict[i] >= 0.5:
            # print(f'predict[i] = {predict[i]}')
            base[i] = 1
            base[i + 1] = 1

    return base

def tovcf(sv_callers, contig2length, sv_types, outvcfpath, version):

    vcf_output = open(outvcfpath, 'w')
    
    print("##fileformat=VCFv4.2", file=vcf_output)
    print("##fileDate={0}".format(time.strftime("%Y-%m-%d|%I:%M:%S%p|%Z|%z")), file=vcf_output)
    print("##source=INSnet_pro-v{0}".format(version), file=vcf_output)
    print("##FILTER=<ID=PASS,Description=\"All filters passed\">", file=vcf_output)
    for contig, length in contig2length.items():
        print("##contig=<ID={0}, length={1}>".format(contig, length), file=vcf_output)
        
    if "INS" in sv_types:
        print("##ALT=<ID=INS, Description=\"Insertion\">", file=vcf_output)
    if "DEL" in sv_types:
        print("##ALT=<ID=DEL, Description=\"Deletion\">", file=vcf_output)
    if "INV" in sv_types:
        print("##ALT=<ID=INV, Description=\"Inversion\">", file=vcf_output)
    if "BND" in sv_types:
        print("##ALT=<ID=BND,Description=\"Breakend\">", file=vcf_output)
        
    print("##INFO=<ID=END,Number=1,Type=Integer,Description=\"End position of the structural variant\">", file=vcf_output)
    print("##INFO=<ID=SVTYPE,Number=1,Type=String,Description=\"Type of structural variant\">", file=vcf_output)
    print("##INFO=<ID=SVLEN,Number=1,Type=Integer,Description=\"Difference in length between REF and ALT alleles\">", file=vcf_output)
    print("##INFO=<ID=SUPPORT,Number=1,Type=Integer,Description=\"Number of read support this record\">", file=vcf_output)
    print("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">", file=vcf_output)
    print("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t.", file=vcf_output)
    
    ins_id = 0
    for sv in sv_callers:
        if sv[4] == 'INS':
            recinfo = 'SVLEN=' + str(int(sv[2])) + ';SVTYPE=' + str(sv[4]) + ';END=' + str(sv[1]) + ';SUPPORT=' + str(sv[3]) + '\tGT\t' + '.\n'
        vcf_output.write(sv[0] + '\t' + str(int(sv[1])) + '\t' + 'INS-pro.'+ str(ins_id) + '\t' + 'N' + '\t' + '<' + str(sv[4]) + '>' + '\t' + str(int(sv[3])+1) + '\t' + 'PASS' + '\t' + recinfo)
        ins_id += 1
    vcf_output.close()

def batchdata(data, timesteps, step, num_gpus = 2, window = 200):

    if step != 0:
        data = data.reshape(-1, 5)[step:(step - window)]
    data = data.reshape(-1, 200, 5)

    size = data.shape[0] // (timesteps * num_gpus)
    size_ = data.shape[0] % (timesteps * num_gpus)

    return data[: size * (timesteps * num_gpus)], data[size * (timesteps * num_gpus) :]

def process_data(data, batch_size, timesteps, offset, num_gpus, predict_ins, device):

    # print(f'data = {data.shape}')
    features_main, features_remain = batchdata(data, timesteps, offset)
    features_main = features_main.reshape(-1, 200, 5, 1) if len(features_main) > 0 else np.array([])
    features_remain = features_remain.reshape(-1, 200, 5, 1) if len(features_remain) > 0 else np.array([])
    # print(f'features_main = {features_main.shape}, features_remain = {features_remain.shape}')

    remain = features_remain.shape[0] % (timesteps * num_gpus)
    padding_size = (timesteps * num_gpus) - remain
    # print(f'remain = {remain}, padding_size = {padding_size}')

    if remain != 0:
        patch_features = np.zeros((padding_size, 200, 5, 1))
        # print(f'patch_features = {patch_features.shape}, features_remain = {features_remain.shape}, features_main = {features_main.shape}')
        features_padded = np.concatenate((features_remain, patch_features), axis=0)
        if len(features_main) > 0:
            features = np.concatenate((features_main, features_padded), axis=0)
        else:
            features = features_padded
    else:
        features = features_main
    
    # print(f'features = {features.shape}')
    predict = predict_data(features, predict_ins, device, batch_size, timesteps)
    # print(f'predict = {predict.shape}')
    if len(predict) == 0:
        return np.array([])
    else:
        predict_main = predict[:-1, :, :]
        predict_remain =  predict[-1:, :, :]
        # print(f'predict_main = {predict_main.shape}, predict_remain = {predict_remain.shape}')
        predict_main = predict_main.flatten()
        predict_remain = predict_remain.flatten()
        if padding_size == timesteps * num_gpus:
            predict_remain = predict_remain[:]
        else:
            predict_remain = predict_remain[:-padding_size]
        # print(f'predict_main = {predict_main.shape}, predict_remain = {predict_remain.shape}')
        predict_end = np.concatenate((predict_main, predict_remain), axis=0)

        return predict_end

def predict_data(feature, predict_ins, device, batch_size, timesteps):

    if len(feature) == 0:
        return np.array([])
    if isinstance(feature, np.ndarray):
        feature = torch.tensor(feature, dtype=torch.float32)

    dataset = TensorDataset (feature)
    dataloader = DataLoader(dataset, batch_size=batch_size * timesteps, shuffle=False)

    predict_ins.eval()
    predictions = []

    with torch.no_grad():  # 关闭梯度计算
        for batch in dataloader:
            inputs = batch[0].to(device)  # 确保数据在正确的设备上
            # print(f'inputs no_grad = {inputs.shape}')
            outputs = predict_ins(inputs)  # 模型预测
            predictions.append(outputs.cpu().numpy())  # 将预测结果转换为 numpy 数组并存储

    # 将所有预测结果合并为一个数组
    return np.concatenate(predictions, axis=0)

def predict_funtion(gpu_name, save_length, timesteps, ins_predict_weight, data_path, bamfilepath, outvcfpath, contigs, support):

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_name
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bamfile_data = pysam.AlignmentFile(bamfilepath,'rb')
    index_stats = bamfile_data.get_index_statistics()
    contigs_lengths = {stat.contig: bamfile_data.lengths[i] for i, stat in enumerate(index_stats)}

    print(f'contigs_lengths = {contigs_lengths}')

    resultlist = [['CONTIG', 'START', 'SVLEN', 'READ_SUPPORT', 'SVTYPE']]

    predict_ins = TTB_Insnet(timesteps)
    # 加载权重
    state_dict = torch.load(ins_predict_weight)
    # 如果权重是通过 DataParallel 保存的, 需要移除 'module.' 前缀
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    predict_ins.load_state_dict(new_state_dict, strict=False)
    # predict_ins = nn.DataParallel(predict_ins)
    predict_ins.to(DEVICE)

    for chr_name in contigs:
        chr_name = chr_name[3:] if 'chr' in chr_name else chr_name
        chr_length = contigs_lengths[str(chr_name)]
        iders = math.ceil(chr_length / save_length)
        start = 0
        print('+++++++',chr_name,'++++++++++')
        print('chr:',chr_name, iders)
        for ider in range(iders):
            print(f'insertion_predict_chr : chr_name = {chr_name}, ider / iders = {ider} / {iders}')
            try:
                data_name = data_path +'/chr'+ chr_name + '_' + str(start)  +  '_' + str(start + save_length) + '.npy'
                data = np.load(data_name)
            except FileNotFoundError:
                start = start + save_length
                continue
            else:
                index_name = data_path +'/chr'+ chr_name + '_' + str(start)  +  '_' + str(start + save_length) + '_index.npy'
                index  = np.load(index_name)
                if len(data) == 0:
                    continue
                print(f'data = {data.shape}, index = {index.shape}')
                base = process_data(data, 20, 100, 0, 2, predict_ins, DEVICE) # 0
                base = predict_step(base, process_data(data, 20, 100, 100, 2, predict_ins, DEVICE)) # 2
                base = predict_step(base, process_data(data, 20, 100, 50, 2, predict_ins, DEVICE)) # 1
                base = predict_step(base, process_data(data, 20, 100, 150, 2, predict_ins, DEVICE)) # 3
 
                contig, start = chr_name, start
                resultlist += merge_insnet_long(base, index, contig, bamfile_data)

                start = start + save_length
    print(f'resultlist = {len(resultlist)}')
    sv_callers = []
    for read in resultlist[1:]:
        if read[3] >= int(support) and read[2] >= 50: 
            sv_callers.append([read[0], read[1], read[2], read[3], 'INS', '.'])
    sv_types = ['INS']

    tovcf(sv_callers, contigs_lengths, sv_types, outvcfpath, version=1)
    
    return resultlist

# 70x
# ins_predict_weight = '/home/laicx/03.study/04.Insnet/01.insnet_gao/train_model/04.model/train_insnet.pth'
# data_path = '/home/laicx/03.study/04.Insnet/00.INSnet/000.dataset/01.HG002_PB_70x_RG_HP10XtrioRTG'
# bamfilepath = '/home/laicx/00.dataset/HG002_PB_70x_RG_HP10XtrioRTG.bam'
# outvcfpath = '/home/laicx/03.study/04.Insnet/01.insnet_gao/08.vcf_data/support_12_insnet/01.HG002_PB_35x_RG_HP10XtrioRTG_insnet.vcf'

# # 35x
# ins_predict_weight = '/home/laicx/03.study/04.Insnet/01.insnet_gao/train_model/04.model/train_insnet.pth'
# data_path = '/home/laicx/03.study/04.Insnet/00.INSnet/000.dataset/01.HG002_PB_35x_RG_HP10XtrioRTG'
# bamfilepath = '/home/laicx/00.dataset/HG002_PB_35x_RG_HP10XtrioRTG.bam'
# outvcfpath = '/home/laicx/03.study/04.Insnet/01.insnet_gao/08.vcf_data/support_12_insnet/01.HG002_PB_35x_RG_HP10XtrioRTG_insnet.vcf'

# # 20x
# ins_predict_weight = '/home/laicx/03.study/04.Insnet/01.insnet_gao/train_model/04.model/train_insnet.pth'
# data_path = '/home/laicx/03.study/04.Insnet/00.INSnet/000.dataset/01.HG002_PB_20x_RG_HP10XtrioRTG'
# bamfilepath = '/home/laicx/00.dataset/HG002_PB_20x_RG_HP10XtrioRTG.bam'
# outvcfpath = '/home/laicx/03.study/04.Insnet/01.insnet_gao/08.vcf_data/support_12_insnet/01.HG002_PB_20x_RG_HP10XtrioRTG_insnet.vcf'

# # 10x
# ins_predict_weight = '/home/laicx/03.study/04.Insnet/01.insnet_gao/train_model/04.model/train_insnet.pth'
# data_path = '/home/laicx/03.study/04.Insnet/00.INSnet/000.dataset/01.HG002_PB_10x_RG_HP10XtrioRTG'
# bamfilepath = '/home/laicx/00.dataset/HG002_PB_10x_RG_HP10XtrioRTG.bam'
# outvcfpath = '/home/laicx/03.study/04.Insnet/01.insnet_gao/08.vcf_data/support_12_insnet/01.HG002_PB_10x_RG_HP10XtrioRTG_insnet.vcf'

# # 5x
# ins_predict_weight = '/home/laicx/03.study/04.Insnet/01.insnet_gao/train_model/04.model/train_insnet.pth'
# data_path = '/home/laicx/03.study/04.Insnet/00.INSnet/000.dataset/01.HG002_PB_5x_RG_HP10XtrioRTG'
# bamfilepath = '/home/laicx/00.dataset/HG002_PB_5x_RG_HP10XtrioRTG.bam'
# outvcfpath = '/home/laicx/03.study/04.Insnet/01.insnet_gao/08.vcf_data/support_12_insnet/01.HG002_PB_5x_RG_HP10XtrioRTG_insnet.vcf'

# # 48x
# ins_predict_weight = '/home/laicx/03.study/04.Insnet/01.insnet_gao/train_model/04.model/train_insnet.pth'
# data_path = '/home/laicx/03.study/04.Insnet/00.INSnet/000.dataset/02.HG002_GRCh37_ONT-UL_UCSC_20200508.phased'
# bamfilepath = '/home/laicx/00.dataset/HG002_GRCh37_ONT-UL_UCSC_20200508.phased.bam'
# outvcfpath = '/home/laicx/03.study/04.Insnet/01.insnet_gao/08.vcf_data/support_12_insnet/02.HG002_GRCh37_ONT-UL_UCSC_20200508.phased_insnet.vcf'
# support = 10

# # 20x
# ins_predict_weight = '/home/laicx/03.study/04.Insnet/01.insnet_gao/train_model/04.model/train_insnet.pth'
# data_path = '/home/laicx/03.study/04.Insnet/00.INSnet/000.dataset/02.HG002_GRCh37_ONT-UL_UCSC_20200508_20x.phased'
# bamfilepath = '/home/laicx/00.dataset/HG002_GRCh37_ONT-UL_UCSC_20200508_20x.phased.bam'
# outvcfpath = '/home/laicx/03.study/04.Insnet/01.insnet_gao/08.vcf_data/support_12_insnet/02.HG002_GRCh37_ONT-UL_UCSC_20200508_20x.phased_insnet.vcf'
# support = 5

# # 10x
# ins_predict_weight = '/home/laicx/03.study/04.Insnet/01.insnet_gao/train_model/04.model/train_insnet.pth'
# data_path = '/home/laicx/03.study/04.Insnet/00.INSnet/000.dataset/02.HG002_GRCh37_ONT-UL_UCSC_20200508_20x.phased'
# bamfilepath = '/home/laicx/00.dataset/HG002_GRCh37_ONT-UL_UCSC_20200508_10x.phased.bam'
# outvcfpath = '/home/laicx/03.study/04.Insnet/01.insnet_gao/08.vcf_data/support_12_insnet/02.HG002_GRCh37_ONT-UL_UCSC_20200508_10x.phased_insnet.vcf'
# support = 3

# # 5x
# ins_predict_weight = '/home/laicx/03.study/04.Insnet/01.insnet_gao/train_model/04.model/train_insnet.pth'
# data_path = '/home/laicx/03.study/04.Insnet/00.INSnet/000.dataset/02.HG002_GRCh37_ONT-UL_UCSC_20200508_5x.phased'
# bamfilepath = '/home/laicx/00.dataset/HG002_GRCh37_ONT-UL_UCSC_20200508_5x.phased.bam'
# outvcfpath = '/home/laicx/03.study/04.Insnet/01.insnet_gao/08.vcf_data/support_12_insnet/02.HG002_GRCh37_ONT-UL_UCSC_20200508_5x.phased_insnet.vcf'
# support = 2

'''-----------------------------------------------------------------------------------------------------------------------'''

# # 70x
# ins_predict_weight = '/home/laicx/03.study/04.Insnet/01.insnet_gao/train_model/04.model/train_cat_filter.pth'
# data_path = '/home/laicx/03.study/04.Insnet/00.INSnet/000.dataset/01.HG002_PB_70x_RG_HP10XtrioRTG'
# bamfilepath = '/home/laicx/00.dataset/HG002_PB_70x_RG_HP10XtrioRTG.bam'
# outvcfpath = '/home/laicx/03.study/04.Insnet/01.insnet_gao/08.vcf_data/support_12_cat/01.HG002_PB_70x_RG_HP10XtrioRTG_cat_filter.vcf'
# support = 11

# # 35x
# ins_predict_weight = '/home/laicx/03.study/04.Insnet/01.insnet_gao/train_model/04.model/train_cat_filter.pth'
# data_path = '/home/laicx/03.study/04.Insnet/00.INSnet/000.dataset/01.HG002_PB_35x_RG_HP10XtrioRTG'
# bamfilepath = '/home/laicx/00.dataset/HG002_PB_35x_RG_HP10XtrioRTG.bam'
# outvcfpath = '/home/laicx/03.study/04.Insnet/01.insnet_gao/08.vcf_data/support_12_cat/01.HG002_PB_35x_RG_HP10XtrioRTG_cat_filter.vcf'
# support = 6

# # 20x
# ins_predict_weight = '/home/laicx/03.study/04.Insnet/01.insnet_gao/train_model/04.model/train_cat_filter.pth'
# data_path = '/home/laicx/03.study/04.Insnet/00.INSnet/000.dataset/01.HG002_PB_20x_RG_HP10XtrioRTG'
# bamfilepath = '/home/laicx/00.dataset/HG002_PB_20x_RG_HP10XtrioRTG.bam'
# outvcfpath = '/home/laicx/03.study/04.Insnet/01.insnet_gao/08.vcf_data/support_12_cat/01.HG002_PB_20x_RG_HP10XtrioRTG_cat_filter.vcf'
# support = 4

# # 10x
# ins_predict_weight = '/home/laicx/03.study/04.Insnet/01.insnet_gao/train_model/04.model/train_cat_filter.pth'
# data_path = '/home/laicx/03.study/04.Insnet/00.INSnet/000.dataset/01.HG002_PB_10x_RG_HP10XtrioRTG'
# bamfilepath = '/home/laicx/00.dataset/HG002_PB_10x_RG_HP10XtrioRTG.bam'
# outvcfpath = '/home/laicx/03.study/04.Insnet/01.insnet_gao/08.vcf_data/support_12_cat/01.HG002_PB_10x_RG_HP10XtrioRTG_cat_filter.vcf'
# support = 3

# # 5x
# ins_predict_weight = '/home/laicx/03.study/04.Insnet/01.insnet_gao/train_model/04.model/train_cat_filter.pth'
# data_path = '/home/laicx/03.study/04.Insnet/00.INSnet/000.dataset/01.HG002_PB_5x_RG_HP10XtrioRTG'
# bamfilepath = '/home/laicx/00.dataset/HG002_PB_5x_RG_HP10XtrioRTG.bam'
# outvcfpath = '/home/laicx/03.study/04.Insnet/01.insnet_gao/08.vcf_data/support_12_cat/01.HG002_PB_5x_RG_HP10XtrioRTG_cat_filter.vcf'
# support = 2

# # 48x
# ins_predict_weight = '/home/laicx/03.study/04.Insnet/01.insnet_gao/train_model/04.model/train_cat_filter.pth'
# data_path = '/home/laicx/03.study/04.Insnet/00.INSnet/000.dataset/02.HG002_GRCh37_ONT-UL_UCSC_20200508.phased'
# bamfilepath = '/home/laicx/00.dataset/HG002_GRCh37_ONT-UL_UCSC_20200508.phased.bam'
# outvcfpath = '/home/laicx/03.study/04.Insnet/01.insnet_gao/08.vcf_data/support_12_cat/02.HG002_GRCh37_ONT-UL_UCSC_20200508.cat_filter.vcf'
# support = 10

# # 20x
# ins_predict_weight = '/home/laicx/03.study/04.Insnet/01.insnet_gao/train_model/04.model/train_cat_filter.pth'
# data_path = '/home/laicx/03.study/04.Insnet/00.INSnet/000.dataset/02.HG002_GRCh37_ONT-UL_UCSC_20200508_20x.phased'
# bamfilepath = '/home/laicx/00.dataset/HG002_GRCh37_ONT-UL_UCSC_20200508_20x.phased.bam'
# outvcfpath = '/home/laicx/03.study/04.Insnet/01.insnet_gao/08.vcf_data/support_12_cat/02.HG002_GRCh37_ONT-UL_UCSC_20200508_20x.cat_filter.vcf'
# support = 5

# # 10x
# ins_predict_weight = '/home/laicx/03.study/04.Insnet/01.insnet_gao/train_model/04.model/train_cat_filter.pth'
# data_path = '/home/laicx/03.study/04.Insnet/00.INSnet/000.dataset/02.HG002_GRCh37_ONT-UL_UCSC_20200508_20x.phased'
# bamfilepath = '/home/laicx/00.dataset/HG002_GRCh37_ONT-UL_UCSC_20200508_10x.phased.bam'
# outvcfpath = '/home/laicx/03.study/04.Insnet/01.insnet_gao/08.vcf_data/support_12_cat/02.HG002_GRCh37_ONT-UL_UCSC_20200508_10x.cat_filter.vcf'
# support = 3

# # 5x
# ins_predict_weight = '/home/laicx/03.study/04.Insnet/01.insnet_gao/train_model/04.model/train_cat_filter.pth'
# data_path = '/home/laicx/03.study/04.Insnet/00.INSnet/000.dataset/02.HG002_GRCh37_ONT-UL_UCSC_20200508_5x.phased'
# bamfilepath = '/home/laicx/00.dataset/HG002_GRCh37_ONT-UL_UCSC_20200508_5x.phased.bam'
# outvcfpath = '/home/laicx/03.study/04.Insnet/01.insnet_gao/08.vcf_data/support_12_cat/02.HG002_GRCh37_ONT-UL_UCSC_20200508_5x.cat_filter.vcf'
# support = 2

import argparse

# def predict_function(gpu_name, save_length, timesteps, ins_predict_weight, data_path, bamfilepath, outvcfpath, contigs, support):
#     """
#     模拟预测函数的实现。
#     """
#     print(f"Running prediction with parameters:")
#     print(f"GPU Name: {gpu_name}")
#     print(f"Save Length: {save_length}")
#     print(f"Timesteps: {timesteps}")
#     print(f"Insert Predict Weight: {ins_predict_weight}")
#     print(f"Data Path: {data_path}")
#     print(f"BAM File Path: {bamfilepath}")
#     print(f"Output VCF Path: {outvcfpath}")
#     print(f"Contigs: {contigs}")
#     print(f"Support: {support}")
#     # 返回一个示例结果
#     return ["result1", "result2", "result3"]

def main():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Run prediction with given parameters.")
    
    # 添加命令行参数
    parser.add_argument("--gpu_name", default="0", type=str, required=True, help="Name of the GPU to use.")
    parser.add_argument("--save_length", default=10000000, type=int, required=True, help="Save length parameter.")
    parser.add_argument("--timesteps", type=int, required=True, help="Timesteps parameter.")
    parser.add_argument("--ins_predict_weight", type=str, required=True, help="Path to the insert predict weight.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data.")
    parser.add_argument("--bamfilepath", type=str, required=True, help="Path to the BAM file.")
    parser.add_argument("--outvcfpath", type=str, required=True, help="Path to the output VCF file.")
    parser.add_argument("--contigs", type=str, nargs="+", required=True, help="List of contigs (e.g., chr1 chr2).")
    parser.add_argument("--support", type=int, required=True, help="Support value.")

    # 解析命令行参数
    args = parser.parse_args()

    # 调用预测函数并传递参数
    resultlist = predict_funtion(
        args.gpu_name,
        args.save_length,
        args.timesteps,
        args.ins_predict_weight,
        args.data_path,
        args.bamfilepath,
        args.outvcfpath,
        args.contigs,
        args.support
    )

    # 打印结果
    print("Prediction results:", resultlist)

if __name__ == "__main__":
    main()

# contigs = ['12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']
# # support = 11
# timesteps = 100

# # 设置随机种子
# seed = 123
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# gpu_name = '0'
# save_length = 10000000

# resultlist = predict_funtion(gpu_name, save_length, timesteps, ins_predict_weight, data_path, bamfilepath, outvcfpath, contigs, support)
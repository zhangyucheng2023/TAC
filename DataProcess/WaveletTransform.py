# 对初始数据进行小波变换

import os
import pywt
import numpy as np
from PIL import Image


def compress_data(data):
    # *****原代码仅计算了6个压力值*****
    dim1 = (data[0, 0] + data[0, 1] + data[0, 2] + data[0, 3] + data[0, 4] + data[0, 5] + data[0, 6] + data[0, 7] +
            data[0, 8] + data[0, 9] + data[0, 10] + data[0, 11]) / 12
    dim2 = (data[1, 0] + data[1, 1] + data[1, 2] + data[1, 3] + data[1, 4] + data[1, 5] + data[1, 6] + data[1, 7] +
            data[1, 8] + data[1, 9] + data[1, 10] + data[1, 11]) / 12
    # dim3 = (data[2, 0] + data[2, 1] + data[2, 2] + data[2, 3] + data[2, 4] + data[2, 5] + data[2, 6] + data[2, 7] +
    #         data[2, 8] + data[2, 9] + data[2, 10] + data[2, 11]) / 12
    # dim4 = (data[3, 0] + data[3, 1] + data[3, 2] + data[3, 3] + data[3, 4] + data[3, 5] + data[3, 6] + data[3, 7] +
    #         data[3, 8] + data[3, 9] + data[3, 10] + data[3, 11]) / 12
    dim5 = (data[0, 12] + data[0, 13] + data[0, 14]) / 3
    dim6 = (data[1, 12] + data[1, 13] + data[1, 14]) / 3
    # dim7 = (data[2, 12] + data[2, 13] + data[2, 14]) / 3
    # dim8 = (data[3, 12] + data[3, 13] + data[3, 14]) / 3
    # print(dim1.shape,dim2.shape)
    # final = np.array([dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8])
    final = np.array([dim1, dim2, dim5, dim6])
    return final


def get_wavelet(data, range):
    wavename = 'morl'
    totalscal = 601
    fc = pywt.central_frequency(wavename)
    cparam = range * fc * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)
    [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename)
    return abs(cwtmatr)


def wavelet_concate(data):
    c = []
    for i in range(4):
        if i < 2:
            r = 200
        else:
            r = 2
        wave = get_wavelet(data[i], r)
        img_wave = Image.fromarray(wave)
        out = img_wave.resize((200, 120), Image.Resampling.LANCZOS)
        out = np.array(out)
        c.append(out)
    combine_data = np.array([c[0], c[1], c[2], c[3]])
    return combine_data


if __name__ == "__main__":
    path = "./DataProcess/Data/npydata12"
    materials = os.listdir(path)
    if not os.path.exists("./DataProcess/Data/wavedata12"):
        os.mkdir("./DataProcess/Data/wavedata12")
    for i, material in enumerate(materials):
        print(i, material)
        material_path = path + '/' + material
        files = os.listdir(material_path)
        for file in files:
            print(file)
            file_path = material_path + '/' + file  # "./Data/npydata12/Aluminium/Aluminium_1.csv"
            data = np.load(file_path)
            # print(data.shape)
            c_data = compress_data(data)
            wavegraph = wavelet_concate(c_data)
            # print(wavegraph.shape)

            if not os.path.exists("./DataProcess/Data/wavedata12/" + material):
                os.makedirs("./DataProcess/Data/wavedata12/" + material)
            np.save("./DataProcess/Data/wavedata12/" + material + '/' + file, wavegraph)

# test=files[265]
# Data=np.load("./0518augmentation/"+test)
# c_data=compress_data(Data)


# data=wavelet_concate(c_data)
# print(data.shape)
# x=data[0]
# plt.imshow(x)
# plt.show()


# x=get_wavelet(data,200)
# plt.imshow(x)
# plt.show()
# img=Image.fromarray(np.uint8(x))
# out=img.resize((200, 120))
# out1=np.array(out)
# plt.imshow(out1)
# plt.show()


# t=np.arange(0,1000)
# wavename = 'cgau8'
# totalscal = 601
# fc = pywt.central_frequency(wavename)
# cparam = 200 * fc * totalscal
# scales = cparam / np.arange(totalscal, 1, -1)
# print(fc,cparam,scales)
# [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename)
# print(cwtmatr.shape)
# print(frequencies.shape)
# plt.figure(figsize=(8, 4))
# plt.subplot(211)
# plt.plot(t, data)
# plt.xlabel(u"sapmle")
# plt.ylabel(u"normalized force")
# plt.subplot(212)
# plt.contourf(t, frequencies, abs(cwtmatr))
# plt.ylabel(u"frequency(Hz)")
# plt.xlabel(u"sample")
# plt.subplots_adjust(hspace=0.4)
# plt.show()

# x=abs(cwtmatr)
# plt.imshow(x)
# plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

import lab6_kuba as lab6

##############################################################################
######   Konfiguracja       ##################################################
##############################################################################

kat = '.\\videos'  # katalog z plikami wideo
plik = "clip_1.mp4"  # nazwa pliku
ile = 15  # ile klatek odtworzyć? <0 - całość
key_frame_counter = 1  # co która klatka ma być kluczowa i nie podlegać kompresji
plot_frames = 15  # automatycznie wyrysuj wykresy
auto_pause_frames = 14  # automatycznie za pauzuj dla klatki
#subsampling = "4:1:1"  # parametry dla chroma subsampling
dzielnik = 4  # dzielnik przy zapisie różnicy
wyswietlaj_kaltki = True  # czy program ma wyświetlać klatki
ROI = [[500, 900, 500, 900]]  # wyświetlane fragmenty (można podać kilka )
RLE_code = lab6.RLE_code
RLE_decode = lab6.RLE_decode


##############################################################################
####     Kompresja i dekompresja    ##########################################
##############################################################################
class data:
    def init(self):
        self.Y = None
        self.Cb = None
        self.Cr = None


def Chroma_subsampling(L, subsampling):
    if subsampling == "4:4:4":
        L = L
    elif subsampling == "4:2:2":
        L = L[:, ::2]
    elif subsampling == "4:4:0":
        L = L[::2, ::]
    elif subsampling == "4:2:0":
        L = L[::2, ::2]
    elif subsampling == "4:1:1":
        L = L[:, ::4]
    else: # 4:1:0
        L = L[::2, ::4]
    return L

def Chroma_resampling(L, subsampling):
    if subsampling == "4:4:4":
        L = L
    elif subsampling == "4:2:2":
        L = np.repeat(L, 2, axis=1)
    elif subsampling == "4:4:0":
        L = np.repeat(L, 2, axis=0)
    elif subsampling == "4:2:0":
        L = np.repeat(L, 2, axis=0)
        L = np.repeat(L, 2, axis=1)
    elif subsampling == "4:1:1":
        L = np.repeat(L, 4, axis=1)
    else:  # 4:1:0
        L = np.repeat(L, 2, axis=0)
        L = np.repeat(L, 4, axis=1)
    return L


def frame_image_to_class(frame, subsampling):
    Frame_class = data()
    Frame_class.Y = frame[:, :, 0].astype(int)

    Frame_class.Cb = Chroma_subsampling(frame[:, :, 2].astype(int), subsampling)
    Frame_class.Cr = Chroma_subsampling(frame[:, :, 1].astype(int), subsampling)

    return Frame_class


def frame_layers_to_image(Y, Cr, Cb, subsampling):

    Cb = Chroma_resampling(Cb, subsampling)
    Cr = Chroma_resampling(Cr, subsampling)

    return np.dstack([Y, Cr, Cb]).clip(0, 255).astype(np.uint8)


def compress_KeyFrame(Frame_class, subsampling, rle):
    KeyFrame = data()

    if rle:
        KeyFrame.Y = RLE_code(Frame_class.Y)

        KeyFrame.Cb = RLE_code(Frame_class.Cb)

        KeyFrame.Cr = RLE_code(Frame_class.Cr)
    else:
        KeyFrame.Y = Chroma_subsampling(Frame_class.Y, subsampling)
        KeyFrame.Cb = Chroma_subsampling(Frame_class.Cb, subsampling)
        KeyFrame.Cr = Chroma_subsampling(Frame_class.Cr, subsampling)

    return KeyFrame


def decompress_KeyFrame(KeyFrame, subsampling, rle):
    if rle:
        Y = RLE_decode(KeyFrame.Y)

        Cb = RLE_decode(KeyFrame.Cb)

        Cr = RLE_decode(KeyFrame.Cr)
    else:
        Y = Chroma_resampling(KeyFrame.Y, subsampling)
        Cb = Chroma_resampling(KeyFrame.Cb, subsampling)
        Cr = Chroma_resampling(KeyFrame.Cr, subsampling)

    frame_image = frame_layers_to_image(Y, Cr, Cb, subsampling)
    return frame_image


def compress_not_KeyFrame(Frame_class, KeyFrame, subsampling, rle):
    Compress_data = data()

    if rle:
        Compress_data.Y = RLE_code((Frame_class.Y - RLE_decode(KeyFrame.Y)) / dzielnik)
        Compress_data.Cb = RLE_code(Frame_class.Cb - RLE_decode(KeyFrame.Cb) / dzielnik)
        Compress_data.Cr = RLE_code(Frame_class.Cr - RLE_decode(KeyFrame.Cr) / dzielnik)

    else:
        Compress_data.Cb = (Frame_class.Cb - Chroma_resampling(KeyFrame.Cb, subsampling)) / dzielnik
        Compress_data.Cr = (Frame_class.Cr - Chroma_resampling(KeyFrame.Cr, subsampling)) / dzielnik

        Compress_data.Y = Chroma_subsampling((Frame_class.Y - Chroma_resampling(KeyFrame.Y, subsampling)) / dzielnik, subsampling)
        Compress_data.Cb = Chroma_subsampling(Compress_data.Cb, subsampling)
        Compress_data.Cr = Chroma_subsampling(Compress_data.Cr, subsampling)

    return Compress_data


def decompress_not_KeyFrame(Compress_data, KeyFrame, subsampling, rle):

    if rle:
        Y = RLE_decode(KeyFrame.Y) + dzielnik * RLE_decode(Compress_data.Y)
        Cb = RLE_decode(KeyFrame.Cb) + dzielnik * RLE_decode(Compress_data.Cb)
        Cr = RLE_decode(KeyFrame.Cr) + dzielnik * RLE_decode(Compress_data.Cr)

    else:
        Y = Chroma_resampling(KeyFrame.Y, subsampling) + dzielnik * Chroma_resampling(Compress_data.Y, subsampling)
        Cb = Chroma_resampling(KeyFrame.Cb, subsampling) + dzielnik * Chroma_resampling(Compress_data.Cb, subsampling)
        Cr = Chroma_resampling(KeyFrame.Cr, subsampling) + dzielnik * Chroma_resampling(Compress_data.Cr, subsampling)

    return frame_layers_to_image(Y, Cr, Cb, subsampling)


def plotDiffrence(ReferenceFrame, DecompressedFrame, ROI, title):
    # bardzo słaby i sztuczny przykład wykorzystania tej opcji
    # przerobić żeby porównanie było dokonywane w RGB nie YCrCb i/lub zastąpić innym porównaniem
    # ROI - Region of Insert współrzędne fragmentu który chcemy przybliżyć i ocenić w formacie [w1,w2,k1,k2]
    fig, axs = plt.subplots(1, 3, sharey=True)
    fig.set_size_inches(16, 5)

    axs[0].imshow(ReferenceFrame[ROI[0]:ROI[1], ROI[2]:ROI[3]])
    axs[2].imshow(DecompressedFrame[ROI[0]:ROI[1], ROI[2]:ROI[3]])
    diff = ReferenceFrame[ROI[0]:ROI[1], ROI[2]:ROI[3]].astype(float) - \
           DecompressedFrame[ROI[0]:ROI[1],
           ROI[2]:ROI[3]].astype(float)

    print(np.min(diff), np.max(diff))
    axs[1].imshow(diff, vmin=np.min(diff), vmax=np.max(diff))
    axs[1].set_title(title)


##############################################################################
####     Głowna pętla programu      ##########################################
##############################################################################

cap = cv2.VideoCapture(kat + '\\' + plik)

if ile < 0:
    ile = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cv2.namedWindow('Normal Frame')
cv2.namedWindow('Decompressed Frame')

compression_information = np.zeros((3, ile))

rle = False

def zad1():
    for subsampling in ["4:4:4", "4:2:2", "4:4:0", "4:2:0", "4:1:1", "4:1:0"]:
        for dzielnik in [1, 2, 4]:
            print(f"{subsampling}, {dzielnik}")
            for i in range(ile):
                ret, frame = cap.read()
                if wyswietlaj_kaltki:
                    cv2.imshow('Normal Frame', frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                Frame_class = frame_image_to_class(frame, subsampling)
                if (i % key_frame_counter) == 0:  # pobieranie klatek kluczowych
                    KeyFrame = compress_KeyFrame(Frame_class, subsampling, rle)
                    if rle:
                        cY = KeyFrame.Y[2]
                        cCb = KeyFrame.Cb[2]
                        cCr = KeyFrame.Cr[2]
                    else:
                        cY = KeyFrame.Y
                        cCb = KeyFrame.Cb
                        cCr = KeyFrame.Cr
                    Decompresed_Frame = decompress_KeyFrame(KeyFrame, subsampling, rle)
                else:  # kompresja
                    Compress_data = compress_not_KeyFrame(Frame_class, KeyFrame, subsampling, rle)
                    if rle:
                        cY = Compress_data.Y[2]
                        cCb = Compress_data.Cb[2]
                        cCr = Compress_data.Cr[2]
                    else:
                        cY = Compress_data.Y
                        cCb = Compress_data.Cb
                        cCr = Compress_data.Cr
                    Decompresed_Frame = decompress_not_KeyFrame(Compress_data, KeyFrame, subsampling, rle)

                compression_information[0, i] = (frame[:, :, 0].size - cY.size) / frame[:, :, 0].size
                compression_information[1, i] = (frame[:, :, 0].size - cCb.size) / frame[:, :, 0].size
                compression_information[2, i] = (frame[:, :, 0].size - cCr.size) / frame[:, :, 0].size

                if wyswietlaj_kaltki:
                    cv2.imshow('Decompressed Frame', cv2.cvtColor(Decompresed_Frame, cv2.COLOR_YCrCb2BGR))

                if np.any(plot_frames == i):  # rysuj wykresy
                    for r in ROI:
                        frame = cv2.cvtColor(frame, cv2.COLOR_YCrCb2BGR)
                        Decompresed_Frame = cv2.cvtColor(Decompresed_Frame, cv2.COLOR_YCrCb2BGR)
                        plotDiffrence(frame, Decompresed_Frame, r, f"subsampling - {subsampling}, dzielnik - {dzielnik}")
                if np.any(auto_pause_frames == i):
                    cv2.waitKey(-1)  # wait until any key is pressed

                k = cv2.waitKey(1) & 0xff

                if k == ord('q'):
                    break
                elif k == ord('p'):
                    cv2.waitKey(-1)  # wait until any key is pressed

    plt.show()

def zad2(subsampling, dzielnik, key_frame):
    wyswietlaj_kaltki = False
    rle = True
    for i in range(ile):
        print(i)
        ret, frame = cap.read()
        if wyswietlaj_kaltki:
            cv2.imshow('Normal Frame', frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        Frame_class = frame_image_to_class(frame, subsampling)
        if (i % key_frame) == 0:  # pobieranie klatek kluczowych
            KeyFrame = compress_KeyFrame(Frame_class, subsampling, rle)
            if rle:
                cY = KeyFrame.Y[2]
                cCb = KeyFrame.Cb[2]
                cCr = KeyFrame.Cr[2]
            else:
                cY = KeyFrame.Y
                cCb = KeyFrame.Cb
                cCr = KeyFrame.Cr
            Decompresed_Frame = decompress_KeyFrame(KeyFrame, subsampling, rle)
        else:  # kompresja
            Compress_data = compress_not_KeyFrame(Frame_class, KeyFrame, subsampling, rle)
            if rle:
                cY = Compress_data.Y[2]
                cCb = Compress_data.Cb[2]
                cCr = Compress_data.Cr[2]
            else:
                cY = Compress_data.Y
                cCb = Compress_data.Cb
                cCr = Compress_data.Cr
            Decompresed_Frame = decompress_not_KeyFrame(Compress_data, KeyFrame, subsampling, rle)

        compression_information[0, i] = (frame[:, :, 0].size - cY.size) / frame[:, :, 0].size
        compression_information[1, i] = (frame[:, :, 0].size - cCb.size) / frame[:, :, 0].size
        compression_information[2, i] = (frame[:, :, 0].size - cCr.size) / frame[:, :, 0].size
        if wyswietlaj_kaltki:
            cv2.imshow('Decompressed Frame', cv2.cvtColor(Decompresed_Frame, cv2.COLOR_YCrCb2BGR))

    plt.figure()
    plt.plot(np.arange(0, ile), compression_information[0, :] * 100)
    plt.plot(np.arange(0, ile), compression_information[1, :] * 100)
    plt.plot(np.arange(0, ile), compression_information[2, :] * 100)
    pl.legend(['Y', 'Cr', 'Cb'])
    plt.title("File:{}, subsampling={}, divider={}, KeyFrame={} ".format(plik, subsampling, dzielnik, key_frame))
    plt.show()

def main():
    # zad1()
    for f in [2, 4, 6, 8]:
        f
        zad2("4:2:0", 4, f)


if __name__ == "__main__":
    main()
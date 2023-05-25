import cv2
import numpy as np
import matplotlib.pyplot as plt
from lab6 import RLE

##############################################################################
######   Konfiguracja       ##################################################
##############################################################################

kat = '.\\videos'  # katalog z plikami wideo
plik = "clip_4.mp4"  # nazwa pliku
ile = 15  # ile klatek odtworzyć? <0 - całość
key_frame_counter = 1  # co która klatka ma być kluczowa i nie podlegać kompresji
plot_frames = 14  # automatycznie wyrysuj wykresy
auto_pause_frames = 14  # automatycznie za pauzuj dla klatki
# subsampling = "4:4:4"  # parametry dla chroma subsampling
# dzielnik = 4  # dzielnik przy zapisie różnicy
wyswietlaj_kaltki = False  # czy program ma wyświetlać klatki
ROI = [[500, 900, 500, 900]]  # wyświetlane fragmenty (można podać kilka )


##############################################################################
####     Kompresja i dekompresja    ##########################################
##############################################################################
class data:
    def init(self):
        self.Y = None
        self.Cb = None
        self.Cr = None


def Chroma_subsampling(L, subsampling):
    match subsampling:
        case "4:4:4":
            L = L
        case "4:4:0":
            L = L[::2, ::]
        case "4:2:2":
            L = L[:, ::2]
        case "4:2:0":
            L = L[::2, ::2]
        case "4:1:1":
            L = L[:, ::4]
        case _:
            L = L[::2, ::4]
    return L


def Chroma_resampling(L, subsampling):
    match subsampling:
        case "4:4:4":
            L = L
        case "4:4:0":
            L = np.repeat(L, 2, axis=0)
        case "4:2:2":
            L = np.repeat(L, 2, axis=1)
        case "4:2:0":
            L = np.repeat(L, 2, axis=0)
            L = np.repeat(L, 2, axis=1)
        case "4:1:1":
            L = np.repeat(L, 4, axis=1)
        case _:
            L = np.repeat(L, 4, axis=1)
            L = np.repeat(L, 2, axis=0)
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


def compress_KeyFrame(Frame_class, method='rle', subsampling=None):
    KeyFrame = data()
    match method:
        case 'rle':
            KeyFrame.Y = RLE(Frame_class.Y).encode()
            KeyFrame.Cb = RLE(Frame_class.Cb).encode()
            KeyFrame.Cr = RLE(Frame_class.Cr).encode()
        case 'subsampling':
            if subsampling is not None:
                KeyFrame.Y = Chroma_subsampling(Frame_class.Y, subsampling)
                KeyFrame.Cb = Chroma_subsampling(Frame_class.Cb, subsampling)
                KeyFrame.Cr = Chroma_subsampling(Frame_class.Cr, subsampling)
            else:
                raise ValueError('Nie podano parametrów subsamplingu')
    return KeyFrame


def decompress_KeyFrame(KeyFrame, method='rle', subsampling=None):
    match method:
        case 'rle':
            Y = RLE(KeyFrame.Y).decode(KeyFrame.Y)
            Cb = RLE(KeyFrame.Cb).decode(KeyFrame.Cb)
            Cr = RLE(KeyFrame.Cr).decode(KeyFrame.Cr)
        case 'subsampling':
            if subsampling is not None:
                Y = Chroma_resampling(KeyFrame.Y, subsampling)
                Cb = Chroma_resampling(KeyFrame.Cb, subsampling)
                Cr = Chroma_resampling(KeyFrame.Cr, subsampling)
            else:
                raise ValueError('Nie podano parametrów subsamplingu')
        case _:
            raise ValueError('Nie prawidłowa metoda dekompresji')

    frame_image = frame_layers_to_image(Y, Cr, Cb, subsampling)
    return frame_image


def compress_not_KeyFrame(Frame_class, KeyFrame, dzielnik=4, method='rle', subsampling=None):
    Compress_data = data()
    match method:
        case 'rle':
            Compress_data.Y = RLE((Frame_class.Y - RLE(KeyFrame.Y).decode(KeyFrame.Y)) / dzielnik).encode()
            Compress_data.Cb = RLE((Frame_class.Cb - RLE(KeyFrame.Cb).decode(KeyFrame.Cb)) / dzielnik).encode()
            Compress_data.Cr = RLE((Frame_class.Cr - RLE(KeyFrame.Cr).decode(KeyFrame.Cr)) / dzielnik).encode()
        case 'subsampling':
            if subsampling is not None:
                Compress_data.Cb = (Frame_class.Cb - Chroma_resampling(KeyFrame.Cb, subsampling)) / dzielnik
                Compress_data.Cr = (Frame_class.Cr - Chroma_resampling(KeyFrame.Cr, subsampling)) / dzielnik
                Compress_data.Y = Chroma_subsampling((Frame_class.Y - Chroma_resampling(KeyFrame.Y, subsampling)) / dzielnik, subsampling)
                Compress_data.Cb = Chroma_subsampling(Compress_data.Cb, subsampling)
                Compress_data.Cr = Chroma_subsampling(Compress_data.Cr, subsampling)
    return Compress_data


def decompress_not_KeyFrame(Compress_data, KeyFrame, dzielnik=4, method='rle', subsampling=None):
    match method:
        case 'rle':
            Y = RLE(KeyFrame.Y).decode(KeyFrame.Y) + dzielnik * RLE(Compress_data.Y).decode(Compress_data.Y)
            Cb = RLE(KeyFrame.Cb).decode(KeyFrame.Cb) + dzielnik * RLE(Compress_data.Cb).decode(Compress_data.Cb)
            Cr = RLE(KeyFrame.Cr).decode(KeyFrame.Cr) + dzielnik * RLE(Compress_data.Cr).decode(Compress_data.Cr)
        case 'subsampling':
            if subsampling is not None:
                Y = Chroma_resampling(KeyFrame.Y, subsampling) + dzielnik * Chroma_resampling(Compress_data.Y, subsampling)
                Cb = Chroma_resampling(KeyFrame.Cb, subsampling) + dzielnik * Chroma_resampling(Compress_data.Cb, subsampling)
                Cr = Chroma_resampling(KeyFrame.Cr, subsampling) + dzielnik * Chroma_resampling(Compress_data.Cr, subsampling)
            else:
                raise ValueError('Nie podano parametrów subsamplingu')
        case _:
            raise ValueError('Nie prawidłowa metoda dekompresji')
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
    fig.suptitle(title, fontsize=16)


##############################################################################
####     Głowna pętla programu      ##########################################
##############################################################################

cap = cv2.VideoCapture(kat + '\\' + plik)

if ile < 0:
    ile = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cv2.namedWindow('Normal Frame')
cv2.namedWindow('Decompressed Frame')

compression_information = np.zeros((3, ile))


def with_rle():
    subsamplings = ["4:4:4", "4:4:0", "4:2:2", "4:2:0", "4:1:1", "4:1:0"]
    divs = [1, 2, 4]
    for subsampling in subsamplings:
        for div in divs:
            print(f"Wykonuje dla subsampling={subsampling} i dzielnika={div}")
            for i in range(ile):
                ret, frame = cap.read()
                if wyswietlaj_kaltki:
                    cv2.imshow('Normal Frame', frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                Frame_class = frame_image_to_class(frame, subsampling)
                if (i % key_frame_counter) == 0:  # pobieranie klatek kluczowych
                    KeyFrame = compress_KeyFrame(Frame_class, 'rle', subsampling)
                    cY = KeyFrame.Y
                    cCb = KeyFrame.Cb
                    cCr = KeyFrame.Cr
                    Decompresed_Frame = decompress_KeyFrame(KeyFrame, 'rle', subsampling)
                else:  # kompresja
                    Compress_data = compress_not_KeyFrame(Frame_class, KeyFrame, 'rle', subsampling)
                    cY = Compress_data.Y
                    cCb = Compress_data.Cb
                    cCr = Compress_data.Cr
                    Decompresed_Frame = decompress_not_KeyFrame(Compress_data, KeyFrame, 'rle', subsampling)

                compression_information[0, i] = (frame[:, :, 0].size - cY.size) / frame[:, :, 0].size
                compression_information[1, i] = (frame[:, :, 0].size - cCb.size) / frame[:, :, 0].size
                compression_information[2, i] = (frame[:, :, 0].size - cCr.size) / frame[:, :, 0].size
                if wyswietlaj_kaltki:
                    cv2.imshow('Decompressed Frame', cv2.cvtColor(Decompresed_Frame, cv2.COLOR_YCrCb2BGR))

                if np.any(plot_frames == i):  # rysuj wykresy
                    for r in ROI:
                        frame = cv2.cvtColor(frame, cv2.COLOR_YCrCb2BGR)
                        Decompresed_Frame = cv2.cvtColor(Decompresed_Frame, cv2.COLOR_YCrCb2BGR)
                        plotDiffrence(frame, Decompresed_Frame, r, f"subsampling={subsampling}, dzielnik={div}")
                if np.any(auto_pause_frames == i):
                    cv2.waitKey(-1)  # wait until any key is pressed

                k = cv2.waitKey(1) & 0xff

                if k == ord('q'):
                    break
                elif k == ord('p'):
                    cv2.waitKey(-1)  # wait until any key is pressed
    plt.show()


def without_rle():
    subsamplings = ["4:4:4", "4:4:0", "4:2:2", "4:2:0", "4:1:1", "4:1:0"]
    divs = [1, 2, 4]
    for subsampling in subsamplings:
        for div in divs:
            print(f"Wykonuje dla subsampling={subsampling} i dzielnika={div}")
            for i in range(ile):
                ret, frame = cap.read()
                if wyswietlaj_kaltki:
                    cv2.imshow('Normal Frame', frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                Frame_class = frame_image_to_class(frame, subsampling)
                if (i % key_frame_counter) == 0:  # pobieranie klatek kluczowych
                    KeyFrame = compress_KeyFrame(Frame_class, 'subsampling', subsampling)
                    cY = KeyFrame.Y
                    cCb = KeyFrame.Cb
                    cCr = KeyFrame.Cr
                    Decompresed_Frame = decompress_KeyFrame(KeyFrame, 'subsampling', subsampling)
                else:  # kompresja
                    Compress_data = compress_not_KeyFrame(Frame_class, KeyFrame, div, 'subsampling', subsampling)
                    cY = Compress_data.Y
                    cCb = Compress_data.Cb
                    cCr = Compress_data.Cr
                    Decompresed_Frame = decompress_not_KeyFrame(Compress_data, KeyFrame, div, 'subsampling', subsampling)

                compression_information[0, i] = (frame[:, :, 0].size - cY.size) / frame[:, :, 0].size
                compression_information[1, i] = (frame[:, :, 0].size - cCb.size) / frame[:, :, 0].size
                compression_information[2, i] = (frame[:, :, 0].size - cCr.size) / frame[:, :, 0].size
                if wyswietlaj_kaltki:
                    cv2.imshow('Decompressed Frame', cv2.cvtColor(Decompresed_Frame, cv2.COLOR_YCrCb2BGR))

                if np.any(plot_frames == i):  # rysuj wykresy
                    for r in ROI:
                        frame = cv2.cvtColor(frame, cv2.COLOR_YCrCb2BGR)
                        Decompresed_Frame = cv2.cvtColor(Decompresed_Frame, cv2.COLOR_YCrCb2BGR)
                        plotDiffrence(frame, Decompresed_Frame, r, f"subsampling={subsampling}, dzielnik={div}")
                if np.any(auto_pause_frames == i):
                    cv2.waitKey(-1)  # wait until any key is pressed

                k = cv2.waitKey(1) & 0xff

                if k == ord('q'):
                    break
                elif k == ord('p'):
                    cv2.waitKey(-1)  # wait until any key is pressed
    plt.show()


def zad2(key_frame):
    for i in range(ile):
        print(i)
        ret, frame = cap.read()
        if wyswietlaj_kaltki:
            cv2.imshow('Normal Frame', frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        Frame_class = frame_image_to_class(frame, "4:2:0")
        if (i % key_frame) == 0:  # pobieranie klatek kluczowych
            KeyFrame = compress_KeyFrame(Frame_class, 'rle', "4:2:0")
            cY = KeyFrame.Y
            cCb = KeyFrame.Cb
            cCr = KeyFrame.Cr
            Decompresed_Frame = decompress_KeyFrame(KeyFrame, 'rle', "4:2:0")
        else:  # kompresja
            Compress_data = compress_not_KeyFrame(Frame_class, KeyFrame, 4, 'rle', "4:2:0")
            cY = Compress_data.Y
            cCb = Compress_data.Cb
            cCr = Compress_data.Cr
            Decompresed_Frame = decompress_not_KeyFrame(Compress_data, KeyFrame, 4, 'rle', "4:2:0")

        compression_information[0, i] = (frame[:, :, 0].size - cY.size) / frame[:, :, 0].size
        compression_information[1, i] = (frame[:, :, 0].size - cCb.size) / frame[:, :, 0].size
        compression_information[2, i] = (frame[:, :, 0].size - cCr.size) / frame[:, :, 0].size
        if wyswietlaj_kaltki:
            cv2.imshow('Decompressed Frame', cv2.cvtColor(Decompresed_Frame, cv2.COLOR_YCrCb2BGR))

    plt.figure()
    plt.plot(np.arange(0, ile), compression_information[0, :], label="Y")
    plt.plot(np.arange(0, ile), compression_information[1, :], label="Cb")
    plt.plot(np.arange(0, ile), compression_information[2, :], label="Cr")
    plt.legend()
    plt.title(f"File:{plik}, subsampling=4:2:0, dzielnik=4, KeyFrame={key_frame}")
    plt.show()


if __name__ == '__main__':
    # with_rle()
    # without_rle()
    key_frames = [2, 4, 6, 8]
    for key in key_frames:
        zad2(key)

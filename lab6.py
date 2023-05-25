import os

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

big_test = np.array([
    np.array([1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1]),
    np.array([1, 2, 3, 1, 2, 3, 1, 2, 3]),
    np.array([5, 1, 5, 1, 5, 5, 1, 1, 5, 5, 1, 1, 5]),
    np.array([-1, -1, -1, -5, -5, -3, -4, -2, 1, 2, 2, 1]),
    np.zeros((1, 520)),
    np.arange(0, 521, 1),
    np.eye(7),
    np.dstack([np.eye(7), np.eye(7), np.eye(7)]),
    np.ones((1, 1, 1, 1, 1, 1, 10)),
], dtype=np.ndarray)


class RLE:
    def __init__(self, picture: np.ndarray):
        self.matrix: np.ndarray = picture

    def encode(self) -> list:
        """
        Funkcja koduje obrazek metodą RLE. Polega ona na zliczaniu powtarzających się wartości w obrazku i zapisywaniu
        ich w postaci par (liczba powtórzeń, wartość).

        Returns:
            list: Lista zawierająca informacje o obrazku oraz sam obrazek zakodowany. Pierwszy wymiar [0] zawiera
            zakodowany obrazek, drugi wymiar [1] zawiera informacje o ilości wymiarów obrazka, a trzeci wymiar [2]
            zawiera informacje o oryginalnym rozmiarze obrazka.

        """
        new_img: list = []
        data: list = []
        new_img.append(data)
        new_img += [len(self.matrix.shape)]  # Ile wymiarów new_img[1]
        new_img += list(self.matrix.shape)  # Rozmiar new_img[2]
        if self.matrix.ndim > 1:
            self.matrix = self.matrix.flatten()

        count: int = 1
        for i in tqdm(range(len(self.matrix) - 1)):
            if self.matrix[i] == self.matrix[i + 1]:
                count += 1
            else:
                data.append(count)
                data.append(self.matrix[i])
                count = 1
        data.append(count)
        data.append(self.matrix[-1])

        return np.asarray(new_img)

    def decode(self, encoded: list) -> np.ndarray:
        """
        Funkcja dekoduje obrazek zakodowany metodą RLE. Polega ona na odtworzeniu obrazka na podstawie informacji o
        ilości powtórzeń oraz wartości. W przypadku gdy liczba powtórzeń jest większa niż 1, to wartość jest dodawana
        do listy tyle razy ile wynosi liczba powtórzeń.

        Args:
            encoded: Zakodowany obrazek wraz z informacjami o nim.

        Returns:
            np.ndarray: Oryginalny obrazek.

        """
        data = encoded[0]
        dimensions = encoded[1]
        shape = encoded[2:]
        img = []
        for i in tqdm(range(len(data))):
            if i % 2 == 0:
                img += [data[i + 1]] * data[i]
        img = np.asarray(img)
        img = img.reshape(shape)
        return img


class ByteRun:
    def __init__(self, picture: np.ndarray):
        self.matrix: np.ndarray = picture

    def encode(self):
        """
        Funkcja koduje obrazek metodą ByteRun. Jeśli następujące po sobie wartości są takie same, to zapisywana jest
        informacja w postaci (-liczba powtórzeń, wartość). Jeśli następujące po sobie wartości są różne, to informacja
        jest zapisywana w postaci (liczba wartości, wartość1, wartość2, ...).

        Returns:
            list: Lista zawierająca informacje o obrazku oraz sam obrazek zakodowany. Pierwszy wymiar [0] zawiera
            zakodowany obrazek, drugi wymiar [1] zawiera informacje o ilości wymiarów obrazka, a trzeci wymiar [2]
            zawiera informacje o oryginalnym rozmiarze obrazka.

        """
        new_img: list = []
        data: list = []
        ramka: list = []
        new_img.append(data)
        new_img += [len(self.matrix.shape)]
        new_img += list(self.matrix.shape)
        if self.matrix.ndim > 1:
            self.matrix = self.matrix.flatten()

        count: int = 1
        for i in tqdm(range(len(self.matrix) - 1)):
            if self.matrix[i] == self.matrix[i + 1]:
                if len(ramka) > 0:
                    data.append(len(ramka))
                    data.extend(ramka)
                    ramka = []
                count += 1
            else:
                if count == 1:
                    ramka.append(self.matrix[i])
                else:
                    data.append(-count)
                    data.append(self.matrix[i])
                    count = 1
        if count == 1:
            ramka.append(self.matrix[-1])
            data.append(len(ramka))
            data.extend(ramka)
        else:
            data.append(-count)
            data.append(self.matrix[-1])

        return new_img

    def decode(self, encoded: list) -> np.ndarray:
        """
        Funkcja dekoduje obrazek zakodowany metodą ByteRun.

        Args:
            encoded: Zakodowany obrazek wraz z informacjami o nim.

        Returns:
            np.ndarray: Oryginalny obrazek.

        """
        data = encoded[0]
        dimensions = encoded[1]
        shape = encoded[2:]
        img = []
        i = 0
        while i < len(data):
            if data[i] < 0:
                img += [data[i + 1]] * (-data[i])
                i += 2
            else:
                img.extend(data[i + 1:i + data[i] + 1])
                i += data[i] + 1
        img = np.asarray(img)
        img = img.reshape(shape)
        return img


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, np.ndarray):
        size = obj.nbytes
    elif isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


if __name__ == '__main__':
    print("Co chcesz zrobić?\n"
          "1. Zbadać przypadki testowe\n"
          "2. Zbadać własny obrazek\n")
    choice = input("Twój wybór: ")
    match choice:
        case '1':
            for test in big_test:
                os.system("cls")
                print(f"Test {test}")
                rle = RLE(test)
                encoded_rle = rle.encode()
                decoded_rle = rle.decode(encoded_rle)
                print(f"Zakodowany obrazek: {encoded_rle}")
                print(f"Odkodowany obrazek: {decoded_rle}")
                print(f"Czy obrazki są takie same? {np.array_equal(test, decoded_rle)}")
                input("Naciśnij enter aby przejść do testu ByteRun")
                br = ByteRun(test)
                encoded_br = br.encode()
                decoded_br = br.decode(encoded_br)
                print(f"Zakodowany obrazek: {encoded_br}")
                print(f"Odkodowany obrazek: {decoded_br}")
                print(f"Czy obrazki są takie same? {np.array_equal(test, decoded_br)}")
                input("Naciśnij enter aby przejść do następnego testu")

        case '2':
            os.system("cls")
            files = os.listdir("./")
            for file in files:
                print(file)
            file_name = input("Podaj nazwę pliku: ")
            os.system("cls")
            img = plt.imread(file_name)
            encoded_rle = RLE(img).encode()
            decoded_rle = RLE(img).decode(encoded_rle)
            encoded_br = ByteRun(img).encode()
            decoded_br = ByteRun(img).decode(encoded_br)
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.title("Kolorowe zdjęcie", fontsize=20)
            plt.text(0, 100,
                     f"RLE: stopień kompresji: {get_size(img) / get_size(encoded_rle)}\n"
                     f"ByteRun: stopień kompresji: {get_size(img) / get_size(encoded_br)}"
                     f"\nCzy obrazki są takie same? {np.array_equal(img, decoded_rle)}",
                     fontsize=20,
                     bbox=dict(facecolor='red', alpha=0.5))
            plt.show()

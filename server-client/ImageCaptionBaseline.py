import random
from _warnings import warn
import pickle
import numpy as np
import json
from matplotlib import pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from tqdm.notebook import tqdm
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import DataLoader, Dataset
import cv2



class Imagecaption(object):
    def __init__(self):
        """Constructor"""

    # загружаем датасет
    img_codes = np.load('data/image_codes.npy')
    captions = json.load(open('data/captions_tokenized.json'))



    def train_tokenizer(captions):
        # инициализируем токенайзер используя сгенерированный словарь
        tokenizer = BertWordPieceTokenizer('captions-vocab.txt')
        tokenizer.enable_truncation(max_length=16)
        return tokenizer
    tokenizer = train_tokenizer(captions)
    """ Создадим датасет"""
    class ExtDataset(Dataset):
        def __init__(self, images, captions, tokenizer):
            self.images = images
            self.captions = captions
            self.tokenizer = tokenizer

        # возвращает кол-во примеров в датасете
        def __len__(self):
            return len(self.images)

        # по индексу возвращает словарь, в котором наше описани и изображение в векторизированном виде
        def __getitem__(self, idx):
            # по индексу возьмем вектор изображения
            image = self.images[idx].tolist()
            # по индексу возьмем описание, будем брать первое описание из 5 доступных
            sample = self.captions[idx][0]
            # токенизируем описание ['the dog is sleeping'] >>> ['the', 'dog', 'is', 'sleep', 'ing']
            # вызвав метод ids: ['the', 'dog', 'is', 'sleep', 'ing'] >>> [100, 8655, 12, 96, 888, 64, 101]
            # 100: обозначает токен начала предложения [CLS]
            # 101: обозначает токен конца предложения [SEP]
            caption = self.tokenizer.encode(sample).ids

            return {'inputs': image, 'outputs': caption}

    """ У нас все предложения имеют разную длину, нам нужно в нейросеть подавать батчи одинаковой размерности, для этого нам нужно знать максимальную длину предложения, чтобы все вектора добить паддингом до максимальной длины.Т.к. мы используем архитектуру RNN, по своемпостроению она медленная, и если мы будем каждый раз обрабатывать вектор максимальной длины, то обучение будет слишком долгим.Чтобы решить эту проблему, мы посмотрим распределение длины описаний и возьмем минимально возможную длину, которая охватывает максимальное количество примеров. Описания большей длины будут обрезаны."""

    max_len = max(len(sample.split()) for samples in captions for sample in samples)

    # посчитаем количество слов в каждом описании
    distribution = [len(sample.split()) for samples in captions for sample in samples]

    def collate_fn(dataset, max_len=16, image_dim=2048):
        # укажем максимальную длину, не забудем про спец. токены.
        max_len = max_len + 2 # bos & eos

        # инициализируем две матрицы состоящих из нулей, в которые будем добавлять вектора описаний и изображений
        # new_inputs нулевая матрица для изображений, размерность [кол-во примеров x размерность вектора изображения]
        new_inputs = torch.zeros((len(dataset), image_dim), dtype=torch.float)
        # new_outputs нулевая матрица для описаний, размерность [кол-во примеров x макс. длина вектора, кот-ю мы определили]
        new_outputs = torch.zeros((len(dataset), max_len), dtype=torch.long)
        # пробежимся по всему датасету
        # sample: {'inputs': image, 'outputs': caption} как определили в class ExtDataset
        for i, sample in enumerate(dataset):
            # берем i-ю строку и добавляем в неё вектор изображения [0, 0, ..., 0] >>> [8.65001, 1.0012, ..., 0]
            new_inputs[i, :] += np.array(sample['inputs'])
            # берем i-ю строку и добавляем в неё вектор описания [0, 0, ..., 0] >>> [100, 820, ..., 0]
            new_outputs[i, :len(sample['outputs'])] += np.array(sample['outputs'])
        # заполненные матрицы возвращаем
        return {'input_ids': new_inputs, 'outputs': new_outputs}

    """ Создадим наш энкодер, который будет брать вектор изображения и кодировать его в размерность скрытого состояния нашей RNN модели. Далее выход энкодера будем подавать на вход RNN системе в качестве скрытого состояния."""

    class Encoder(nn.Module):
        # hid_dim: вектор скрытого состояния
        # n_layers: кол-во слоёв в RNN архитектуре, мы планируем использовать LSTM с 2-мя слоями
        # для этого нам нужно подготовить правильную размерность для скрытых состояний [n_layres, bs, hid_dim]
        # cnn_feature_dim: размерность извлеченного CNN моделью, вектора с фичами из изображения
        def __init__(self, hid_dim=512, n_layers=2, cnn_feature_dim=2048):
            super().__init__()
            self.n_layers = n_layers
            # инициализируем линейные слои
            self.cnn2h0 = nn.Linear(cnn_feature_dim, hid_dim)
            self.cnn2c0 = nn.Linear(cnn_feature_dim, hid_dim)

        def forward(self, image_vectors):
            # input
            # image_vectors: [bs, cnn_feature_dim]
            # output
            # cnn2h0: [bs, hid_dim]
            # unsqueeze(0).repeat: [n_layers, bs, hid_dim]
            initial_hid = self.cnn2h0(image_vectors).unsqueeze(0).repeat(self.n_layers, 1, 1)
            initial_cell = self.cnn2c0(image_vectors).unsqueeze(0).repeat(self.n_layers, 1, 1)
            return initial_hid, initial_cell

    """ Создадим модель, которая на вход будет принимать вектор с фичами извлеченными с помощью CNN из изображения, а выдавать логиты, распределение на слова в словаре."""

    class CaptionNet(nn.Module):
        # emb_dim: размерность вектора слова
        # hid_dim: размерность скрытого состояния RNN
        # n_layers: кол-во слоёв в RNN
        # cnn_feature_dim: размерность извлеченного CNN моделью, вектора с фичами из изображения
        # vocab_size: размер словаря
        def __init__(self, emb_dim=256, hid_dim=512, n_layers=2, cnn_feature_dim=2048, dropout=0.3, vocab_size=30000):
            super().__init__()
            self.vocab_size = vocab_size
            # создадим матрицу хранящую эмбеддинги (вектора) слов
            self.embedding = nn.Embedding(vocab_size, emb_dim)

            # инициализируем lstm модель
            self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers)

            # инициализируем линейный слой для получения логитов
            # hid_dim * 2: расширение с последующим сужением вектора признаков используется в BERT, даёт лучший результат
            self.fc1 = nn.Linear(hid_dim, hid_dim * 2)
            self.fc2 = nn.Linear(hid_dim * 2 , vocab_size)
            self.dropout = nn.Dropout(dropout)

        def forward(self, captions_ix, initial_hid, initial_cell):
            # [bs] >>> [1, bs]
            captions_ix = captions_ix.unsqueeze(0)
            # [1, bs, emb_dim]
            captions_emb = self.embedding(captions_ix)

            captions_emb = self.dropout(captions_emb)

            # outputs: [1, bs, hid_dim]
            # hidden and cell: [n_layers, bs, hid_dim]
            outputs, (hidden, cell) = self.lstm(captions_emb, (initial_hid, initial_cell)) # shape: [batch, caption_length, lstm_units]

            # logits: [bs, vocab_size]
            logits = self.fc2(F.relu( self.fc1(outputs.squeeze()) ))

            return logits, hidden, cell

    """ Соберем нашу модель из энкодера и декодера. Будем использовать тренировку методом Teacher forcing. В каждый момент времени t, когда модель сгенерировала токен, будем определять, на следующий шаг t+1 отдавать модели сгенерированный ею токен или правильный (как в описании)."""

    class ImgCap(nn.Module):
        def __init__(self, encoder, decoder, device, max_len,tokenizer):
            super().__init__()

            self.encoder = encoder
            self.decoder = decoder
            self.device = device
            self.max_len = max_len
            self.vocab_size = self.decoder.vocab_size
            self.tokenizer = tokenizer

        def forward(self, image_vectors, captions, teacher_forcing_ratio = 0.5):
            # input
            # image_vectors: [bs, image_vec]
            # captions: [len_seq, bs]

            # запишем размер батча и длину последовательности
            batch_size = image_vectors.shape[0]
            seq_len = captions.shape[0]

            # инициализируем тензор с нулевыми матрицами, куда будем записывать сгенерированные описания
            # [seq_len, bs, vocab_size]
            outputs = torch.zeros(seq_len, batch_size, self.vocab_size).to(self.device)
            # hidden and cell: [n_layers, bs, hid_dim]
            hidden, cell = self.encoder(image_vectors)
            # возьмем первые токены [CLS] из всех описаний в батче, чтобы отдать для генерации описания нашей RNN
            # input: [bs]
            input = captions[0]
            # будем генерировать токены исходя из длины последовательности (макс. длина, которую мы определили выше в def collate_fn)
            for t in range(1, seq_len):
                # output: [bs, vocab_size]
                # hidden and cell: [n_layers, bs, hid_dim]
                output, hidden, cell = self.decoder(input, hidden, cell)
                # outputs[t]: [bs, vocab_size]
                outputs[t] = output
                # random.random() - random number from 0 to 1
                teacher_force = random.random() < teacher_forcing_ratio
                # выберем самое вероятное слово из распределения для каждого описания в батче
                top1 = output.max(1)[1]
                # определяем, будем использовать сгененированные моделью токены или отдадим им правильные
                input = (captions[t] if teacher_force else top1)
            return outputs

        def generate_one_example(self, image, inception, name):
            # image: [299, 299, 3] -> [width, height, channel]
            # inception: модель CNN для извлечения фич из изображения
            # [width, height, channel] >> [channel, width, height]
            image = np.array(cv2.resize(image,(299,299))).astype('float32') / 255.
            image = torch.tensor(image.transpose([2, 0, 1]), dtype=torch.float32)
            # извлечем фичи из изображения, нам понадобится вектор vectors_neck
            vectors_8x8, vectors_neck, logits = inception(image[None])
            outputs = []
            # поместим данные на gpu
            image_vectors = vectors_neck.to(self.device)
            #получим скрытое состояние из RNN
            hidden, cell = self.encoder(image_vectors)

            # в токенайзере возьмем id токена означающего начало описания
            input = torch.tensor([self.tokenizer.token_to_id('[CLS]')]).to(self.device)
            # здесь вероятно нужно self.max_len, нужно уточнить
            for t in range(1, self.max_len):
                # output: [bs=1, vocab_size]
                output, hidden, cell = self.decoder(input, hidden, cell)
                # из распределения слов в словаре возьмем токен с самым высоким значением
                top1 = output.max(0)[1]
                outputs.append(top1)
                # добавим размерность, т.к. decoder принимает данные с размерностью [1, bs]
                input = (top1.unsqueeze(0))

            # у токенайзера возьмем id токена, означающего конец описания
            EOS_IDX = self.tokenizer.token_to_id('[SEP]')
            res = []
            # возьмем последовательностей токенов и сгенерируем описание [100, 97893, 347, 735, 101] >>> [the dog is sleeping]
            res = ""
            for t in outputs:
                if t.item() != EOS_IDX:
                    word = self.tokenizer.id_to_token(t.item())
                    if word in ['person','man','woman']:
                        res+=name + ' '
                    else:
                        res+= word + ' '
                else:
                    break
            return res

    #@title Set parameters { vertical-output: true, display-mode: "form" }

    lr = 1e-3 #@param
    emb_dim = 128 #@param
    hid_dim = 256 #@param
    n_layers = 2 #@param
    dropout = 0.3 #@param
    batch_size = 300 #@param
    num_epochs = 20 #@param

    clip = 5
    max_len = 18
    vocab_size = 30000
    cnn_feature_dim = 2048

    device = 'cpu'
    PAD_IDX = tokenizer.token_to_id('[PAD]')
    """ Инициализируем нашу модель и поместим на gpu"""

    encoder = Encoder(hid_dim, n_layers, cnn_feature_dim).to(device)
    decoder = CaptionNet(emb_dim, hid_dim, n_layers, cnn_feature_dim, dropout, vocab_size).to(device)
    model = ImgCap(encoder, decoder, device, max_len, tokenizer).to(device)

    """ Загрузим лучшую модель."""

    model.load_state_dict(torch.load('imgcap.pth'))

    from torch.autograd import Variable
    from torchvision.models.inception import Inception3
    from warnings import warn

    class BeheadedInception3(Inception3):
        """ Like torchvision.models.inception.Inception3 but the head goes separately """
        def forward(self, x):
            if self.transform_input:
                x = x.clone()
                x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
                x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
                x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            else: warn("Input isn't transformed")
            x = self.Conv2d_1a_3x3(x)
            x = self.Conv2d_2a_3x3(x)
            x = self.Conv2d_2b_3x3(x)
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            x = self.Conv2d_3b_1x1(x)
            x = self.Conv2d_4a_3x3(x)
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            x = self.Mixed_5b(x)
            x = self.Mixed_5c(x)
            x = self.Mixed_5d(x)
            x = self.Mixed_6a(x)
            x = self.Mixed_6b(x)
            x = self.Mixed_6c(x)
            x = self.Mixed_6d(x)
            x = self.Mixed_6e(x)
            x = self.Mixed_7a(x)
            x = self.Mixed_7b(x)
            x_for_attn = x = self.Mixed_7c(x)
            # 8 x 8 x 2048
            x = F.avg_pool2d(x, kernel_size=8)
            # 1 x 1 x 2048
            x_for_capt = x = x.view(x.size(0), -1)
            # 2048
            x = self.fc(x)
            # 1000 (num_classes)
            return x_for_attn, x_for_capt, x

    """ Скачаем обученную CNN модель inception и инициализируем её."""
    def beheaded_inception_v3(model):
        inception_url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        model.load_state_dict(torch.utils.model_zoo.load_url(inception_url))
        return model

    model2 = BeheadedInception3(transform_input=True)
    inception = beheaded_inception_v3(model2).train(False)
    import FaceRecognition
    recognition = FaceRecognition.FaceRecognition()

    def __call__(self, image):
        name = self.recognition.train(image)[0]
        return self.model.generate_one_example(image, self.inception, name)

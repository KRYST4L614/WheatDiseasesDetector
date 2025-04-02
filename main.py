# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms, models
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# import os
#
#
# # Конфигурация
# class Config:
#     CLASSES = ['Dark_brown_spotting', 'Brown_rust', 'Powdery_mildew', "Pyrenophorosis", "Root_rot", "Septoria", "Smut"
#                "Snow_mold", "Striped_mosaic", "Yellow_rust", "Healthy"]
#     IMG_SIZE = 224
#     BATCH_SIZE = 32
#     EPOCHS = 20
#     LR = 0.001
#     DATA_PATH = 'data\\'
#
#
# # Датасет
# class WheatDiseaseDataset(Dataset):
#     def __init__(self, data_path, transform=None):
#         self.data = []
#         self.transform = transform
#
#         for class_idx, class_name in enumerate(Config.CLASSES):
#             class_path = os.path.join(data_path, class_name)
#             for img_name in os.listdir(class_path)[:100]:
#                 self.data.append((os.path.join(class_path, img_name), class_idx))
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         img_path, label = self.data[idx]
#         img = Image.open(img_path).convert('RGB')
#
#         if self.transform:
#             img = self.transform(img)
#
#         return img, label
#
#
# # Аугментация данных
# train_transform = transforms.Compose([
#     transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])
#
# val_transform = transforms.Compose([
#     transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])
#
#
# # Модель (на основе EfficientNet)
# class DiseaseClassifier(nn.Module):
#     def __init__(self, num_classes=len(Config.CLASSES)):
#         super().__init__()
#         self.base_model = models.efficientnet_b0(pretrained=True)
#         self.base_model.classifier[1] = nn.Linear(
#             self.base_model.classifier[1].in_features, num_classes)
#
#     def forward(self, x):
#         return self.base_model(x)
#
#
# # Обучение
# def train_model():
#     # Загрузка данных
#     train_dataset = WheatDiseaseDataset(
#         os.path.join(Config.DATA_PATH, 'train'),
#         train_transform)
#     val_dataset = WheatDiseaseDataset(
#         os.path.join(Config.DATA_PATH, 'val'),
#         val_transform)
#
#     train_loader = DataLoader(train_dataset,
#                               batch_size=Config.BATCH_SIZE,
#                               shuffle=True)
#     val_loader = DataLoader(val_dataset,
#                             batch_size=Config.BATCH_SIZE)
#
#     # Инициализация модели
#     model = DiseaseClassifier()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=Config.LR)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#
#     # Цикл обучения
#     for epoch in range(Config.EPOCHS):
#         print(epoch)
#         model.train()
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#         # Валидация
#         model.eval()
#         val_loss = 0.0
#         correct = 0
#
#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model(inputs)
#                 val_loss += criterion(outputs, labels).item()
#                 _, preds = torch.max(outputs, 1)
#                 correct += torch.sum(preds == labels.data)
#
#         val_loss /= len(val_loader)
#         val_acc = correct.double() / len(val_dataset)
#
#         print(f'Epoch {epoch + 1}/{Config.EPOCHS} | '
#               f'Val Loss: {val_loss:.4f} | '
#               f'Val Acc: {val_acc:.4f}')
#
#     # Сохранение модели
#     torch.save(model.state_dict(), 'wheat_disease_model.pth')
#     return model
#
#
# if __name__ == '__main__':
#     model = train_model()


# import torch
# from torchvision import transforms, models
# from PIL import Image
# import torch.nn as nn
# import os
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# # Конфигурация (должна совпадать с обучением)
# class Config:
#     CLASSES = ['Healthy', 'Brown_rust', 'Yellow_rust']
#     IMG_SIZE = 224
#
#
# class DiseaseClassifier(nn.Module):
#     def __init__(self, num_classes=len(Config.CLASSES)):
#         super().__init__()
#         self.base_model = models.efficientnet_b0(pretrained=False)
#         self.base_model.classifier[1] = nn.Linear(
#             self.base_model.classifier[1].in_features, num_classes)
#
#     def forward(self, x):
#         return self.base_model(x)
#
#
# # Загрузка модели
# class DiseasePredictor:
#     def __init__(self, model_path):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = self.load_model(model_path)
#         self.transform = transforms.Compose([
#             transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#         ])
#
#     def load_model(self, model_path):
#         # Инициализируем модель с правильной архитектурой
#         model = DiseaseClassifier()
#
#         # Загружаем сохраненные веса
#         state_dict = torch.load(model_path, map_location=self.device)
#         model.load_state_dict(state_dict)
#
#         model.eval()
#         model.to(self.device)
#         return model
#
#     def predict(self, image_path):
#         # Загрузка и преобразование изображения
#         img = Image.open(image_path).convert('RGB')
#         img_t = self.transform(img).unsqueeze(0).to(self.device)
#
#         # Предсказание
#         with torch.no_grad():
#             outputs = self.model(img_t)
#             probs = torch.nn.functional.softmax(outputs, dim=1)
#             conf, preds = torch.max(probs, 1)
#
#         return {
#             'class': Config.CLASSES[preds.item()],
#             'confidence': conf.item(),
#             'probabilities': {c: p.item() for c, p in zip(Config.CLASSES, probs.squeeze())}
#         }
#
#     def visualize_prediction(self, image_path, prediction):
#         img = Image.open(image_path)
#         plt.figure(figsize=(10, 5))
#
#         # Изображение
#         plt.subplot(1, 2, 1)
#         plt.imshow(img)
#         plt.title(f"Predicted: {prediction['class']}\nConfidence: {prediction['confidence']:.2f}")
#         plt.axis('off')
#
#         # График вероятностей
#         plt.subplot(1, 2, 2)
#         colors = ['green' if x == prediction['class'] else 'gray' for x in Config.CLASSES]
#         plt.barh(Config.CLASSES, list(prediction['probabilities'].values()), color=colors)
#         plt.xlabel('Probability')
#         plt.xlim(0, 1)
#         plt.tight_layout()
#         plt.show()
#
#
# # Пример использования
# if __name__ == '__main__':
#     # Инициализация предсказателя
#     predictor = DiseasePredictor('wheat_disease_model.pth')
#
#     # Анализ изображения
#     image_path = 'test_wheat.jpg'  # Укажите путь к вашему изображению
#     if os.path.exists(image_path):
#         result = predictor.predict(image_path)
#         print(f"Результат анализа: {result['class']}")
#         print(f"Уверенность: {result['confidence']:.2f}")
#         print("Все вероятности:")
#         for cls, prob in result['probabilities'].items():
#             print(f"  {cls}: {prob:.4f}")
#
#         # Визуализация
#         predictor.visualize_prediction(image_path, result)
#     else:
#         print(f"Файл {image_path} не найден!")


# import os
# import cv2
# import numpy as np
# from skimage.feature import graycomatrix, graycoprops
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt
# from PIL import Image
#
#
# # Конфигурация
# class Config:
#     CLASSES = ['Healthy', 'Brown_rust', 'Yellow_rust', 'Dark_brown_spotting', 'Powdery_mildew']
#     DATA_PATH = 'wheat_diseases\\'
#     GLCM_PROPERTIES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
#     GLCM_DISTANCES = [1]
#     GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
#     TEST_SIZE = 0.2
#     RANDOM_STATE = 42
#
#
# # Функция для извлечения GLCM признаков
# def extract_glcm_features(image_path):
#     # Загрузка и преобразование в grayscale
#     img = np.array(Image.open(image_path).convert('L'))
#
#     # Нормализация изображения
#     img = (img * 255).astype(np.uint8)
#
#     # Вычисление GLCM матрицы
#     glcm = graycomatrix(img,
#                         distances=Config.GLCM_DISTANCES,
#                         angles=Config.GLCM_ANGLES,
#                         levels=256,
#                         symmetric=True,
#                         normed=True)
#
#     # Извлечение признаков
#     features = []
#     for prop in Config.GLCM_PROPERTIES:
#         features.extend(graycoprops(glcm, prop).ravel())
#
#     return features
#
#
# # Загрузка и подготовка данных
# def load_data():
#     features = []
#     labels = []
#
#     for class_idx, class_name in enumerate(Config.CLASSES):
#         class_path = os.path.join(Config.DATA_PATH,"train", class_name)
#         for img_name in os.listdir(class_path)[:100]:  # Ограничение выборки
#             if not (img_name.endswith('jpg') or img_name.endswith('png')):
#                 continue
#             img_path = os.path.join(class_path, img_name)
#             features.append(extract_glcm_features(img_path))
#             labels.append(class_idx)
#
#     return np.array(features), np.array(labels)
#
#
# # Обучение модели
# def train_model():
#     # Загрузка данных
#     X, y = load_data()
#
#     # Разделение на train/test
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE)
#
#     # Инициализация и обучение модели
#     model = RandomForestClassifier(n_estimators=100, random_state=Config.RANDOM_STATE)
#     history = model.fit(X_train, y_train)
#
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['accuracy'], label='Train Accuracy')
#     plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
#     plt.title('Accuracy')
#     plt.legend()
#
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['loss'], label='Train Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.title('Loss')
#     plt.legend()
#
#     plt.tight_layout()
#     plt.show()
#
#     # Оценка модели
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f'Test Accuracy: {accuracy:.2f}')
#
#     return model
#
#
# # Функция для предсказания на новом изображении
# def predict_image(model, image_path):
#     features = extract_glcm_features(image_path)
#     proba = model.predict_proba([features])[0]
#     prediction = model.predict([features])[0]
#
#     return {
#         'class': Config.CLASSES[prediction],
#         'confidence': max(proba),
#         'probabilities': {c: p for c, p in zip(Config.CLASSES, proba)}
#     }
#
#
# # Визуализация результатов
# def visualize_prediction(image_path, prediction):
#     img = Image.open(image_path)
#     plt.figure(figsize=(10, 5))
#
#     plt.subplot(1, 2, 1)
#     plt.imshow(img)
#     plt.title(f"Predicted: {prediction['class']}\nConfidence: {prediction['confidence']:.2f}")
#     plt.axis('off')
#
#     plt.subplot(1, 2, 2)
#     colors = ['green' if x == prediction['class'] else 'gray' for x in Config.CLASSES]
#     plt.barh(Config.CLASSES, list(prediction['probabilities'].values()), color=colors)
#     plt.xlabel('Probability')
#     plt.xlim(0, 1)
#     plt.tight_layout()
#     plt.show()
#
#
# if __name__ == '__main__':
#     # Обучение модели
#     model = train_model()
#
#     # Пример предсказания
#     image_path = 'Powdery.jpg'  # Укажите путь к тестовому изображению
#     if os.path.exists(image_path):
#         result = predict_image(model, image_path)
#         print(f"Результат анализа: {result['class']}")
#         print(f"Уверенность: {result['confidence']:.2f}")
#         print("Все вероятности:")
#         for cls, prob in result['probabilities'].items():
#             print(f"  {cls}: {prob:.4f}")
#
#         visualize_prediction(image_path, result)
#     else:
#         print(f"Файл {image_path} не найден!")


# import os
# import cv2
# import numpy as np
# from skimage.feature import graycomatrix
# import matplotlib.pyplot as plt
# from PIL import Image
#
#
# print("Path to dataset files:", path)
# def visualize_glcm(image_path):
#     # Загрузка и преобразование изображения
#     img = np.array(Image.open(image_path).convert('L'))
#     img = (img * 255).astype(np.uint8)
#
#     # Вычисление GLCM для разных углов
#     angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
#     titles = ['0°', '45°', '90°', '135°']
#
#     # Создаем фигуру с 5 subplots (оригинал + 4 угла)
#     fig, axes = plt.subplots(1, 5, figsize=(20, 4))
#     fig.suptitle('GLCM Matrix Visualization', fontsize=16)
#
#     # Отображаем оригинальное изображение
#     axes[0].imshow(img, cmap='gray')
#     axes[0].set_title('Original Image')
#     axes[0].axis('off')
#
#     # Вычисляем и отображаем GLCM для каждого угла
#     for i, angle in enumerate(angles, 1):
#         glcm = graycomatrix(img, distances=[1], angles=[angle],
#                             levels=256, symmetric=True, normed=True)
#         glcm_matrix = np.squeeze(glcm)
#
#         im = axes[i].imshow(glcm_matrix, cmap='hot')
#         axes[i].set_title(f'Angle: {titles[i - 1]}')
#         axes[i].axis('off')
#
#         # Добавляем colorbar для каждого subplot
#         divider = make_axes_locatable(axes[i])
#         cax = divider.append_axes("right", size="5%", pad=0.05)
#         plt.colorbar(im, cax=cax)
#
#     plt.tight_layout()
#     plt.show()
#
#
# # Пример использования
# if __name__ == '__main__':
#     image_path = 'Yellow_rust1160.jpg'  # Укажите путь к изображению
#
#     if os.path.exists(image_path):
#         visualize_glcm(image_path)
#     else:
#         print(f"Файл {image_path} не найден!")

#
# import numpy as np
# from skimage.feature import graycomatrix
# from PIL import Image
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# def visualize_glcm_matrix(image_path, distance=1, angle=0):
#     # Загрузка и подготовка изображения
#     img = np.array(Image.open(image_path).convert('L'))
#     img = (img * 255).astype(np.uint8)
#
#     # Вычисление GLCM
#     glcm = graycomatrix(img,
#                         distances=[distance],
#                         angles=[angle],
#                         levels=256,
#                         symmetric=True,
#                         normed=True)
#
#     # Преобразование в 2D матрицу
#     glcm_matrix = np.squeeze(glcm).astype(float)
#
#     # Создаем фигуру
#     fig, ax = plt.subplots(figsize=(12, 10))
#
#     # Отображаем матрицу как таблицу
#     df = pd.DataFrame(glcm_matrix[:20, :20])  # Показываем только 20x20 для наглядности
#
#     # Создаем табличное представление
#     table = plt.table(cellText=np.round(df.values, 4),
#                       rowLabels=df.index,
#                       colLabels=df.columns,
#                       loc='center',
#                       cellLoc='center')
#
#     # Настройки внешнего вида
#     table.scale(1, 1.5)
#     ax.axis('off')
#     ax.set_title(f'GLCM Matrix (Distance={distance}, Angle={np.degrees(angle):.0f}°)',
#                  fontsize=14, pad=20)
#
#     # Добавляем пояснения
#     plt.figtext(0.5, 0.05,
#                 "Каждая ячейка показывает вероятность перехода от интенсивности i (строка) к j (столбец)",
#                 ha="center", fontsize=12)
#
#     plt.tight_layout()
#     plt.show()
#
#
# # Пример использования
# if __name__ == '__main__':
#     image_path = 'test_wheat.jpg'  # Укажите путь к изображению
#
#     if os.path.exists(image_path):
#         # Визуализация для разных углов
#         for angle in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
#             visualize_glcm_matrix(image_path, distance=1, angle=angle)
#     else:
#         print(f"Файл {image_path} не найден!")

# import os
# import pickle
#
# import cv2
# import numpy as np
# from skimage.feature import graycomatrix, graycoprops
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import LabelBinarizer
# import matplotlib.pyplot as plt
# from PIL import Image
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dropout, Activation, Dense, Flatten
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import to_categorical
#
#
# # Конфигурация
# class Config:
#     DATA_PATH = 'wheat_diseases\\train\\'
#     CLASSES = []
#     for class_name in os.listdir(DATA_PATH):
#         CLASSES.append(class_name)
#     print(CLASSES)
#     GLCM_PROPERTIES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', ]
#     GLCM_DISTANCES = [1]
#     GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
#     TEST_SIZE = 0.1
#     TRAIN_SIZE = 400
#     RANDOM_STATE = 42
#     EPOCHS = 100
#     BATCH_SIZE = 8
#     PATIENCE = 10
#     MEANS = {}
#
# def calculate_difference_entropy(glcm):
#     """
#     Вычисляет Difference Entropy из матрицы GLCM.
#
#     Параметры:
#         glcm : ndarray
#             Матрица GLCM, полученная из graycomatrix()
#
#     Возвращает:
#         float: Значение Difference Entropy
#     """
#     # Нормализуем матрицу GLCM, чтобы получить вероятности
#     glcm = glcm.astype(np.float64)
#     glcm_sum = np.sum(glcm)
#     if glcm_sum > 0:
#         glcm /= glcm_sum
#
#     # Вычисляем матрицу разностей (P(i-j))
#     size = glcm.shape[0]
#     diff_matrix = np.zeros(size, dtype=np.float64)
#
#     for i in range(size):
#         for j in range(size):
#             diff = abs(i - j)
#             diff_matrix[diff] += glcm[i, j, 0, 0]  # Берем первый угол и расстояние
#
#     # Вычисляем Difference Entropy
#     entropy = np.sum(diff_matrix*np.log(diff_matrix))
#
#     return -1*entropy
#
# # Функция для извлечения GLCM признаков
# def extract_glcm_features(image_path):
#     # Загрузка и преобразование в grayscale
#     img = np.array(Image.open(image_path).convert('L'))
#
#     # Нормализация изображения
#     img = (img * 255).astype(np.uint8)
#
#     # Вычисление GLCM матрицы
#     glcm = graycomatrix(img,
#                         distances=Config.GLCM_DISTANCES,
#                         angles=Config.GLCM_ANGLES,
#                         levels=256,
#                         symmetric=True,
#                         normed=True)
#
#     # Извлечение признаков
#     features = []
#     for prop in Config.GLCM_PROPERTIES:
#         features.extend(graycoprops(glcm, prop).ravel())
#     # features.extend(calculate_difference_entropy(glcm).ravel())
#
#     return np.array(features)
#
#
# # Загрузка и подготовка данных
# def load_data():
#     features = []
#     labels = []
#
#     for class_idx, class_name in enumerate(Config.CLASSES):
#         class_path = os.path.join(Config.DATA_PATH, class_name)
#         for img_name in os.listdir(class_path)[:Config.TRAIN_SIZE]:
#             if not (img_name.endswith('png') or img_name.endswith('jpg')):
#                 continue
#             img_path = os.path.join(class_path, img_name)
#             features.append(extract_glcm_features(img_path))
#             labels.append(class_name)
#
#     # One-hot encoding (только LabelBinarizer!)
#     lb = LabelBinarizer()
#     labels = lb.fit_transform(labels)  # Форма (n_samples, n_classes)
#
#     return np.array(features), np.array(labels), lb
#
#
# def create_model(input_shape, num_classes):
#     model = Sequential()
#     model.add(Dense(512, activation='relu'))
#     model.add(Dropout(0.25))
#     model.add(Dense(512, activation='relu'))
#     model.add(Dropout(0.25))
#     model.add(Dense(num_classes, activation='softmax'))  # Выход (None, 3)
#
#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',  # Для one-hot меток
#                   metrics=['accuracy'])
#     return model
#
#
# # Обучение модели
# def train_model():
#     # Загрузка данных
#     X, y, lb = load_data()
#
#     # Разделение на train/test
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE)
#
#     # Нормализация данных
#     X_mean = X_train.mean(axis=0)
#     X_std = X_train.std(axis=0)
#     X_train = (X_train - X_mean) / (X_std + 1e-10)
#     X_test = (X_test - X_mean) / (X_std + 1e-10)
#
#     # Создание модели
#     model = create_model(X_train.shape[1], y_train.shape[1])
#
#     # Ранняя остановка
#     early_stopping = EarlyStopping(monitor='val_loss',
#                                    patience=Config.PATIENCE,
#                                    restore_best_weights=True)
#
#     # Обучение модели
#     history = model.fit(
#         X_train, y_train,
#         epochs=Config.EPOCHS,
#         batch_size=Config.BATCH_SIZE,
#         validation_split=0.1,
#         callbacks=[early_stopping],
#         verbose=1)
#
#     # Оценка модели
#     loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
#     print(f'Test Accuracy: {accuracy:.2f}')
#
#     # Сохранение модели и параметров нормализации
#     model.save('tf_model.h5')
#     np.savez('normalization_params.npz', mean=X_mean, std=X_std)
#     with open('label_binarizer.pkl', 'wb') as f:
#         pickle.dump(lb, f)
#
#     # Графики обучения
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['accuracy'], label='Train Accuracy')
#     plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
#     plt.title('Accuracy')
#     plt.legend()
#
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['loss'], label='Train Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.title('Loss')
#     plt.legend()
#
#     plt.tight_layout()
#     plt.show()
#
#     return model, X_mean, X_std, lb
#
#
# # Функция для предсказания на новом изображении
# def predict_image(model, X_mean, X_std, lb, image_path):
#     features = extract_glcm_features(image_path)
#     features = (features - X_mean) / (X_std + 1e-10)
#     proba = model.predict(features.reshape(1, -1), verbose=0)[0]
#     prediction = np.argmax(proba)
#     class_name = lb.classes_[prediction]
#
#     return {
#         'class': class_name,
#         'confidence': proba[prediction],
#         'probabilities': {c: p for c, p in zip(lb.classes_, proba)}
#     }
#
#
# # Визуализация результатов
# def visualize_prediction(image_path, prediction, lb):
#     img = Image.open(image_path)
#     plt.figure(figsize=(10, 5))
#
#     plt.subplot(1, 2, 1)
#     plt.imshow(img)
#     plt.title(f"Predicted: {prediction['class']}\nConfidence: {prediction['confidence']:.2f}")
#     plt.axis('off')
#
#     plt.subplot(1, 2, 2)
#     colors = ['green' if x == prediction['class'] else 'gray' for x in lb.classes_]
#     plt.barh(lb.classes_, list(prediction['probabilities'].values()), color=colors)
#     plt.xlabel('Probability')
#     plt.xlim(0, 1)
#     plt.tight_layout()
#     plt.show()
#
#
# if __name__ == '__main__':
#     # Обучение модели или загрузка сохраненной
#     # print("Training new model...")
#     # model, X_mean, X_std, lb = train_model()
#     try:
#         model = load_model('tf_model.h5')
#         params = np.load('normalization_params.npz')
#         X_mean, X_std = params['mean'], params['std']
#         with open('label_binarizer.pkl', 'rb') as f:
#             lb = pickle.load(f)
#         print("Model loaded from file")
#     except:
#         print("Training new model...")
#         model, X_mean, X_std, lb = train_model()
#
#     # Пример предсказания
#     image_path = 'Yellow_rust.jpg'  # Укажите путь к тестовому изображению
#     if os.path.exists(image_path):
#         result = predict_image(model, X_mean, X_std, lb, image_path)
#         print(f"Результат анализа: {result['class']}")
#         print(f"Уверенность: {result['confidence']:.2f}")
#         print("Все вероятности:")
#         for cls, prob in result['probabilities'].items():
#             print(f"  {cls}: {prob:.4f}")
#
#         visualize_prediction(image_path, result, lb)
#     else:
#         print(f"Файл {image_path} не найден!")

import os
import pickle
import numpy as np
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from skimage import img_as_ubyte, transform
import pandas as pd


class Config:
    DATA_PATH = 'wheat_diseases/train'
    CLASSES = sorted(os.listdir(DATA_PATH))
    GLCM_PROPERTIES = ['energy', 'contrast', 'homogeneity', 'entropy', 'dissimilarity', 'correlation', 'ASM', 'mean']
    COLOR_COMBINATIONS = ['R', 'G', 'B', 'RG', 'GB', 'RB']  # 6 цветовых компонент
    GLCM_DISTANCES = [1]
    GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    TEST_SIZE = 0.2
    TRAIN_SIZE = 20
    RANDOM_STATE = 42
    EPOCHS = 100
    BATCH_SIZE = 8
    PATIENCE = 10
    IMG_SIZE = (255, 255)


def extract_color_combinations(img):
    """Извлекает 6 цветовых комбинаций: R, G, B, RG, GB, RB"""
    # Разделяем на RGB каналы
    size = Config.IMG_SIZE
    r, g, b = img.split()
    r = transform.resize(np.array(r), size, anti_aliasing=True)
    g = transform.resize(np.array(b), size, anti_aliasing=True)
    b = transform.resize(np.array(b), size, anti_aliasing=True)
    rg = transform.resize(np.array((r.astype(np.float32) + g.astype(np.float32)) / 2), size, anti_aliasing=True)
    gb = transform.resize(np.array((g.astype(np.float32) + b.astype(np.float32)) / 2), size, anti_aliasing=True)
    rb = transform.resize(np.array((r.astype(np.float32) + b.astype(np.float32)) / 2), size, anti_aliasing=True)

    # Создаем комбинации каналов
    combinations = {
        'R': img_as_ubyte(r),
        'G': img_as_ubyte(g),
        'B': img_as_ubyte(b),
        'RG': img_as_ubyte((rg - np.min(rg)) / (np.max(rg) - np.min(rg))),
        'GB': img_as_ubyte((gb - np.min(gb)) / (np.max(gb) - np.min(gb))),
        'RB': img_as_ubyte((rb - np.min(rb)) / (np.max(rb) - np.min(rb)))
    }

    return combinations


def extract_glcm_features(image_path):
    """Извлекает признаки Харалика для всех 6 цветовых комбинаций"""
    img = Image.open(image_path)
    color_combinations = extract_color_combinations(img)
    # Конвертируем в 8-битное изображение (0-255)
    # image = img_as_ubyte(color_combinations)

    all_features = []

    for combo_name in Config.COLOR_COMBINATIONS:
        channel_data = color_combinations[combo_name]

        max_val = channel_data.max()
        quantized = np.digitize(channel_data, bins=np.linspace(0, max_val, 8)) - 1

        # Вычисление GLCM матрицы
        glcm = graycomatrix(quantized,
                            distances=Config.GLCM_DISTANCES,
                            angles=Config.GLCM_ANGLES,
                            levels=8,
                            symmetric=True,
                            normed=False)

        # Извлечение признаков для каждого свойства
        for prop in Config.GLCM_PROPERTIES:
            feature = graycoprops(glcm, prop).ravel()
            all_features.extend(feature)

    return np.array(all_features)

def load_data():
    features = []
    labels = []

    for class_name in Config.CLASSES:
        class_path = os.path.join(Config.DATA_PATH, class_name)
        for img_name in os.listdir(class_path)[:Config.TRAIN_SIZE]:
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(class_path, img_name)
            try:
                features.append(extract_glcm_features(img_path))
                labels.append(class_name)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    print(labels)

    return np.array(features), np.array(labels), lb


def create_model(input_shape, num_classes):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        Dropout(0.25),
        Dense(256, activation='relu'),
        Dropout(0.25),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model():
    X, y, lb = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE)

    # Нормализация данных
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_train = (X_train - X_mean) / (X_std + 1e-10)
    X_test = (X_test - X_mean) / (X_std + 1e-10)

    model = create_model(X_train.shape[1], y_train.shape[1])

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=Config.PATIENCE,
                                   restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Accuracy: {accuracy:.2f}')

    # Сохранение модели
    model.save('tf_model_6ch.h5')
    np.savez('normalization_params_6ch.npz', mean=X_mean, std=X_std)
    with open('label_binarizer_6ch.pkl', 'wb') as f:
        pickle.dump(lb, f)

    # Визуализация обучения
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return model, X_mean, X_std, lb


def predict_image(model, X_mean, X_std, lb, image_path):
    features = extract_glcm_features(image_path)
    features = (features - X_mean) / (X_std + 1e-10)
    proba = model.predict(features.reshape(1, -1), verbose=0)[0]
    prediction = np.argmax(proba)

    return {
        'class': lb.classes_[prediction],
        'confidence': proba[prediction],
        'probabilities': {c: p for c, p in zip(lb.classes_, proba)}
    }


def visualize_prediction(image_path, prediction, lb):
    img = Image.open(image_path)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Predicted: {prediction['class']}\nConfidence: {prediction['confidence']:.2f}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    colors = ['green' if x == prediction['class'] else 'gray' for x in lb.classes_]
    plt.barh(lb.classes_, list(prediction['probabilities'].values()), color=colors)
    plt.xlabel('Probability')
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.show()


def visualize_glcm_matrix(image_path, distance=1, angle=0):
    # Загрузка и подготовка изображения
    img = np.array(Image.open(image_path).convert('L'))
    img = (img * 255).astype(np.uint8)

    # Вычисление GLCM
    glcm = graycomatrix(img,
                        distances=[distance],
                        angles=[angle],
                        levels=256,
                        symmetric=True,
                        normed=True)

    # Преобразование в 2D матрицу
    glcm_matrix = np.squeeze(glcm).astype(float)

    # Создаем фигуру
    fig, ax = plt.subplots(figsize=(12, 10))

    # Отображаем матрицу как таблицу
    df = pd.DataFrame(glcm_matrix[:20, :20])  # Показываем только 20x20 для наглядности

    # Создаем табличное представление
    table = plt.table(cellText=np.round(df.values, 4),
                      rowLabels=df.index,
                      colLabels=df.columns,
                      loc='center',
                      cellLoc='center')

    # Настройки внешнего вида
    table.scale(1, 1.5)
    ax.axis('off')
    ax.set_title(f'GLCM Matrix (Distance={distance}, Angle={np.degrees(angle):.0f}°)',
                 fontsize=14, pad=20)

    # Добавляем пояснения
    plt.figtext(0.5, 0.05,
                "Каждая ячейка показывает вероятность перехода от интенсивности i (строка) к j (столбец)",
                ha="center", fontsize=12)

    plt.tight_layout()
    plt.show()

def visualize_glcm(image_path):
    # Загрузка и преобразование изображения
    img = Image.open(image_path)
    r, g, b = img.split()
    r = np.array(r)
    r = transform.resize(r, Config.IMG_SIZE, anti_aliasing=True)
    # Конвертируем в 8-битное изображение (0-255)
    image = img_as_ubyte(r)

    # Квантуем до заданного количества уровней
    max_val = image.max()
    quantized = np.digitize(image, bins=np.linspace(0, max_val, 8)) - 1

    # Вычисление GLCM
    glcm = graycomatrix(quantized,
                        distances=[1],
                        angles=[0],
                        levels=8,
                        symmetric=True,
                        normed=False)

    # Преобразование в 2D матрицу
    glcm_matrix = np.squeeze(glcm).astype(float)

    # Создаем фигуру
    fig, ax = plt.subplots(figsize=(12, 10))

    # Отображаем матрицу как таблицу
    df = pd.DataFrame(glcm_matrix[:8, :8])  # Показываем только 20x20 для наглядности

    # Создаем табличное представление
    table = plt.table(cellText=np.round(df.values, 4),
                      rowLabels=df.index,
                      colLabels=df.columns,
                      loc='center',
                      cellLoc='center')

    # Настройки внешнего вида
    table.scale(1, 1.5)
    ax.axis('off')
    ax.set_title(f'GLCM Matrix (Distance={1}, Angle={np.degrees(0):.0f}°)',
                 fontsize=14, pad=20)

    # Добавляем пояснения
    plt.figtext(0.5, 0.05,
                "Каждая ячейка показывает вероятность перехода от интенсивности i (строка) к j (столбец)",
                ha="center", fontsize=12)

    plt.tight_layout()
    plt.show()

    plt.imshow(quantized)
    plt.show()



if __name__ == '__main__':
    # print("Training new model...")
    # model, X_mean, X_std, lb = train_model()
    try:
        model = load_model('tf_model_6ch.h5')
        params = np.load('normalization_params_6ch.npz')
        X_mean, X_std = params['mean'], params['std']
        with open('label_binarizer_6ch.pkl', 'rb') as f:
            lb = pickle.load(f)
        print("Model loaded from file")
    except:
        print("Training new model...")
        model, X_mean, X_std, lb = train_model()

    test_image = 'Brown_rust.jpg'  # Укажите путь к тестовому изображению
    if os.path.exists(test_image):
        result = predict_image(model, X_mean, X_std, lb, test_image)
        print(f"Результат анализа: {result['class']}")
        print(f"Уверенность: {result['confidence']:.2f}")
        print("Все вероятности:")
        for cls, prob in result['probabilities'].items():
            print(f"  {cls}: {prob:.4f}")

        visualize_prediction(test_image, result, lb)
    else:
        print(f"Файл {test_image} не найден!")

    if os.path.exists(test_image):
        # Визуализация для разных углов
        visualize_glcm(test_image)
    else:
        print(f"Файл {test_image} не найден!")
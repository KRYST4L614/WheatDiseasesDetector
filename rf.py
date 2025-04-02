import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from PIL import Image


# Конфигурация
class Config:
    DATA_PATH = 'data\\'
    CLASSES = os.listdir(DATA_PATH)
    GLCM_PROPERTIES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    GLCM_DISTANCES = [1]
    GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    COLOR_COMBINATIONS = ['R', 'G', 'B', 'RG', 'GB', 'RB']  # 6 цветовых компонент
    TRAIN_SIZE = 50
    EPOCHS = 100
    BATCH_SIZE = 8
    PATIENCE = 10


def extract_color_combinations(img):
    """Извлекает 6 цветовых комбинаций: R, G, B, RG, GB, RB"""
    # Разделяем на RGB каналы
    r, g, b = img.split()
    r = np.array(r)
    g = np.array(g)
    b = np.array(b)

    # Создаем комбинации каналов
    combinations = {
        'R': r,
        'G': g,
        'B': b,
        'RG': (r.astype(np.float32) + g.astype(np.float32)) / 2,
        'GB': (g.astype(np.float32) + b.astype(np.float32)) / 2,
        'RB': (r.astype(np.float32) + b.astype(np.float32)) / 2
    }

    return combinations


def extract_glcm_features(image_path):
    """Извлекает признаки Харалика для всех 6 цветовых комбинаций"""
    img = Image.open(image_path)
    color_combinations = extract_color_combinations(img)

    all_features = []

    for combo_name in Config.COLOR_COMBINATIONS:
        channel_data = color_combinations[combo_name]

        # Нормализация и преобразование в uint8
        channel_data = (channel_data * 255).astype(np.uint8)

        # Вычисление GLCM матрицы
        glcm = graycomatrix(channel_data,
                            distances=Config.GLCM_DISTANCES,
                            angles=Config.GLCM_ANGLES,
                            levels=256,
                            symmetric=True,
                            normed=True)

        # Извлечение признаков для каждого свойства
        for prop in Config.GLCM_PROPERTIES:
            feature = graycoprops(glcm, prop).ravel()
            all_features.extend(feature)

    return np.array(all_features)


# Функция для извлечения GLCM признаков
def extract_glcm_features(image_path):
    # Загрузка и преобразование в grayscale
    img = np.array(Image.open(image_path).convert('L'))

    # Нормализация изображения
    img = (img * 255).astype(np.uint8)

    # Вычисление GLCM матрицы
    glcm = graycomatrix(img,
                        distances=Config.GLCM_DISTANCES,
                        angles=Config.GLCM_ANGLES,
                        levels=256,
                        symmetric=True,
                        normed=True)

    # Извлечение признаков
    features = []
    for prop in Config.GLCM_PROPERTIES:
        features.extend(graycoprops(glcm, prop).ravel())

    return features


# Загрузка и подготовка данных
def load_data():
    features = []
    labels = []

    for class_idx, class_name in enumerate(Config.CLASSES):
        class_path = os.path.join(Config.DATA_PATH, class_name)
        for img_name in os.listdir(class_path)[:100]:  # Ограничение выборки
            if not (img_name.endswith('jpg') or img_name.endswith('png')):
                continue
            img_path = os.path.join(class_path, img_name)
            features.append(extract_glcm_features(img_path))
            labels.append(class_idx)

    return np.array(features), np.array(labels)


# Обучение модели
def train_model():
    # Загрузка данных
    X, y = load_data()

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE)

    # Инициализация и обучение модели
    model = RandomForestClassifier(n_estimators=100, random_state=Config.RANDOM_STATE)
    model.fit(X_train, y_train)

    # Оценка модели
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {accuracy:.2f}')

    return model


# Функция для предсказания на новом изображении
def predict_image(model, image_path):
    features = extract_glcm_features(image_path)
    proba = model.predict_proba([features])[0]
    prediction = model.predict([features])[0]

    return {
        'class': Config.CLASSES[prediction],
        'confidence': max(proba),
        'probabilities': {c: p for c, p in zip(Config.CLASSES, proba)}
    }


# Визуализация результатов
def visualize_prediction(image_path, prediction):
    img = Image.open(image_path)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Predicted: {prediction['class']}\nConfidence: {prediction['confidence']:.2f}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    colors = ['green' if x == prediction['class'] else 'gray' for x in Config.CLASSES]
    plt.barh(Config.CLASSES, list(prediction['probabilities'].values()), color=colors)
    plt.xlabel('Probability')
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Обучение модели
    model = train_model()

    # Пример предсказания
    image_path = 'Dark_brown.png'  # Укажите путь к тестовому изображению
    if os.path.exists(image_path):
        result = predict_image(model, image_path)
        print(f"Результат анализа: {result['class']}")
        print(f"Уверенность: {result['confidence']:.2f}")
        print("Все вероятности:")
        for cls, prob in result['probabilities'].items():
            print(f"  {cls}: {prob:.4f}")

        visualize_prediction(image_path, result)
    else:
        print(f"Файл {image_path} не найден!")
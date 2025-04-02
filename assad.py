import os
import pickle
import numpy as np
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


class Config:
    DATA_PATH = 'wheat_diseases/train/'
    CLASSES = sorted(os.listdir(DATA_PATH))
    GLCM_PROPERTIES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    COLOR_COMBINATIONS = ['R', 'G', 'B', 'RG', 'GB', 'RB']  # 6 цветовых компонент
    GLCM_DISTANCES = [1]
    GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    TEST_SIZE = 0.2
    TRAIN_SIZE = 100
    RANDOM_STATE = 42
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
        Dense(512, activation='relu', input_shape=(input_shape,)),
        Dropout(0.25),
        Dense(512, activation='relu'),
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
        validation_split=0.1,
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

    test_image = 'Dark_brown.png'  # Укажите путь к тестовому изображению
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
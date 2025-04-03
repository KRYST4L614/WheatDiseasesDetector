import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os


class WheatLeafAnalyzer:
    def __init__(self, n_clusters=3, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]):
        self.n_clusters = n_clusters
        self.distances = distances
        self.angles = angles
        self.scaler = MinMaxScaler()
        self.classifier = SVC(kernel='rbf', gamma='scale')

    def preprocess_image(self, image_path):
        """Предварительная обработка изображения"""
        # Загрузка изображения
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        # Нормализация
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        normalized = image / 255.0

        # Многоцветная кластеризация
        pixels = normalized.reshape(-1, 3)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = kmeans.fit_predict(pixels)
        clustered = kmeans.cluster_centers_[labels].reshape(image.shape)

        # Преобразование в grayscale
        gray = cv2.cvtColor((clustered * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        return gray

    def compute_glcm_features(self, gray_image):
        """Вычисление признаков Харалика на основе GLCM"""
        # Квантование изображения до 256 уровней серого
        gray_image = (gray_image / 16).astype(np.uint8)

        # Вычисление матрицы смежности
        glcm = graycomatrix(gray_image, distances=self.distances, angles=self.angles,
                            levels=16, symmetric=True, normed=True)

        # Вычисление признаков
        features = []
        for prop in ['contrast', 'dissimilarity', 'homogeneity',
                     'energy', 'correlation', 'ASM']:
            feature = graycoprops(glcm, prop)
            features.append(feature.mean())

        # Дополнительные признаки
        # Энтропия
        entropy = -np.sum(glcm * np.log2(glcm + (glcm == 0)))
        features.append(entropy)

        # Максимальная вероятность
        max_prob = np.max(glcm)
        features.append(max_prob)

        # Обратный момент различия
        idm = np.sum(glcm / (1 + np.arange(glcm.shape[0]) ** 2))
        features.append(idm)

        # Информационная мера корреляции 1
        # (более сложные признаки можно добавить по необходимости)

        return np.array(features)

    def extract_features(self, image_path):
        """Извлечение признаков из изображения"""
        gray = self.preprocess_image(image_path)
        features = self.compute_glcm_features(gray)
        return features

    def train(self, X, y):
        """Обучение классификатора"""
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, y)

    def predict(self, X):
        """Предсказание класса"""
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict(X_scaled)

    def evaluate(self, X, y):
        """Оценка качества классификации"""
        X_scaled = self.scaler.transform(X)
        y_pred = self.classifier.predict(X_scaled)
        print(classification_report(y, y_pred))


def load_dataset(data_dir):
    """Загрузка набора данных"""
    classes = os.listdir(data_dir)
    X = []
    y = []

    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for image_file in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_file)
            try:
                analyzer = WheatLeafAnalyzer()
                features = analyzer.extract_features(image_path)
                X.append(features)
                y.append(class_name)
            except Exception as e:
                print(f"Ошибка обработки {image_path}: {e}")

    return np.array(X), np.array(y)


# Пример использования
if __name__ == "__main__":
    # Загрузка данных
    data_dir = "wheat_diseases//train"
    X, y = load_dataset(data_dir)

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение модели
    analyzer = WheatLeafAnalyzer()
    analyzer.train(X_train, y_train)

    # Оценка качества
    print("Результаты на тестовой выборке:")
    analyzer.evaluate(X_test, y_test)

    # Пример предсказания для нового изображения
    test_image_path = "Healthy.jpg"
    features = analyzer.extract_features(test_image_path)
    prediction = analyzer.predict([features])
    print(f"Предсказанный класс: {prediction[0]}")
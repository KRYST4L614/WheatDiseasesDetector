import os
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


class PlantDiseaseClassifier:
    def __init__(self):
        # Параметры алгоритма
        self.gray_levels = 8  # Количество градаций серого
        self.distances = [1]  # Расстояние для GLCM
        self.angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # Углы для GLCM
        self.props = ['contrast', 'correlation', 'energy', 'homogeneity']  # Параметры Харалика
        self.color_combinations = ['R', 'G', 'B', 'RG', 'RB', 'GB']  # Цветовые компоненты
        self.target_size = (300, 100)  # Размер нормализованных изображений
        self.beta = 0.95  # Доверительная вероятность

    def normalize_image(self, img_path):
        """Нормализация изображения листа"""
        # 1. Загрузка изображения
        img = Image.open(img_path).convert('RGB')

        # 2. Удаление фона и сегментация листа (упрощенная версия)
        # Здесь должна быть более сложная логика сегментации
        img_array = np.array(img)

        # 3. Изменение размера
        img = img.resize(self.target_size)

        # 4. Нормализация яркости и контраста
        img_array = np.array(img)
        img_array = img_array / 255.0  # Нормализация [0, 1]

        # 5. Разделение на цветовые каналы
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

        # 6. Создание комбинаций каналов
        color_data = {
            'R': r,
            'G': g,
            'B': b,
            'RG': (r + g) / 2,
            'RB': (r + b) / 2,
            'GB': (g + b) / 2
        }

        return color_data

    def calculate_glcm_features(self, channel_data):
        """Вычисление параметров Харалика для одного канала"""
        # Квантование до 8 уровней
        quantized = (channel_data * (self.gray_levels - 1)).astype(np.uint8)

        # Расчет GLCM
        glcm = graycomatrix(quantized,
                            distances=self.distances,
                            angles=self.angles,
                            levels=self.gray_levels,
                            symmetric=True,
                            normed=True)

        # Извлечение признаков
        features = []
        for prop in self.props:
            feature = graycoprops(glcm, prop).ravel()
            features.extend(feature)

        return np.array(features)

    def create_reference_descriptions(self, healthy_dir, disease_dirs):
        """Создание эталонных описаний для здоровых и больных листьев"""
        reference_data = {}

        # Обработка здоровых листьев
        healthy_features = []
        for img_file in os.listdir(healthy_dir):
            img_path = os.path.join(healthy_dir, img_file)
            if not img_path.endswith(("png", "jpg", "jpeg")):
                continue
            color_data = self.normalize_image(img_path)

            features = []
            for combo in self.color_combinations:
                channel_features = self.calculate_glcm_features(color_data[combo])
                features.extend(channel_features)

            healthy_features.append(features)

        healthy_features = np.array(healthy_features)
        reference_data['healthy'] = {
            'mean': np.mean(healthy_features, axis=0),
            'std': np.std(healthy_features, axis=0),
            'ci': stats.t.interval(0.95, len(healthy_features) - 1,
                                   loc=np.mean(healthy_features, axis=0),
                                   scale=stats.sem(healthy_features, axis=0))
        }

        # Обработка больных листьев
        for disease, dir_path in disease_dirs.items():
            disease_features = []
            for img_file in os.listdir(dir_path):
                img_path = os.path.join(dir_path, img_file)
                if not img_path.endswith(("png", "jpg", "jpeg")):
                    continue
                color_data = self.normalize_image(img_path)

                features = []
                for combo in self.color_combinations:
                    channel_features = self.calculate_glcm_features(color_data[combo])
                    features.extend(channel_features)

                disease_features.append(features)

            disease_features = np.array(disease_features)
            reference_data[disease] = {
                'mean': np.mean(disease_features, axis=0),
                'std': np.std(disease_features, axis=0),
                'ci': stats.t.interval(0.95, len(disease_features) - 1,
                                       loc=np.mean(disease_features, axis=0),
                                       scale=stats.sem(disease_features, axis=0))
            }

        return reference_data

    def diagnose_disease(self, target_images, reference_data):
        """Диагностика заболеваний по целевым изображениям"""
        # 1. Извлечение признаков для целевых изображений
        target_features = []
        for img_path in target_images:
            if not img_path.endswith(("png", "jpg", "jpeg")):
                continue
            color_data = self.normalize_image(img_path)

            features = []
            for combo in self.color_combinations:
                channel_features = self.calculate_glcm_features(color_data[combo])
                features.extend(channel_features)

            target_features.append(features)

        target_features = np.array(target_features)

        # 2. Усреднение признаков
        avg_features = np.mean(target_features, axis=0)

        # 3. Расчет функций принадлежности
        membership_functions = {}

        for disease, data in reference_data.items():
            # Метрика на основе суммы модулей разности
            mf1 = np.sum(np.abs(data['mean'] - avg_features))

            # Метрика на основе евклидова расстояния
            mf2 = np.sqrt(np.sum((data['mean'] - avg_features) ** 2))

            # Метрика на основе корреляции
            corr = np.corrcoef(data['mean'], avg_features)[0, 1]
            mf3 = (corr + 1) / 2  # Приведение к диапазону [0,1]

            membership_functions[disease] = {
                'sum_of_differences': mf1,
                'euclidean_distance': mf2,
                'correlation': mf3
            }

        # 4. Определение диагноза
        # Используем корреляционную метрику
        diagnosis = max(membership_functions.items(),
                        key=lambda x: x[1]['correlation'])[0]

        return diagnosis, membership_functions

    def visualize_results(self, membership_functions):
        """Визуализация результатов диагностики"""
        diseases = list(membership_functions.keys())
        corr_values = [mf['correlation'] for mf in membership_functions.values()]

        plt.figure(figsize=(10, 6))
        plt.barh(diseases, corr_values, color='skyblue')
        plt.xlabel('Значение функции принадлежности (корреляция)')
        plt.title('Результаты диагностики заболеваний растений')
        plt.xlim(0, 1)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.show()


# Пример использования
if __name__ == "__main__":
    # Инициализация классификатора
    classifier = PlantDiseaseClassifier()

    # Пути к данным (примерные)
    data_dirs = {
        'healthy': 'data/Healthy',
        'brown_rust': 'data/Brown_rust',
        'yellow_rust': 'data/Yellow_rust',
        'snow_mold': 'data/Snow_mold',
        'septoria': 'data/Septoria'
    }

    # 1. Создание эталонных описаний
    print("Создание эталонных описаний...")
    reference_data = classifier.create_reference_descriptions(
        healthy_dir=data_dirs['healthy'],
        disease_dirs={k: v for k, v in data_dirs.items() if k != 'healthy'}
    )

    # 2. Диагностика новых изображений
    test_images = ['Yellow_rust.jpg']

    print("\nВыполнение диагностики...")
    diagnosis, mf = classifier.diagnose_disease(test_images, reference_data)

    print(f"\nРезультат диагностики: {diagnosis}")
    print("\nЗначения функций принадлежности:")
    for disease, values in mf.items():
        print(f"{disease}: {values['correlation']:.3f}")

    # 3. Визуализация результатов
    classifier.visualize_results(mf)
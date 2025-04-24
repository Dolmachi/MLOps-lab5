import configparser
from pathlib import Path
from logger import Logger
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler


root_dir = Path(__file__).parent.parent
CONFIG_PATH = str(Path(root_dir, 'config.ini'))
DATA_PATH = str(Path(root_dir, 'data', 'processed_products.csv'))
MODEL_PATH = str(Path(root_dir, 'model'))


class Trainer:
    def __init__(self):
        self.logger = Logger().get_logger(__name__)
        
        # Конфигурация Spark
        config = configparser.ConfigParser()
        config.read(CONFIG_PATH)
        spark_conf = SparkConf().setAll(config['SPARK'].items())
        
        # Создаем сессию
        self.spark = SparkSession.builder \
            .appName("WordCountWithDataFrame") \
            .master("local[*]") \
            .config(conf=spark_conf) \
            .getOrCreate()
        
    def train_model(self, k=5):
        """Обучает алгоритм кластеризации kmeans"""
        
        # Считываем данные
        df = self.spark.read.option("header", True) \
               .option("sep", "\t") \
               .option("inferSchema", True) \
               .csv(DATA_PATH)
               
        # Преобразование данных в вектор
        assembler = VectorAssembler(inputCols=df.columns, outputCol="features")
        assembled_df = assembler.transform(df)
        
        # Нормализация
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
        scaler_model = scaler.fit(assembled_df)
        scaled_df = scaler_model.transform(assembled_df)
        
        # Обучаем модель
        # Эксперименты в ноутбуке показали, что оптимальное k=5
        kmeans = KMeans(k=k, seed=42, featuresCol="scaled_features")
        model = kmeans.fit(scaled_df)
        
        # Сохраняем модель
        model.write().overwrite().save(MODEL_PATH)
        self.logger.info("Модель успешно сохранена!")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_model(k=5)
    trainer.spark.stop()
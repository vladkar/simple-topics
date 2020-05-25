# -*- coding: utf-8 -*-

import os
from datetime import datetime

from nltk.corpus import stopwords
from nltk.stem.snowball import RussianStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from pyspark.ml.feature import NGram, StopWordsRemover
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, udf, expr, regexp_replace, concat, lower, size, lit
from pyspark.sql.types import StringType, ArrayType


def get_session(spark_config={}):
    builder = SparkSession.builder.appName('cradar')

    for key, val in spark_config.items():
        builder.config(key, val)

    session = builder.getOrCreate()

    return session


# dataframe transfom helper
def transform(self, f):
    return f(self)


# region Data persistence

def read_json(spark, file_in):
    return spark.read.json(file_in)
                          

def read_csv(spark, file_in):
    return spark.read.csv(file_in,
                          header=True, 
                          multiLine=True, 
                          ignoreLeadingWhiteSpace=True, 
                          ignoreTrailingWhiteSpace=True, 
                          encoding="UTF-8",
                          sep=',',
                          quote='"', 
                          escape='"')


def save_data(df, dir_out):
    if os.path.isdir(dir_out):
        for the_file in os.listdir(dir_out):
            file_path = os.path.join(dir_out, the_file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
                
    df.show(4)

    df.select([
        'text',
        'lemmas',
        'ngrams']).write.format('parquet').mode('overwrite').save(dir_out)


# endregion


# region Simple logger

def log(msg):
    print(f'{datetime.now():%Y-%m-%d %H:%M:%S.%f%z}: {msg}')


# endregion


# region NLP helpers

def tokenize(doc):
    tokenizer = RegexpTokenizer(r'[a-z0-9]+')
    tokens = tokenizer.tokenize(doc)
    res = list(filter(lambda x: len(x) < 15, tokens))
    return res


def stem_ru(doc):
    stemmer = RussianStemmer()
    res = [stemmer.stem(token) for token in doc]
    return res


def lemm(doc):
    lemmatizer = WordNetLemmatizer()
    res = [lemmatizer.lemmatize(token) for token in doc]
    return res


# endregion


# region Transformations

def to_lower(df):
    return df.withColumn('text', lower(df['reviewText']))


def clean(df):
    return df.withColumn('text', regexp_replace('text', '[^a-z]', ' '))


def tokenization(df):
    tokenize_nltk_udf = udf(tokenize, ArrayType(StringType()))
    return df.withColumn('tokens_noise', tokenize_nltk_udf(df.text))


def filter_stop_words(df):
    stop_words_nltk = stopwords.words('english')
    remover = StopWordsRemover(inputCol='tokens_noise', outputCol='tokens', stopWords=stop_words_nltk)

    return remover.transform(df)


def stemming(df):
    stem_nltk_udf = udf(lemm, ArrayType(StringType()))
    return df.withColumn('lemmas', stem_nltk_udf(df.tokens))


def filter_by_len(df):
    return df.withColumn('lemmas', expr('filter(lemmas, x -> (length(x) > 2) and (length(x) < 15))'))


def filter_empty(df):
    return df.withColumn('lemma_count', size(col('lemmas'))).filter(col('lemma_count') >= 1)


def ngrams(df):
    ngram = NGram(n=2, inputCol='lemmas', outputCol='ngrams')
    return ngram.transform(df)


# endregion


def transform_data(df):
    df = (df
          .transform(lambda df: to_lower(df))
          .transform(lambda df: clean(df))
          .transform(lambda df: tokenization(df))
          .transform(lambda df: filter_stop_words(df))
          .transform(lambda df: stemming(df))
          .transform(lambda df: filter_by_len(df))
          .transform(lambda df: filter_empty(df))
          .transform(lambda df: ngrams(df))
          )

    return df


def main():
    jsons_in = './data/amazon_reviews'
    dir_out = './df_spark'
    
    sample = True
    average_samples = 20000
    random = 42

    log('Entry point to ETL')
    spark = get_session()
    DataFrame.transform = transform
    

    log('ETL started')
    df = read_json(spark, jsons_in)
    
    if sample:
        df = df.sample(False, average_samples * 1.0 / df.count(), seed=random)
    
    df.show(5)
    
    df = transform_data(df)
    log('Processing is done')
    
    save_data(df, dir_out)
    
    spark.stop()
    log('Data was saved')


if __name__ == '__main__':
    main()

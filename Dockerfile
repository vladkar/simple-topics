FROM jupyter/pyspark-notebook
LABEL maintainer="<vladislav.k.work@gmail.com>"

# IPYTHON
USER root
RUN /opt/conda/bin/conda install jupyter -y --quiet && mkdir /opt/notebooks

EXPOSE 8888

# remove auth
RUN /opt/conda/bin/jupyter notebook --allow-root --generate-config --y \
	&& echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py

RUN pip install findspark
RUN pip install gensim
RUN pip install nltk
RUN [ "python", "-c", "import nltk; nltk.download('all')" ]
RUN pip install pyldavis

COPY ./data /opt/notebooks/data
COPY ./ETL /opt/notebooks/ETL
COPY ./analysis /opt/notebooks

CMD /opt/conda/bin/jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --allow-root --port=8888 --no-browser
FROM jupyter/scipy-notebook
WORKDIR /home/jovyan/work
RUN pip install PyQt5
ADD . /home/jovyan/work
CMD ["start-notebook.sh"]
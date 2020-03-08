ARG BASE_CONTAINER=jupyter/scipy-notebook
FROM $BASE_CONTAINER

COPY requirements.txt $HOME

# Install Tensorflow
RUN pip install -r requirements.txt && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

USER root

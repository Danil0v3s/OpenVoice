FROM continuumio/anaconda3

RUN apt-get update && apt-get install -y wget build-essential gcc libsndfile1 ffmpeg

WORKDIR /app
COPY . /app

RUN conda create -n openvoice python=3.9
RUN echo "source activate openvoice" > ~/.bashrc
ENV PATH=/opt/conda/envs/openvoice/bin:$PATH

RUN pip install jupyter
RUN pip install git+https://github.com/Danil0v3s/MeloTTS.git@openvoice
RUN python -m unidic download
RUN pip install -e .

EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]

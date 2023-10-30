<h1>FaceMask Detection</h1>
<div align="center" dir="auto">
<a href="https://pytorch.org/get-started/locally/" rel="nofollow"><img src="https://camo.githubusercontent.com/5b90a2636e7d3247534bdc67c391162fe068def2780192540c72c5c4cb7382cc/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5059544f5243482d312e31302b2d7265643f7374796c653d666f722d7468652d6261646765266c6f676f3d7079746f726368" alt="PyTorch - Version" data-canonical-src="https://img.shields.io/badge/PYTORCH-1.10+-red?style=for-the-badge&amp;logo=pytorch" style="max-width: 100%;"></a>
<a href="https://www.python.org/downloads/" rel="nofollow"><img src="https://camo.githubusercontent.com/9563a47966e5e5d773f6221b3dbd3dc8c103c4001d80b4f05ca0beab94303f07/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f505954484f4e2d332e372b2d7265643f7374796c653d666f722d7468652d6261646765266c6f676f3d707974686f6e266c6f676f436f6c6f723d7768697465" alt="Python - Version" data-canonical-src="https://img.shields.io/badge/PYTHON-3.7+-red?style=for-the-badge&amp;logo=python&amp;logoColor=white" style="max-width: 100%;"></a>
<br></p>
</div>

<details open="">
  <summary>Table of Contents</summary>
  <ol dir="auto">
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#project-structure">Project Structure</a>
    </li>
    <li>
      <a href="#data-preparation">Data Preparation</a>
    </li>
    <li><a href="#Install">How to run repository</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

<h2 tabindex="-1" id="user-content-about-the-project" dir="auto"><a class="heading-link" href="#about-the-project">About The Project<svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></h2>

<p dir="auto">In this project we will use deep learning to detect the human with facemask or no facemask</p>


<h2 tabindex="-1" id="user-content-about-the-project" dir="auto"><a class="heading-link" href="#Install">How to run repository<svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></h2>
  <h3>1.For training</h3>
  <p>+Step0: Install Anaconda or Miniconda</p>
  <p>+Step1: Open anaconda prompt</p>
  <p>+Step2: Create conda environment<p>
  <pre>conda create --name facemask python=3</pre>
  <p>+Step3:Clone this repository</p>
  <pre>git clone https://github.com/LuongTuanAnh163002/FaceMask_Detection.git</pre>
  <p>+Step4:Move to FaceMask_Detection folder</p>
  <pre>cd FaceMask_Detection.git</pre>
  <p>+Step5:Activate conda environment</p>
  <pre>conda activate facemask</pre>
  <p>+Step6: Install all packges need</p>
  <pre>pip install -r requirements.txt</pre>
  <p>+Step7: Run the code below to training</p>
  <pre>python train.py --epochs 50 --freeze</pre>
  <p>After you run and done training, all results save in runs/train/exp/..., folder runs automatic create after training done:</p>

  <h3>2.For detect</h3>
  <h4>2.1.Detect for video</h4>
  <p>+Detect with my model</p>
  <pre>python detect.py --demo 1 --source ~./file_name.mp4</pre>
  <p>+Detect with your model</p>
  <pre>python detect.py --demo 1 --weights file_model.h5 --source ~./file_name.mp4</pre>
  <h4>2.2.Detect for webcam</h4>
  <p>+Detect with my model</p>
  <pre>python detect.py --demo 0</pre>
  <p>+Detect with your model</p>
  <pre>python detect.py --weights file_model.h5 --demo 0</pre>
  <h3>2.For load tensorboard</h3>
  <pre>tensorboard --logdir=.../runs/train/exp/logs</pre>
<h2>Try with example in google colab</h2>
<a href="https://drive.google.com/file/d/1jUyWwxNbavW8ahRRY-kiNNF4Iphin9vT/view?usp=sharing" rel="nofollow"><img src="https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" style="max-width: 100%;"></a>

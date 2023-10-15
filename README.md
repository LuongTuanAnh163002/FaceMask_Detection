<h2>How to run repository</h2>
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

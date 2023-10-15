<h2>How to run repository with custom dataset</h2>
  <h3>1.For training</h3>
  <p>+Step0: Dowload data you need to classification, the struct of folder contain data to classification like data preparation part</p>
  <p>+Step1: Open CMD, terminal, anaconda prompt</p>
  <p>+Step2: Clone this repository<p>
  <pre>git clone https://github.com/LuongTuanAnh163002/Resnet_AnhLT.git</pre>
  <p>+Step3: Move to the Resnet_AnhLT folder</p>
  <pre>cd Resnet_AnhLT</pre>
  <p>+Step4: Go to "data" folder then create another file .yaml like dog_cat.yaml</p>
  <p>+Step5: Install all packges need</p>
  <pre>pip install -r requirements.txt</pre>
  <p>+Step6: Run the code below to training for pretrain</p>
  <pre>python train.py --model_type [resnet18, resnet34, resnet50, resnet101, resnet152] --pretrained --freeze --data file_name.yaml --epochs 50 --device [0, 1, 2,..]</pre>
  <p>+Step7: Run the code below to training for weight initialize</p>
  <pre>python train.py --model_type [resnet18, resnet34, resnet50, resnet101, resnet152] --data file_name.yaml --epochs 50 --device [0, 1, 2,..]</pre>
  <p>+Step8: Run the code below to training for using weight from previous train</p>
  <pre>python train.py --model_type [resnet18, resnet34, resnet50, resnet101, resnet152] --weight_init [file_name.pt, file_name.pth] --data file_name.yaml --epochs 50 --device [0, 1, 2,..]</pre>
  <p>After you run and done training, all results save in runs/train/exp/..., folder runs automatic create after training done:</p>

  <h3>2.For detect</h3>
  <p>+Detect for file</p>
  <pre>python detect.py --source file_name.jpg --weights ../runs/train/../weights/__.pth --device [0, 1, 2,..]</pre>
  <p>+Detect for folder</p>
  <pre>python detect.py --source path_folder --weights ../runs/train/../weights/__.pth --device [0, 1, 2,..]</pre>

##Traffic Sign Recognition

A web application for recognizing traffic signs using Spring Boot and TensorFlow, trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

##Features





Trains a convolutional neural network (CNN) to classify 43 traffic sign categories.



Displays real-time training progress, including loss and accuracy charts.



Allows users to upload images for traffic sign prediction with confidence scores.



Built with Spring Boot 3.2.5, TensorFlow 2.10.0, Tailwind CSS, and Chart.js.

##Prerequisites





Java: JDK 17, Maven



Python: 3.8 or higher, virtual environment



Dataset: GTSRB (download from http://benchmark.ini.rub.de/)



IDE: IntelliJ IDEA (recommended)

##Setup Instructions





1. Clone the Repository:

git clone https://github.com/<your-username>/traffic-sign-recognition.git
cd traffic-sign-recognition



2. Install Java Dependencies:

mvn clean install



3. Set Up Python Environment:

python3 -m venv venv
source venv/bin/activate
pip install tensorflow==2.10.0 numpy pandas pillow



## Download GTSRB Dataset:





Download GTSRB_Training_Images.zip from http://benchmark.ini.rub.de/.



Extract to GTSRB/Training/ in the project root.



##Configure IntelliJ IDEA:





Open the project in IntelliJ IDEA.



Set Python SDK: File > Project Structure > SDKs > + > Python SDK > venv/bin/python.



Create a Run Configuration for TrafficApplication.java.

## Running the Application

1. Run TrafficApplication.java to start the Spring Boot server.


2. Open http://localhost:8080 in a web browser.


## Train the Model:

Click "Start Training" to train the CNN on the GTSRB dataset.

Monitor progress, loss, and accuracy via the web interface.

Predict Traffic Signs:

Upload a traffic sign image (PPM, PNG, or JPG).

View the predicted sign and confidence score.

Project Structure

traffic-sign-recognition/
├── src/
│   ├── main/
│   │   ├── java/com/example/traffic/
│   │   │   ├── TrafficApplication.java
│   │   │   ├── controller/
│   │   │   │   ├── TrainingController.java
│   │   ├── resources/
│   │   │   ├── static/
│   │   │   │   ├── index.html
│   │   │   │   ├── train.py
│   │   │   ├── application.properties
├── pom.xml
├── README.md
├── .gitignore

## Troubleshooting

1. AVX Error: If TensorFlow reports missing AVX instructions:

Ensure tensorflow==2.10.0 is installed (pip show tensorflow).

Try TensorFlow 1.15 if issues persist (pip install tensorflow==1.15), but update train.py (contact for assistance).

Check CPU: sysctl -a | grep machdep.cpu.features. If no AVX, consider a different machine.

2. Dataset Issues:

Verify GTSRB/Training contains subfolders 00000/ to 00042/ with PPM images and CSV files.

Re-download from http://benchmark.ini.rub.de/ if necessary.

3. IntelliJ Errors:

Update to IntelliJ IDEA 2023.x (2021.1 is outdated).

Disable Python debugger if pydev errors occur: Edit Run Configuration > uncheck “Run with Python debugger.”

4. Web Interface Issues:

Check browser console (F12 > Console) for JavaScript errors.

Ensure train.py generates training_log.json and predictions.json.

5. Dataset

Source: German Traffic Sign Recognition Benchmark (GTSRB) - http://benchmark.ini.rub.de/

Classes: 43, as defined in train.py and illustrated at http://www.researchgate.net/profile/Reghunadhan-Rajesh/publication/252048039/figure/fig1/AS:650847071502351@1532185442921/classes-of-German-Traffic-Sign.png

Format: PPM images with CSV annotations

License


# Project Title

## Description

This project consists of three main components focusing on 3D inference, fingertip position detection, and pose detection using computer vision and machine learning techniques. It utilizes libraries such as OpenCV and MediaPipe for image processing and pose estimation.

The end goal of this project is to make 3D inference of real distance based on provided length and width of a mat as well as 2D pixel annotated point (for instanced if have a virtual mat identified through 2D annotation, if you can detect the 2D position of a person foot on the mat, the idea is to try to infer the read distance of the foot to each corner of the mat). Successful implementation of the project can potentially introduce a cost effective replacement of expensive medical measurement mat in demential research.
### Components

- **3D Inference**: Located in the `3D-infer` directory, this component is responsible for generating 3D inferences from input data.
- **Fingertip Position Detection**: Found in the `fingertip_position` directory, this module detects and records the positions of fingertips in images.
- **Pose Detection**: Situated in the `pose_detection` directory, this part of the project detects human poses from images and saves the results.

## Getting Started

### Dependencies

- Python 3.10
- OpenCV
- MediaPipe
- Pandas

Ensure you have Python 3.10 installed on your system. The required Python libraries can be installed using the `requirement.txt` file.

### Running

```sh
python3 -m venv venv
source ./venv/bin/activate
pip3 install -r requirements.txt
cd intending_folder
python3 main.py

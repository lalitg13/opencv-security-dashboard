# opencv python webcam facial recognition system

so i made this python project that turns my webcam into a home security camera. it uses machine learning to figure out who is standing in front of it, takes pictures of strangers, makes my mac make noise, and streams the video to a website so i can watch it from my phone.

## features

* **face detection & recognition**: uses openCV to spot faces and actually figure out if it's me or someone else. green box means friend, red box means stranger.
* **spy camera mode**: if an unknown person shows up, it waits a few seconds so it doesn't spam, takes a picture of them, saves it in a folder, and sends a notification to my mac.
* **web dashboard**: built a simple flask website so i can open up my browser and stream the live camera feed anywhere in the house.

## files in this folder

* `app.py`: runs the flask website server and streams the video.
* `collect_data.py`: takes like 30 pictures of your face so the computer learns what you look like.
* `train_model.py`: trains the ai model using all those pictures.
* `recognize_face.py`: the script that watches the webcam and labels who is there.
* `security_alert.py`: takes secret screenshots when it sees someone it doesn't recognize.
* `templates/index.html`: the html page for the web stream.

## how to run

make sure you have python, install openCV and flask in a virtual environment, and run python3 app.py.
from flask import Flask, redirect, url_for, request, render_template
import cv2

app = Flask(__name__)

@app.route('/success')
def success():

	camera_port = 0
	camera = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW) 
	return_value, image = camera.read()
	cv2.imwrite("image.bmp", image)
	    
	camera.release()
	cv2.destroyAllWindows()

	return 'Done'

@app.route('/home',methods = ['POST', 'GET'])
def home():
   if request.method == 'POST':
      return redirect(url_for('success'))
   else:
      return render_template('home.html')


if __name__ == '__main__':
   app.run(debug = True)
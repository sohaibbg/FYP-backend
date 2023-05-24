run command "python flask_app.py" to run
see FYP.postman_collection.json for help

POST localhost:5000/ssim-index
form-data 'img1' and 'img2'
response as {"response": 0.78324689}

POST localhost:5000/maskify
form-data 'img'
response as png img

POST localhost:5000/orient-and-crop
form-data 'img'
response as {"cropped": str1, "oriented" str2}
str1, str2 are ASCII encoded images
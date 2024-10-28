# Heath-Application
<h2>This ia an ongoing project</h2>

<h5>To set up this project in your local machine you can do the following steps:-</h5>
1. Download the project either as a zip file or using te git ssh<br/>
2. If you download the zip folder unzip it(not need if you do it via git)<br/>
3. There is a folder named dataset_train_tumor.zip you have to extract that folder<br/>
   Note: The dataset has been taken from https://www.kaggle.com/<br/>
4. Then you can run the brain_tumor_cnn_model.py<br/>
   i. Use one among the two given line if you have time and want more accurate data use the one currently being used else comment that line and uncomment the commented line:<br/>   
      training_model = image_model.fit(train_image_generator, steps_per_epoch = 100, epochs = 50, callbacks=[es, cp, lrr],
                                    validation_data = valid_image_generator)<br/>
      
      training_model = image_model.fit(train_image_generator, steps_per_epoch = 10, epochs = 5, callbacks=[es, cp, lrr],
                                        validation_data = valid_image_generator)<br/>
      (<b>Note: </b> Use one among the above two, currently one of them is commented and there are multiple place you have to look or this.)
5. Create a folder named "predict_image".<br/>
6. Then go the web-application folder and run the app.py file <br/>
7. You can view the application on http://127.0.0.1:5000/<br/>
8. You can now check the output.<br/>
# Note: If you don't run the brain_tumor_cnn_model.py before running the app.py you will not get any output or might get errors.
# This is an ongoing project so, you can also contribute if you wish. Send the merge request and will see and decide.

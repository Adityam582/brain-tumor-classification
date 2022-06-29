# NeuroAI: A deep learning project in the area of neuroscience

* This is an end-to-end machine learning lifecycle project, when I started from picking the dataset up to deploying as a web application. This helped me learn various nuance of a ML project development. 

* Dataset page: https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection

* We follow ML workflow as depicted in the figure below

![ML Workflow](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fstatic.packt-cdn.com%2Fproducts%2F9781788831307%2Fgraphics%2F13a6defd-b5b4-4062-aad0-cb7464630a3c.png&f=1&nofb=1)

Image Credits: big-data-and-business-intelligence, Packtpub [link](https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781788831307/1/ch01lvl1sec13/standard-ml-workflow)


* After the ML Model is created, we use transfer learning, with workflow as shown below:

Transfer Learning Workflow

![Transfer Learning Workflow](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.8bXjC-el8Yb7AwhgNqaLcwHaEK%26pid%3DApi&f=1)

Image Credits: Strata London - Deep Learning, Turi, Inc. [link](http://www.slideshare.net/turi-inc/strata-london-deep-learning-052015)

* A binary class image classifier based on the public dataset.

* The model uses transfer learning and builds on the State-Of-The-Art models: VGG16 and ResNet50. 

* We obtain a test accuracy of 92% and 96% respectively on VGG16 and ResNet50 models.

* Check the web application deployed using Streamlit and heroku.

* Click here to use application deployed on [Streamlit](https://adityam582-brain-tumor-classification-app-ulea8d.streamlitapp.com/)

* Click here to use application deployed on [Heroku](https://neuroai-image-classifier.herokuapp.com/)


# Why ?

When abnormal cells grow inside the brain, a brain tumor develops. The tumors in brain can be broadly classified as either malignant or benign. If identified in initial stages, proper medical treatment can help fast recovery. This deep learning model can be used to assist medical practitioners as prelimiary analysis of brain scan images to identify existence of tumor. Based on this, medical practitioners can take a better well-informed decision. 



# How ?

NeuroAI uses transfer learning to leverage the power of pre-trained models VGG16 and ResNet50, and building a deep neural network on top of it to obtain 92% and 96% accuracy respectively.

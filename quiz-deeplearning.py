#!/usr/bin/env python
# coding: utf-8

# QUIZ TOPIC - DEEP LEARNING
# 1. Which is the following is true about neurons?
# A. A neuron has a single input and only single output
# B. A neuron has multiple inputs and multiple outputs
# C. A neuron has a single input and multiple outputs
# D. All of the above
#     
# 
# 2. Which of the following is an example of deep learning?
# A. Self-driving cars
# B. Pattern recognition
# C. Natural language processing
# D. All of the above
# 
# 3. Which of the following statement is not correct?
# A. Neural networks mimic the human brain
# B. It can only work for a single input and a single output
# C. It can be used in image processing
# D. None
# 4. Autoencoder is an example of
# A. Deep learning
# B. Machine learning
# C. Data mining
# D. None
# 5. Which of the following deep learning models uses back propagation?
# A. Convolutional Neural Network
# B. Multilayer Perceptron Network
# C. Recurrent Neural Network
# D. All of the above
# 6. Which of the following steps can be taken to prevent overfitting in a neural network?
# A. Dropout of neurons
# B. Early stopping
# C. Batch normalization
# D. All of the above
# 7. Neural networks can be used inA. Regression problems
# B. Classification problems
# C. Clustering problems
# D. All of the above
# 8. In a classification problem, which of the following activation function is most widely used in the
# output layer of neural networks?
# A. Sigmoid function
# B. Hyperbolic function
# C. Rectifier function
# D. All of the above
# 9. Which of the following is a deep learning library?
# A. Tensorflow
# B. Keras
# C. PyTorch
# D. All of the above
# 10. Which of the following is true about bias?
# A. Bias is inherent in any predictive model
# B. Bias impacts the output of the neurons
# C. Both A and B
# D. None
# 11. What is the purpose of a loss function?
# A. Calculate the error value of the forward network
# B. Optimize the error values according to the error rate
# C. Both A and B
# D. None
# 12. Which of the following is a loss function?
# A. Sigmoid function
# B. Cross entropy
# C. ReLu
# D. All of the above
# 13. Which of the following loss function is used in regression?
# A. Logarithmic loss
# B. Cross entropy
# C. Mean squared error
# D. None
# 14. Suppose you have a dataset from where you have to predict three classes. Then which of the
# following configuration you should use in the output layer?
# A. Activation function = softmax, loss function = cross entropy
# B. Activation function = sigmoid, loss function = cross entropy
# C. Activation function = softmax, loss function = mean squared error
# D. Activation function = sigmoid, loss function = mean squared error
# 15. What is gradient descent?
# A. Activation function
# B. Loss function
# C. Optimization algorithm
# D. None
# 16. What does a gradient descent algorithm do?
# A. Tries to find the parameters of a model that minimizes the cost function
# B. Adjusts the weights at the input layers
# C. Both A and B
# D. None
# 17. Which of the following activation function can not be used in the output layer of an image
# classification model?
# A. ReLu
# B. Softmax
# C. Sigmoid
# D. None
# 18. For a binary classification problem, which of the following activation function is used?
# A. ReLu
# B. Softmax
# C. Sigmoid
# D. None
# 19. Which of the following makes a neural network non-linear?
# A. Convolution function
# B. Batch gradient descent
# C. Rectified linear unit
# D. All of the above
# 20. In a neural network, which of the following causes the loss not to decrease faster?
# A. Stuck at a local minima
# B. High regularization parameter
# C. Slow learning rate
# D. All of the above
# 21. For an image classification task, which of the following deep learning algorithm is best suited?
# A. Recurrent Neural Network
# B. Multi-Layer Perceptron
# C. Convolution Neural Network
# D. All of the above
# 22. Suppose the number of nodes in the input layer is 5 and the hidden layer is 10. The maximum
# number of connections from the input layer to the hidden layer would beA. More than 50
# B. Less than 50
# C. 50
# D. None
# 23. Which of the following is true about dropout?
# A. Applied in the hidden layer nodes
# B. Applied in the output layer nodes
# C. Both A and B
# D. None
# 24. Which of the following is a correct order for the Convolutional Neural Network operation?
# A. Convolution -> max pooling -> flattening -> full connection
# B. Max pooling -> convolution -> flattening -> full connection
# C. Flattening -> max pooling -> convolution -> full connection
# D. None
# 25. Convolutional Neural Network is used inA. Image classification
# B. Text classification
# C. Computer vision
# D. All of the above
# 26. Which of the following neural network model has a shared weight structure?
# A. Recurrent Neural Network
# B. Convolution Neural Network
# C. Both A and B
# D. None
# 27. LSTM is a variation ofA. Convolutional Neural Network
# B. Recurrent Neural Network
# C. Multi Layer Perceptron Network
# D. None
# 28. Which of the following neural networks is the best for machine translation?
# A. 1D Convolutional Neural Network
# B. 2D Convolutional Neural Network
# C. Recurrent Neural Network
# D. None
# 29. Which of the following neural networks has a memory?
# A. 1D CNN
# B. 2D CNN
# C. LSTM
# D. None
# 30. Batch normalization helps to preventA. activation functions to become too high or low
# B. the training speed to become too slow
# C. Both A and B
# D. None
# 
# 
# QUIZ KONUSU - DERİN ÖĞRENME
# 1. Nöronlar için aşağıdakilerden hangisi doğrudur?
# A. Bir nöronun tek girişi ve tek çıkışı vardır
# B. Bir nöronun birden çok girişi ve birden çok çıkışı vardır
# C. Bir nöronun tek bir girişi ve birden fazla çıkışı vardır
# D. Yukarıdakilerin hepsi
#     
# 
# 2. Aşağıdakilerden hangisi derin öğrenmeye bir örnektir?
# A. Sürücüsüz arabalar
# B. Örüntü tanıma
# C. Doğal dil işleme
# D. Yukarıdakilerin hepsi
# 
# 3. Aşağıdaki ifadelerden hangisi doğru değildir?
# A. Sinir ağları insan beynini taklit eder
# B. Yalnızca tek bir giriş ve tek bir çıkış için çalışabilir
# C. Görüntü işlemede kullanılabilir
# D. Yok
# 4. Otomatik kodlayıcı bir örnektir
# A. Derin öğrenme
# B. Makine öğrenimi
# C. Veri madenciliği
# D. Yok
# 5. Aşağıdaki derin öğrenme modellerinden hangisinde geriye yayılım kullanılır?
# A. Konvolüsyonel Sinir Ağı
# B. Çok Katmanlı Algılayıcı Ağı
# C. Tekrarlayan Sinir Ağı
# D. Yukarıdakilerin hepsi
# 6. Bir sinir ağında fazla uydurmayı önlemek için aşağıdaki adımlardan hangisi yapılabilir?
# A. Nöronların düşmesi
# B. Erken durdurma
# C. Toplu normalleştirme
# D. Yukarıdakilerin hepsi
# 7. Sinir ağları A'da kullanılabilir. Regresyon sorunları
# B. Sınıflandırma sorunları
# C. Kümeleme sorunları
# D. Yukarıdakilerin hepsi
# 8. Bir sınıflandırma probleminde, aşağıdaki aktivasyon fonksiyonlarından hangisi en yaygın olarak kullanılır?
# sinir ağlarının çıktı katmanı?
# A. Sigmoid işlevi
# B. Hiperbolik fonksiyon
# C. Doğrultucu işlevi
# D. Yukarıdakilerin hepsi
# 9. Aşağıdakilerden hangisi bir derin öğrenme kitaplığıdır?
# A. Tensör akışı
# B. Keras
# C. PyTorch
# D. Yukarıdakilerin hepsi
# 10. Önyargı hakkında aşağıdakilerden hangisi doğrudur?
# A. Yanlılık, herhangi bir tahmine dayalı modelin doğasında vardır
# B. Önyargı, nöronların çıktısını etkiler
# C. Hem A hem de B
# D. Yok
# 11. Bir kayıp fonksiyonunun amacı nedir?
# A. Yönlendirme ağının hata değerini hesaplayın
# B. Hata değerlerini hata oranına göre optimize edin
# C. Hem A hem de B
# D. Yok
# 12. Aşağıdakilerden hangisi bir kayıp fonksiyonudur?
# A. Sigmoid işlevi
# B. Çapraz entropi
# C. ReLu
# D. Yukarıdakilerin hepsi
# 13. Aşağıdaki kayıp fonksiyonlarından hangisi regresyonda kullanılır?
# A. Logaritmik kayıp
# B. Çapraz entropi
# C. Ortalama kare hatası
# D. Yok
# 14. Üç sınıfı tahmin etmeniz gereken bir veri kümeniz olduğunu varsayalım. O zaman hangi
# çıktı katmanında aşağıdaki yapılandırmayı kullanmalısınız?
# A. Aktivasyon fonksiyonu = softmax, kayıp fonksiyonu = çapraz entropi
# B. Aktivasyon fonksiyonu = sigmoid, kayıp fonksiyonu = çapraz entropi
# C. Aktivasyon fonksiyonu = softmax, kayıp fonksiyonu = ortalama karesel hata
# D. Etkinleştirme işlevi = sigmoid, kayıp işlevi = ortalama karesel hata
# 15. Gradyan iniş nedir?
# A. Etkinleştirme işlevi
# B. Kayıp fonksiyonu
# C. Optimizasyon algoritması
# D. Yok
# 16. Gradyan iniş algoritması ne yapar?
# A. Maliyet fonksiyonunu en aza indiren bir modelin parametrelerini bulmaya çalışır
# B. Girdi katmanlarındaki ağırlıkları ayarlar
# C. Hem A hem de B
# D. Yok
# 17. Bir görüntünün çıktı katmanında aşağıdaki aktivasyon fonksiyonlarından hangisi kullanılamaz?
# sınıflandırma modeli?
# A.Relu
# B. Softmax
# C. Sigmoid
# D. Yok
# 18. Bir ikili sınıflandırma problemi için aşağıdaki aktivasyon fonksiyonundan hangisi kullanılır?
# A.Relu
# B. Softmax
# C. Sigmoid
# D. Yok
# 19. Aşağıdakilerden hangisi bir sinir ağını doğrusal olmayan hale getirir?
# A. Evrişim işlevi
# B. Toplu gradyan iniş
# C. Doğrultulmuş doğrusal birim
# D. Yukarıdakilerin hepsi
# 20. Bir sinir ağında aşağıdakilerden hangisi kaybın daha hızlı azalmamasına neden olur?
# A. Yerel bir minimumda takılıp kalmak
# B. Yüksek düzenlileştirme parametresi
# C. Yavaş öğrenme oranı
# D. Yukarıdakilerin hepsi
# 21. Bir görüntü sınıflandırma görevi için aşağıdaki derin öğrenme algoritmalarından hangisi en uygunudur?
# A. Tekrarlayan Sinir Ağı
# B. Çok Katmanlı Algılayıcı
# C. Konvolüsyon Sinir Ağı
# D. Yukarıdakilerin hepsi
# 22. Girdi katmanındaki düğüm sayısı 5, gizli katmandaki düğüm sayısı 10 olsun.
# giriş katmanından gizli katmana bağlantı sayısı A olacaktır. 50'den fazla
# B. 50'den az
# C.50
# D. Yok
# 23. Terk hakkında aşağıdakilerden hangisi doğrudur?
# A. Gizli katman düğümlerinde uygulanan
# B. Çıktı katmanı düğümlerinde uygulanan
# C. Hem A hem de B
# D. Yok
# 24. Aşağıdakilerden hangisi Konvolüsyonel Sinir Ağı işlemi için doğru sıralamadır?
# A. Evrişim -> maksimum havuzlama -> düzleştirme -> tam bağlantı
# B. Maksimum havuzlama -> evrişim -> düzleştirme -> tam bağlantı
# C. Düzleştirme -> maksimum havuzlama -> evrişim -> tam bağlantı
# D. Yok
# 25. A'da Konvolüsyonel Sinir Ağı kullanılmıştır. Görüntü sınıflandırması
# B. Metin sınıflandırması
# C. Bilgisayar görüşü
# D. Yukarıdakilerin hepsi
# 26. Aşağıdaki sinir ağı modellerinden hangisinin ortak ağırlık yapısı vardır?
# A. Tekrarlayan Sinir Ağı
# B. Konvolüsyon Sinir Ağı
# C. Hem A hem de B
# D. Yok
# 27. LSTM, A'nın bir varyasyonudur. Evrişimli Sinir Ağı
# B. Tekrarlayan Sinir Ağı
# C. Çok Katmanlı Algılayıcı Ağı
# D. Yok
# 28. Aşağıdaki sinir ağlarından hangisi makine çevirisi için en iyisidir?
# A. 1D Konvolüsyonel Sinir Ağı
# B. 2B Evrişimli Sinir Ağı
# C. Tekrarlayan Sinir Ağı
# D. Yok
# 29. Aşağıdaki sinir ağlarından hangisinin hafızası vardır?
# A. 1D CNN
# B. 2D CNN
# C. LSTM
# D. Yok
# 30. Yarasa
# 
# Answer: C. A neuron has a single input and multiple outputs.
# 
# Answer: D. All of the above.
# 
# Answer: B. It can only work for a single input and a single output.
# 
# Answer: A. Deep learning.
# 
# Answer: D. All of the above.
# 
# Answer: D. All of the above.
# 
# Answer: D. All of the above.
# 
# Answer: A. Sigmoid function.
# 
# Answer: D. All of the above.
# 
# Answer: C. Both A and B.
# 
# Answer: C. Both A and B.
# 
# Answer: B. Cross entropy.
# 
# Answer: C. Mean squared error.
# 
# Answer: A. Activation function = softmax, loss function = cross entropy.
# 
# Answer: C. Optimization algorithm.
# 
# Answer: A. Tries to find the parameters of a model that minimizes the cost function.
# 
# Answer: A. ReLu.
# 
# Answer: C. Sigmoid.
# 
# Answer: D. All of the above.
# 
# Answer: D. All of the above.
# 
# Answer: C. Convolution Neural Network.
# 
# Answer: A. More than 50.
# 
# Answer: A. Applied in the hidden layer nodes.
# 
# Answer: A. Convolution -> max pooling -> flattening -> full connection.
# 
# Answer: D. All of the above.
# 
# Answer: C. Both A and B.
# 
# Answer: B. Recurrent Neural Network.
# 
# Answer is C. Recurrent Neural Network.
# 
# Answer is C. LSTM.
# 
# Answer is A. Activation functions to become too high or low.

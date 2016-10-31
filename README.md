# FCN_via_Keras

## FCN

pythonとkerasを使ってFCNを実装しました。こちらがFCNの論文です。(https://arxiv.org/pdf/1605.06211v1.pdf) FCNは様々な大きさの画像をインプットすることができますが、今回は簡単のため(224*224)ピクセルに大きさを固定して学習させています。

FCN (Fully Convolutional Network) is deep fully convolutional neural network architecture for semantic pixel-wise segmentation. This is implementation of "https://arxiv.org/abs/1605.06211" by using Keras which is a neural networks library. FCN can train by using any size of image, but I trained this network using the images of the same size (224 * 224).

## Architecture of Model
![Model](https://github.com/k3nt0w/garage/blob/master/img/FCN_model.png "Model_of_FCN")

## Caution

私は普段からバックエンドにtheanoを使っているので、あとから気づいたのですが、tensorflowだと動かないみたいです。Deconvolution2Dでerrorを吐きます。もしこんなクソコードでも参考にしてくれる人がいるならば、バックエンドはtheanoを指定して実行してください。(一応、実行する時にtheanoが選ばれるようになっています。)

時間があれば、tensorflowでもちゃんと動くように直したいとおもいます。また根本的に間違っているかもしれないので原因がわかる方はこちらの[ブログ](http://ket-30.hatenablog.com)にコメントしていただくと、ありがたいです。

Please use theano as backend when you use this code because this couldn't work on tensorflow backend. I'm trying debug now. I update this code if I get factor of that.

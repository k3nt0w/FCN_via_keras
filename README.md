# FCN_via_keras

## FCN

これはpythonとkerasを使ってFCNを実装したものです。こちらがFCNの論文です。(https://arxiv.org/pdf/1605.06211v1.pdf) FCNは様々な大きさの画像をインプットすることができますが、今回は簡単のため(224*224)ピクセルに大きさを固定して学習させています。

FCN (Fully Convolutional Network) is deep fully convolutional neural network architecture for semantic pixelwise segmentation. This is implementation of "https://arxiv.org/abs/1605.06211" by using Keras which is a neural networks library. FCN can train by using any size of image, but I trained this network using the images of the same size (224 * 224).

## Architecture of model
Markdown: ![Model](https://github.com/k3nt0w/garage/blob/master/img/FCN_model.png "Model_of_FCN")

## coution

私は普段からバックエンドにtensorflowを使っているので、あとで気づいたのですが、theanoだと動かないみたいです。もしこんなクソコードでも参考にしてくれる人がいるならば、バックエンドはtensorflowを指定して実行してください。
"""
KERAS_BACKEND=tensorflow python train.py
"""
上のようにoptionを設定すれば大丈夫です。  
時間があれば、theanoでもちゃんと動くように直したいとおもいます。また根本的に間違っているかもしれないので原因がわかる方はこちらの[ブログ](http://ket-30.hatenablog.com)にコメントしていただくと、ありがたいです。

Please use tensorflow as backend when you use this code because this couldn't work on theano backend. You can change backend by writing like this.
"""
KERAS_BACKEND=tensorflow python train.py
"""
I'm trying debug now. I update this code if I get factor of that.

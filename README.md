# branch_detection_pycpp
## Deep Learningによる枝検出をC++から動かすプログラム

pythonで記述したDeep Learningによる枝領域の認識を，pybind11というライブラリを介してC++から操作します．
プログラムの流れはコメントアウトを確認してください．ちなみにC++からPythonへは画像データを渡していますが，
どうやらcv::Mat型からndarray(numpy配列)に自動で変換してくれるようです．この辺はよくわかってないところもありますが，
便利ですねってことで許してください．この時に使っているプログラムは[こちら](https://github.com/edmBernard/pybind11_opencv_numpy)をお借りしました.<br>

GPUで処理をすると約160 msで動くので，Python単体で動かしたときと速度に大きな差はないです．これで，Deep Learning以外の，
システムで速度が欲しいところはC++で書いてOKということになりました．<br>

現在Deep Learningの記述にはTensorflowのラッパーであるkerasを使用しています．Tensorflowを生で扱うより記述が簡便なため，これを採用していますが，
やはり速度では劣る部分があります．高速化を目指すなら，
[[Python]KerasをTensorFlowから，TensorFlowをc++から叩いて実行速度を上げる](https://qiita.com/yukiB/items/1ea109eceda59b26cd64)を読むといいでしょう．<br>

## Python環境
- Python 3.7(Anaconda)
- tensorflow-gpu 2.0.0
- keras 2.3.0
- pybind11 2.4

## 注意点
GPUを使う場合は，C++の実行ファイル(.exe)と同じディレクトリにcudart64_100.dll, cudnn64_7.dllを置いてください．GPUをもともと積んでいるPCなら，
エクスプローラで検索すればどこかにあるはずです．

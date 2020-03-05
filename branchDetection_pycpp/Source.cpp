//https://takuyaokada.hatenablog.com/entry/20190412/1555004824

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <fstream>
#include <sstream>
#include <typeinfo>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <chrono>

#include "ndarray_converter.h"
#include <Python.h>
#include <numpy/ndarrayobject.h>

namespace py = pybind11;

int main() {
	//画像の読み込み
	cv::Mat mat = cv::imread("D:/dev/PyWork/semantic_segmentation/testdata/color_52113416.jpg", 1);
	
	//推論のためにサイズを調整する
	cv::resize(mat, mat, cv::Size(480, 320));

	//推論時に使うライブラリがRGBのチャネルを使っているのでBGRからRGBへ変換する
	cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
	
	//確認
	std::cout << mat.size() << std::endl;
	cv::imshow("raw", mat);

	/////////////////////////////////////////////
	// ここからPythonをC++から動かすための処理 //
	/////////////////////////////////////////////
	
	//python interpreterを起動
	py::scoped_interpreter interpreter;

	//Python script fileを読み込む
	py::object global_ns = py::module::import("__main__").attr("__dict__");
	std::ifstream ifs("predict_tool.py");
	std::string script((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
	py::exec(script);

	//Python script file内にある関数をC++側で再定義
	py::object pythonFunc = global_ns["predict"];

	//なんかこれを使うとcv::Mat型をpythonに送れる
	NDArrayConverter converter;
	converter.init_numpy();

	//時間計測
	auto start = std::chrono::system_clock::now();
	
	//推論実行
	auto res = pythonFunc(mat);
	
	//時間計測，表示
	auto end = std::chrono::system_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

	//1回目の推論はGPU起動のためか時間がかかるので2回目を行いその時間を計測確認する
	start = std::chrono::system_clock::now();
	res = pythonFunc(mat);
	end = std::chrono::system_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

	//返り値をndarrayに変換
	py::array_t<double> a = res;
	for (int i = 0; i < a.shape(1); i++) {
		for (int j = 0; j < a.shape(0); j++) {
			//コンソール出力するなり，vector型に格納するなり
			//std::cout << a.at(j, i);
		}
		//std::cout << std::endl;
	}

	return 0;
}
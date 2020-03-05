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
	//�摜�̓ǂݍ���
	cv::Mat mat = cv::imread("D:/dev/PyWork/semantic_segmentation/testdata/color_52113416.jpg", 1);
	
	//���_�̂��߂ɃT�C�Y�𒲐�����
	cv::resize(mat, mat, cv::Size(480, 320));

	//���_���Ɏg�����C�u������RGB�̃`���l�����g���Ă���̂�BGR����RGB�֕ϊ�����
	cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
	
	//�m�F
	std::cout << mat.size() << std::endl;
	cv::imshow("raw", mat);

	/////////////////////////////////////////////
	// ��������Python��C++���瓮�������߂̏��� //
	/////////////////////////////////////////////
	
	//python interpreter���N��
	py::scoped_interpreter interpreter;

	//Python script file��ǂݍ���
	py::object global_ns = py::module::import("__main__").attr("__dict__");
	std::ifstream ifs("predict_tool.py");
	std::string script((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
	py::exec(script);

	//Python script file���ɂ���֐���C++���ōĒ�`
	py::object pythonFunc = global_ns["predict"];

	//�Ȃ񂩂�����g����cv::Mat�^��python�ɑ����
	NDArrayConverter converter;
	converter.init_numpy();

	//���Ԍv��
	auto start = std::chrono::system_clock::now();
	
	//���_���s
	auto res = pythonFunc(mat);
	
	//���Ԍv���C�\��
	auto end = std::chrono::system_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

	//1��ڂ̐��_��GPU�N���̂��߂����Ԃ�������̂�2��ڂ��s�����̎��Ԃ��v���m�F����
	start = std::chrono::system_clock::now();
	res = pythonFunc(mat);
	end = std::chrono::system_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

	//�Ԃ�l��ndarray�ɕϊ�
	py::array_t<double> a = res;
	for (int i = 0; i < a.shape(1); i++) {
		for (int j = 0; j < a.shape(0); j++) {
			//�R���\�[���o�͂���Ȃ�Cvector�^�Ɋi�[����Ȃ�
			//std::cout << a.at(j, i);
		}
		//std::cout << std::endl;
	}

	return 0;
}
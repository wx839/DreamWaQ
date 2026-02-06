// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <mutex>

namespace isaaclab
{

class Algorithms
{
public:
    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;

    std::vector<float> get_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return action;
    }
    
    std::vector<float> action;
protected:
    std::mutex act_mtx_;
};

class OrtRunner : public Algorithms
{
public:
    OrtRunner(std::string model_path)  //OrtRunner类的构造函数 
    {
        // Init Model
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        //这里可能要多加一个session 
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            Ort::TypeInfo input_type = session->GetInputTypeInfo(i);
            input_shapes.push_back(input_type.GetTensorTypeAndShapeInfo().GetShape());
            auto input_name = session->GetInputNameAllocated(i, allocator); 
            // 使用 std::string 存储名称，避免内存泄漏 
            input_names_storage.push_back(std::string(input_name.get())); 
        }
        // 构建 const char* 数组用于 ONNX Runtime API
        for (const auto& name : input_names_storage) {
            input_names.push_back(name.c_str());
        }

        for (const auto& shape : input_shapes) {
            size_t size = 1;
            for (const auto& dim : shape) {
                size *= dim;
            }
            input_sizes.push_back(size);
        }

        // Get output shape
        Ort::TypeInfo output_type = session->GetOutputTypeInfo(0);
        output_shape = output_type.GetTensorTypeAndShapeInfo().GetShape();
        auto output_name = session->GetOutputNameAllocated(0, allocator);
        // 使用 std::string 存储名称，避免内存泄漏 
        output_names_storage.push_back(std::string(output_name.get()));
        for (const auto& name : output_names_storage) { 
            output_names.push_back(name.c_str()); 
        } 
        action.resize(output_shape[1]);
    }

    // 析构函数：不再需要手动释放内存，std::string 会自动管理 
    ~OrtRunner() = default;

    //修改推理框架----------------------------------------------注意obs必须满足要求     
    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs)
    {

        // std::fill(obs.begin(), obs.end(), 1.0f); 
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // make sure all input names are in obs
        for (const auto& name : input_names_storage) { 
            if (obs.find(name) == obs.end()) {
                throw std::runtime_error("Input name " + name + " not found in observations."); 
            }
        }

        // Create input tensors
        std::vector<Ort::Value> input_tensors;
        input_tensors.reserve(input_names.size());
        for(size_t i(0); i < input_names.size(); ++i) 
        {
            auto& input_data = obs.at(input_names_storage[i]);
            auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_sizes[i], input_shapes[i].data(), input_shapes[i].size());
            input_tensors.push_back(std::move(input_tensor));
        }

        // Run the model
        auto output_tensor = session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), 1);

        // Copy output data
        auto floatarr = output_tensor.front().GetTensorMutableData<float>();
        std::lock_guard<std::mutex> lock(act_mtx_);
        std::memcpy(action.data(), floatarr, output_shape[1] * sizeof(float));
        return action;
    }

private:
    Ort::Env env;
    //这个是session所配置的线程数 
    Ort::SessionOptions session_options;
    //加一个新的 
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    // 使用 std::string 存储名称，自动管理内存 
    std::vector<std::string> input_names_storage;
    std::vector<std::string> output_names_storage;
    // const char* 指针数组，指向 storage 中的字符串 
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<int64_t> input_sizes;
    std::vector<int64_t> output_shape;
};
};
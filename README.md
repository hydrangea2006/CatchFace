# CatchFace,一个简单实现提取脸部并实时渲染到3D模型上的项目

本项目以opencv与mediapipe捕捉面部关键点为核心，用笔记本自带的摄像头获取到面部数据后，实时解算52个标准ARKit表情系数，同时用PnP算法计算头部相对于摄像头的旋转向量与位移向量，通过Socket.IO协议传输给前端实时渲染3D模型的表情与头部模型的旋转与位置

---

## 团队分工

* **算法**：负责Opencv采集，mediapipe推理以及滤波函数
* **后端**：负责搭建Flask-Socket.IO服务器转发及跨域配置，与表情系数映射
* **前端**：负责react + R3F模型渲染

## 🛠️开发环境
* **Python版本**：3.12.10
* 开发环境：Windows11 + vs code
* <mark>核心依赖</mark> :Opencv, mediapipe, Sockect.IO, react, R3F
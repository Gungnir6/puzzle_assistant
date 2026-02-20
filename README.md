# Puzzle Assistant

## 项目简介

Puzzle Assistant 是一款基于 Web 的自动化解谜辅助工具，旨在通过计算机视觉技术自动化终末地中故障机器人谜题的求解，用户通过上传游戏截图，系统利用 OpenCV.js 自动提取地图的行列维度及内部数字信息，并通过剪枝DFS算法给出可行解

## 技术栈

本项目采用了现代前端主流开发架构：

* 前端框架：Vue3
* 构建工具：Vite
* 图像处理：OpenCV.js
* 编程语言：JavaScript
* 样式处理：CSS3

## 使用说明

推荐访问 **[https://puzzle-assistant.pages.dev](https://puzzle-assistant.pages.dev)** 直接使用

### 1. 环境准备
确保你的开发环境已安装 Node.js (建议版本 16.0.0 或更高)

### 2. 克隆与安装
使用 Git 克隆项目到本地：
```cmd
git clone https://github.com/Gungnir6/puzzle_assistant.git
cd puzzle_assistant
```

安装项目依赖：
```cmd
npm install
```

### 3. 运行开发服务器
在本地启动预览：
```cmd
npm run dev
```

启动后，访问浏览器显示的本地地址 (通常为 http://localhost:5173)

### 4. 功能操作步骤
1. 进入应用页面后，点击上传区域选择游戏截图，同时支持拖拽上传图片与Ctrl+V粘贴图片

   > 图片要求：选择**数字**而非**图形**显示模式，图片完整较为清晰

2. 系统将调用 OpenCV.js 识别地图的行数、列数及障碍物和锁定块信息

3. 如果数字识别出现问题，可以手动调整：

   - 移动端：点击数字即可循环+1，或者长按数字并上下滑动以调整数字
   - 电脑端：除上述方法外，还可将鼠标悬浮至数字上并滑动鼠标滚轮以调整数字

4. 确认识别结果无误后，点击求解按钮

5. 页面将直观展示最终可行答案（只展示一种）

## 关于项目

本项目仅供学习交流，请勿用于商业用途

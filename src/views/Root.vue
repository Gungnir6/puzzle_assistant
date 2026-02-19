<script setup>
import { ref, onUnmounted, nextTick } from 'vue'

const imageUrl = ref('');
const inputImage = ref(null);
const fileInput = ref(null);
const currentFile = ref(null);

const detectMode = ref('single');
const currentStep = ref('upload');
const matrixData = ref({ rows: [], cols: [] });
const statusText = ref('');
const debugCanvas = ref(null);

const savedSingleBoxes = ref({ rows: [], cols: [] });
const savedDoubleBoxes = ref({ rows: [], cols: [] });

let worker = null;

const triggerUpload = () => {
  if (fileInput.value) fileInput.value.click()
}

const handleFileChange = (event) => {
  const file = event.target.files[0]
  if (file && file.type.startsWith('image/')) {
    currentFile.value = file
    if (imageUrl.value) URL.revokeObjectURL(imageUrl.value)
    imageUrl.value = URL.createObjectURL(file)

    currentStep.value = 'processing';
    matrixData.value = { rows: [], cols: [] };

    setTimeout(() => {
      runAnalysis();
    }, 50);
  }
}

onUnmounted(async () => {
  if (worker) {
    await worker.terminate();
  }
});

const recognizeDigit = async (inputMat, workerInstance) => {
  let mats = [];
  let canvas = document.createElement('canvas');

  try {
    let { maxVal } = cv.minMaxLoc(inputMat);
    if (maxVal < 150) return 0;

    let scaled = new cv.Mat();
    mats.push(scaled);
    let dsize = new cv.Size(inputMat.cols * 4, inputMat.rows * 4);
    cv.resize(inputMat, scaled, dsize, 0, 0, cv.INTER_CUBIC);

    let blurred = new cv.Mat();
    mats.push(blurred);
    cv.GaussianBlur(scaled, blurred, new cv.Size(3, 3), 0);

    let binary = new cv.Mat();
    mats.push(binary);

    cv.threshold(blurred, binary, 80, 255, cv.THRESH_BINARY);

    let inverted = new cv.Mat();
    mats.push(inverted);
    cv.bitwise_not(binary, inverted);

    let padded = new cv.Mat();
    mats.push(padded);
    let color = new cv.Scalar(255, 255, 255);
    cv.copyMakeBorder(inverted, padded, 30, 30, 30, 30, cv.BORDER_CONSTANT, color);

    cv.imshow(canvas, padded);

    await workerInstance.setParameters({
      tessedit_char_whitelist: '0123456789',
      tessedit_pageseg_mode: '10',
    });

    const { data: { text } } = await workerInstance.recognize(canvas);
    let cleanText = text.trim();

    if (!cleanText) return 0;
    let num = parseInt(cleanText);
    return isNaN(num) ? 0 : num;

  } catch (err) {
    console.error("Digit recognize error:", err);
    return 0;
  } finally {
    mats.forEach(m => {
      if (m && !m.isDeleted()) m.delete();
    });
  }
};

const detectMap = async (imgElement) => {
  let src = cv.imread(imgElement);
  let matsToRelease = [src];

  try {
    statusText.value = "正在提取单色网格位置...";
    const roiW = Math.floor(src.cols * 0.7);
    let rect = new cv.Rect(0, 0, roiW, src.rows);
    let roi = src.roi(rect);
    matsToRelease.push(roi);
    let debugMat = roi.clone();
    matsToRelease.push(debugMat);

    let hsv = new cv.Mat();
    cv.cvtColor(roi, hsv, cv.COLOR_RGBA2RGB);
    cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);
    matsToRelease.push(hsv);

    let channels = new cv.MatVector();
    cv.split(hsv, channels);
    let gray = channels.get(2);
    channels.delete();
    matsToRelease.push(gray);

    let mask = new cv.Mat();
    cv.threshold(gray, mask, 80, 255, cv.THRESH_BINARY);
    matsToRelease.push(mask);

    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();
    cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    matsToRelease.push(contours, hierarchy);

    let boxes = [];
    for (let i = 0; i < contours.size(); ++i) {
      let box = cv.boundingRect(contours.get(i));
      if (box.width > 6 && box.height > 6 && box.width < 100) {
        boxes.push(box);
      }
    }

    if (boxes.length < 5) throw new Error("未检测到足够数据");

    let minX = Math.min(...boxes.map(b => b.x));
    let minY = Math.min(...boxes.map(b => b.y));

    let rowCands = boxes.filter(b => b.x < minX + 30).sort((a, b) => a.y - b.y);
    let colCands = boxes.filter(b => b.y < minY + 30).sort((a, b) => a.x - b.x);

    if (colCands.length > 2) {
      let lastGap = colCands[colCands.length - 1].x - colCands[colCands.length - 2].x;
      let avgGap = (colCands[colCands.length - 2].x - colCands[0].x) / (colCands.length - 2);
      if (lastGap > avgGap * 2.5) colCands.pop();
    }

    let rowNums = rowCands;
    let colNums = colCands;

    for (const box of rowNums) {
      cv.rectangle(debugMat, new cv.Point(box.x, box.y), new cv.Point(box.x+box.width, box.y+box.height), [0, 255, 0, 255], 2);
    }
    for (const box of colNums) {
      cv.rectangle(debugMat, new cv.Point(box.x, box.y), new cv.Point(box.x+box.width, box.y+box.height), [255, 0, 0, 255], 2);
    }

    savedSingleBoxes.value = { rows: rowNums, cols: colNums };

    currentStep.value = 'debug';
    await nextTick();
    cv.imshow(debugCanvas.value, debugMat);
    statusText.value = `选区已提取。行:${rowNums.length} 列:${colNums.length}`;

  } catch (e) {
    console.error(e);
    statusText.value = "提取失败";
  } finally {
    matsToRelease.forEach(m => { if (m && !m.isDeleted()) m.delete(); });
  }
};

const executeSingleOCR = async () => {
  currentStep.value = 'processing';
  statusText.value = "正在初始化单色 OCR...";

  let src = cv.imread(inputImage.value);
  let matsToRelease = [src];

  try {
    if (!worker) {
      worker = await Tesseract.createWorker('eng', 1, { logger: () => {} });
      await worker.setParameters({
        tessedit_char_whitelist: '0123456789Ø',
        tessedit_pageseg_mode: '10'
      });
    }

    const roiW = Math.floor(src.cols * 0.7);
    let rect = new cv.Rect(0, 0, roiW, src.rows);
    let roi = src.roi(rect);
    matsToRelease.push(roi);

    let hsv = new cv.Mat();
    cv.cvtColor(roi, hsv, cv.COLOR_RGBA2RGB);
    cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);
    matsToRelease.push(hsv);

    let channels = new cv.MatVector();
    cv.split(hsv, channels);
    let gray = channels.get(2);
    channels.delete();
    matsToRelease.push(gray);

    const rowValues = [];
    const colValues = [];
    const total = savedSingleBoxes.value.rows.length + savedSingleBoxes.value.cols.length;
    let count = 0;

    for (const box of savedSingleBoxes.value.rows) {
      statusText.value = `正在识别单色行... ${++count}/${total}`;
      let digitRoi = gray.roi(box);
      let val = await recognizeDigit(digitRoi, worker);
      rowValues.push(val);
      digitRoi.delete();
    }

    for (const box of savedSingleBoxes.value.cols) {
      statusText.value = `正在识别单色列... ${++count}/${total}`;
      let digitRoi = gray.roi(box);
      let val = await recognizeDigit(digitRoi, worker);
      colValues.push(val);
      digitRoi.delete();
    }

    matrixData.value = { rows: rowValues, cols: colValues };
    currentStep.value = 'result';
    statusText.value = "单色识别完成";

  } catch (e) {
    console.error(e);
    statusText.value = "识别发生错误";
  } finally {
    matsToRelease.forEach(m => { if (m && !m.isDeleted()) m.delete(); });
  }
};

const detectDoubleMapDebug = async (imgElement) => {
  let src = cv.imread(imgElement);
  let matsToRelease = [src];

  statusText.value = "提取双色网格位置...";

  try {
    const roiW = Math.floor(src.cols * 0.7);
    let rect = new cv.Rect(0, 0, roiW, src.rows);
    let roi = src.roi(rect);
    matsToRelease.push(roi);
    let debugMat = roi.clone();
    matsToRelease.push(debugMat);

    let hsv = new cv.Mat();
    cv.cvtColor(roi, hsv, cv.COLOR_RGBA2RGB);
    cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);
    matsToRelease.push(hsv);

    let channels = new cv.MatVector();
    cv.split(hsv, channels);
    let gray = channels.get(2);
    channels.delete();
    matsToRelease.push(gray);

    let mask = new cv.Mat();
    cv.threshold(gray, mask, 80, 255, cv.THRESH_BINARY);
    matsToRelease.push(mask);

    let hKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(40, 1));
    let hLines = new cv.Mat();
    cv.morphologyEx(mask, hLines, cv.MORPH_OPEN, hKernel);
    cv.subtract(mask, hLines, mask);

    let vKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(1, 6));
    cv.morphologyEx(mask, mask, cv.MORPH_CLOSE, vKernel);
    matsToRelease.push(hKernel, hLines, vKernel);

    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();
    cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    matsToRelease.push(contours, hierarchy);

    let boxes = [];
    for (let i = 0; i < contours.size(); ++i) {
      let box = cv.boundingRect(contours.get(i));
      if (box.width > 6 && box.height > 6 && box.width < 100) {
        boxes.push(box);
      }
    }

    if (boxes.length === 0) throw new Error("未检测到有效元素");

    let minY = Math.min(...boxes.map(b => b.y));

    let colCands = boxes.filter(b => b.y < minY + 30);
    colCands.sort((a, b) => a.x - b.x);

    if (colCands.length > 2) {
      let lastGap = colCands[colCands.length - 1].x - colCands[colCands.length - 2].x;
      let avgGap = (colCands[colCands.length - 2].x - colCands[0].x) / (colCands.length - 2);
      if (lastGap > avgGap * 2.5) colCands.pop();
    }

    let finalColPairs = [];
    for (let i = 0; i < colCands.length; i += 2) {
      if (i + 1 >= colCands.length) break;
      let leftBox = colCands[i];
      let rightBox = colCands[i+1];
      let centerX = (leftBox.x + rightBox.x + rightBox.width) / 2;
      finalColPairs.push({ left: leftBox, right: rightBox, centerX });

      cv.rectangle(debugMat, new cv.Point(leftBox.x, leftBox.y), new cv.Point(leftBox.x+leftBox.width, leftBox.y+leftBox.height), [0, 255, 0, 255], 2);
      cv.rectangle(debugMat, new cv.Point(rightBox.x, rightBox.y), new cv.Point(rightBox.x+rightBox.width, rightBox.y+rightBox.height), [255, 0, 0, 255], 2);
      cv.circle(debugMat, new cv.Point(centerX, minY + 30), 3, [0, 0, 255, 255], -1);
    }

    let mapBoundaryX = colCands.length > 0 ? colCands[0].x : 1000;
    let rowCands = boxes.filter(b => (b.x + b.width) < mapBoundaryX + 5);

    let finalRowPairs = [];
    if (rowCands.length > 0) {
      let maxRight = Math.max(...rowCands.map(b => b.x + b.width));
      let cleanRowBoxes = rowCands.filter(b => b.x + b.width > maxRight - 15);

      cleanRowBoxes.sort((a, b) => a.y - b.y);

      for (let i = 0; i < cleanRowBoxes.length; i += 2) {
        if (i + 1 >= cleanRowBoxes.length) break;
        let topBox = cleanRowBoxes[i];
        let bottomBox = cleanRowBoxes[i+1];

        let centerY = (topBox.y + bottomBox.y + bottomBox.height) / 2;
        finalRowPairs.push({ top: topBox, bottom: bottomBox, centerY });

        cv.rectangle(debugMat, new cv.Point(topBox.x, topBox.y), new cv.Point(topBox.x+topBox.width, topBox.y+topBox.height), [255, 0, 0, 255], 2);
        cv.rectangle(debugMat, new cv.Point(bottomBox.x, bottomBox.y), new cv.Point(bottomBox.x+bottomBox.width, bottomBox.y+bottomBox.height), [0, 255, 0, 255], 2);
        cv.circle(debugMat, new cv.Point(maxRight + 10, centerY), 3, [0, 0, 255, 255], -1);
      }
    }

    //保存坐标供后续 OCR 使用
    savedDoubleBoxes.value = { rows: finalRowPairs, cols: finalColPairs };

    //Debug
    currentStep.value = 'debug';
    await nextTick();
    cv.imshow(debugCanvas.value, debugMat);
    statusText.value = `选区已提取。行对数:${finalRowPairs.length} 列对数:${finalColPairs.length}`;

  } catch (e) {
    console.error(e);
    statusText.value = "提取失败";
  } finally {
    matsToRelease.forEach(m => { if (m && !m.isDeleted()) m.delete(); });
  }
};

const executeDoubleOCR = async () => {
  currentStep.value = 'processing';
  statusText.value = "正在初始化双色 OCR...";

  let src = cv.imread(inputImage.value);
  let matsToRelease = [src];

  try {
    if (!worker) {
      worker = await Tesseract.createWorker('eng', 1, { logger: () => {} });
      await worker.setParameters({ tessedit_char_whitelist: '0123456789Ø', tessedit_pageseg_mode: '10' });
    }

    const roiW = Math.floor(src.cols * 0.7);
    let rect = new cv.Rect(0, 0, roiW, src.rows);
    let roi = src.roi(rect);
    matsToRelease.push(roi);

    let hsv = new cv.Mat();
    cv.cvtColor(roi, hsv, cv.COLOR_RGBA2RGB);
    cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);
    matsToRelease.push(hsv);

    let channels = new cv.MatVector();
    cv.split(hsv, channels);
    let gray = channels.get(2);
    channels.delete();
    matsToRelease.push(gray);

    let mask = new cv.Mat();
    cv.threshold(gray, mask, 90, 255, cv.THRESH_BINARY);
    matsToRelease.push(mask);

    let hKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(30, 1));
    let hLines = new cv.Mat();
    cv.morphologyEx(mask, hLines, cv.MORPH_OPEN, hKernel);
    cv.subtract(mask, hLines, mask);

    let vKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(1, 2));
    cv.morphologyEx(mask, mask, cv.MORPH_CLOSE, vKernel);
    matsToRelease.push(hKernel, hLines, vKernel);

    const rowVals = [];
    const colVals = [];
    const total = savedDoubleBoxes.value.rows.length * 2 + savedDoubleBoxes.value.cols.length * 2;
    let count = 0;

    const toRect = (box) => new cv.Rect(box.x, box.y, box.width, box.height);

    for (let pair of savedDoubleBoxes.value.rows) {
      statusText.value = `正在识别双色行... ${++count}/${total}`;
      let topRoi = mask.roi(toRect(pair.top));
      let val2 = await recognizeDigit(topRoi, worker);
      topRoi.delete();

      statusText.value = `正在识别双色行... ${++count}/${total}`;
      let bottomRoi = mask.roi(toRect(pair.bottom));
      let val1 = await recognizeDigit(bottomRoi, worker);
      bottomRoi.delete();

      rowVals.push({ c1: val1, c2: val2 });
    }

    for (let pair of savedDoubleBoxes.value.cols) {
      statusText.value = `正在识别双色列... ${++count}/${total}`;
      let leftRoi = mask.roi(toRect(pair.left));
      let val1 = await recognizeDigit(leftRoi, worker);
      leftRoi.delete();

      statusText.value = `正在识别双色列... ${++count}/${total}`;
      let rightRoi = mask.roi(toRect(pair.right));
      let val2 = await recognizeDigit(rightRoi, worker);
      rightRoi.delete();

      colVals.push({ c1: val1, c2: val2 });
    }

    matrixData.value = { rows: rowVals, cols: colVals };
    currentStep.value = 'result';
    statusText.value = "双色解密完成";
  } catch (e) {
    console.error(e);
    statusText.value = "识别发生错误";
  } finally {
    matsToRelease.forEach(m => { if (m && !m.isDeleted()) m.delete(); });
  }
};

const runAnalysis = async () => {
  if (!window.cv || !inputImage.value) return;

  const processImage = async () => {
    if (detectMode.value === 'single') {
      await detectMap(inputImage.value);
    } else {
      await detectDoubleMapDebug(inputImage.value);
    }
  };

  if (inputImage.value.complete) {
    await processImage();
  } else {
    inputImage.value.onload = processImage;
  }
};

const handleReUpload = () => {
  imageUrl.value = '';
  currentFile.value = null;
  currentStep.value = 'upload';
  statusText.value = '';
  if (fileInput.value) {
    fileInput.value.value = '';
  }
}
</script>

<template>
  <div class="upload-container">
    <input type="file" ref="fileInput" accept="image/*" @change="handleFileChange" style="display: none" />

    <div v-if="currentStep === 'upload'" class="upload-placeholder">
      <div class="tip">故障机器人修复工具</div>

      <div class="mode-selector" style="display: flex; gap: 20px; margin-bottom: 10px;">
        <label style="cursor: pointer; font-size: 1.1rem; color: #333;">
          <input type="radio" value="single" v-model="detectMode" /> 单色解密
        </label>
        <label style="cursor: pointer; font-size: 1.1rem; color: #333;">
          <input type="radio" value="double" v-model="detectMode" /> 双色解密
        </label>
      </div>

      <div class="display-block" @click.stop="triggerUpload">
        <span class="corner t-l"></span><span class="corner t-r"></span>
        <span class="corner b-l"></span><span class="corner b-r"></span>
      </div>
      <div class="tip-sub">点击上传图片</div>
    </div>

    <div v-else-if="currentStep === 'processing'" class="preview-wrapper">
      <img :src="imageUrl" ref="inputImage" style="display:none" />
      <div class="processing-content">
        <div class="tip-sub" style="font-size: 1.5rem; color: #409eff; margin-bottom: 20px;">
          {{ statusText || '准备分析...' }}
        </div>
        <div class="button-group">
          <button class="btn btn-secondary" @click="handleReUpload">取消操作</button>
        </div>
      </div>
    </div>

    <div v-else-if="currentStep === 'debug'" class="preview-wrapper">
      <img :src="imageUrl" ref="inputImage" style="display:none" />
      <div class="result-container">
        <div class="tip-sub" style="font-size: 1.2rem; color: #67c23a; margin-bottom: 15px; font-weight: bold;">
          {{ statusText }}
        </div>

        <canvas ref="debugCanvas" class="output-canvas"></canvas>

        <div class="button-group" style="margin-top: 20px;">
          <button class="btn btn-primary" @click="detectMode === 'single' ? executeSingleOCR() : executeDoubleOCR()">确认选区并识别</button>
          <button class="btn btn-secondary" @click="handleReUpload">重新上传</button>
        </div>
      </div>
    </div>

    <div v-else-if="currentStep === 'result'" class="preview-wrapper">
      <div class="result-container">
        <div class="tip-sub" style="font-size: 1.2rem; color: #67c23a; margin-bottom: 15px; font-weight: bold;">
          {{ statusText }}
        </div>

        <div class="matrix-board">
          <div class="matrix-row header-row">
            <div class="cell corner-cell"></div>
            <div v-for="(colVal, i) in matrixData.cols" :key="'col-'+i" class="cell header-cell" style="display:flex; justify-content:center; align-items:center;">
              <span v-if="detectMode === 'single'">{{ colVal }}</span>
              <div v-else style="display:flex; align-items:center; font-size: 1.1rem; gap: 4px;">
                <span style="color: #67c23a;">{{ colVal.c1 }}</span>
                <span style="color: #ccc; font-size: 0.9rem;">|</span>
                <span style="color: #409eff;">{{ colVal.c2 }}</span>
              </div>
            </div>
          </div>

          <div v-for="(rowVal, rIndex) in matrixData.rows" :key="'row-'+rIndex" class="matrix-row">
            <div class="cell header-cell side-header" style="display:flex; justify-content:center; align-items:center;">
              <span v-if="detectMode === 'single'">{{ rowVal }}</span>
              <div v-else style="display:flex; flex-direction:column; align-items:center; line-height:1.1; font-size: 1rem;">
                <span style="color: #409eff;">{{ rowVal.c2 }}</span>
                <span style="color: #67c23a;">{{ rowVal.c1 }}</span>
              </div>
            </div>
            <div v-for="(colVal, cIndex) in matrixData.cols" :key="'cell-'+rIndex+'-'+cIndex" class="cell grid-cell">
            </div>
          </div>
        </div>

        <div class="button-group" style="margin-top: 20px;">
          <button class="btn btn-primary" @click="handleReUpload">上传新图片</button>
        </div>
      </div>
    </div>

  </div>
</template>

<style>
@import "Root.css";
</style>
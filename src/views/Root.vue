<script setup>
import { ref, onUnmounted, nextTick } from 'vue'

const imageUrl = ref('');
const inputImage = ref(null);
const fileInput = ref(null);
const currentFile = ref(null);
const piecesPreviewUrl = ref('');

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
    statusText.value = `选区已提取：${rowNums.length}行${colNums.length}列`;

    let rightRoiW = src.cols - roiW;
    let piecesRoi = src.roi(new cv.Rect(roiW, 0, rightRoiW, src.rows));
    let piecesDebugMat = piecesRoi.clone();
    matsToRelease.push(piecesRoi, piecesDebugMat);

    let pGray = new cv.Mat();
    cv.cvtColor(piecesRoi, pGray, cv.COLOR_RGBA2GRAY);
    matsToRelease.push(pGray);

    let pMask = new cv.Mat();
    cv.threshold(pGray, pMask, 80, 255, cv.THRESH_BINARY);
    matsToRelease.push(pMask);

    let pContours = new cv.MatVector();
    let pHierarchy = new cv.Mat();
    cv.findContours(pMask, pContours, pHierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    matsToRelease.push(pContours, pHierarchy);

    let tempBoxes = [];
    for (let i = 0; i < pContours.size(); ++i) {
      let box = cv.boundingRect(pContours.get(i));
      if (box.width > 10 && box.height > 10) {
        tempBoxes.push(box);
      }
    }

    let validBoxes = [];
    if (tempBoxes.length > 0) {
      let reasonableBoxes = tempBoxes.filter(b => {
        let ratio = b.width / b.height;
        return ratio > 0.15 && ratio < 6.5;
      });

      if (reasonableBoxes.length > 0) {
        let maxDim = Math.max(...reasonableBoxes.map(b => Math.max(b.width, b.height)));

        for (let box of reasonableBoxes) {
          if (Math.max(box.width, box.height) > maxDim * 0.4) {
            validBoxes.push(box);
          }
        }
      }
    }

    for (let box of validBoxes) {
      cv.rectangle(piecesDebugMat, new cv.Point(box.x, box.y), new cv.Point(box.x + box.width, box.y + box.height), [0, 255, 0, 255], 2);

      let boxGray = pGray.roi(new cv.Rect(box.x, box.y, box.width, box.height));
      let boxEdges = new cv.Mat();
      cv.Canny(boxGray, boxEdges, 40, 120);

      let possibleGrids = [];

      for (let c = 1; c <= 5; c++) {
        for (let r = 1; r <= 5; r++) {
          let cellW = box.width / c;
          let cellH = box.height / r;
          let ratio = cellW / cellH;

          if (ratio > 0.6 && ratio < 1.6) {
            let isPure = true;
            let k = 0;

            for (let i = 0; i < c; i++) {
              for (let j = 0; j < r; j++) {
                let cx = Math.floor(i * cellW);
                let cy = Math.floor(j * cellH);
                let cw = Math.floor(cellW);
                let ch = Math.floor(cellH);

                let shrink = 4;
                let innerW = cw - shrink * 2;
                let innerH = ch - shrink * 2;

                if (innerW > 2 && innerH > 2) {
                  let absCx = box.x + cx;
                  let absCy = box.y + cy;
                  let cellRoi = pMask.roi(new cv.Rect(absCx + shrink, absCy + shrink, innerW, innerH));
                  let whiteCount = cv.countNonZero(cellRoi);
                  let fillRatio = whiteCount / (innerW * innerH);
                  cellRoi.delete();

                  if (fillRatio > 0.80) {
                    k++;
                  } else if (fillRatio < 0.20) {
                  } else {
                    isPure = false;
                    break;
                  }
                } else {
                  isPure = false; break;
                }
              }
              if (!isPure) break;
            }

            if (isPure && k > 0) {
              let edgeCount = 0;
              let totalLen = 0;

              for(let i = 1; i < c; i++){
                let px = Math.floor(i * cellW);
                for(let py = 0; py < box.height; py++){
                  if (boxEdges.ucharAt(py, px) > 0 ||
                      (px > 0 && boxEdges.ucharAt(py, px-1) > 0) ||
                      (px < box.width-1 && boxEdges.ucharAt(py, px+1) > 0)) {
                    edgeCount++;
                  }
                  totalLen++;
                }
              }
              for(let j = 1; j < r; j++){
                let py = Math.floor(j * cellH);
                for(let px = 0; px < box.width; px++){
                  if (boxEdges.ucharAt(py, px) > 0 ||
                      (py > 0 && boxEdges.ucharAt(py-1, px) > 0) ||
                      (py < box.height-1 && boxEdges.ucharAt(py+1, px) > 0)) {
                    edgeCount++;
                  }
                  totalLen++;
                }
              }

              let seamScore = totalLen > 0 ? (edgeCount / totalLen) : 0;
              possibleGrids.push({ c, r, k, total: c * r, seamScore });
            }
          }
        }
      }

      boxGray.delete();
      boxEdges.delete();

      let bestC = 1, bestR = 1;
      if (possibleGrids.length > 0) {
        possibleGrids.sort((a, b) => b.seamScore - a.seamScore);

        if (possibleGrids[0].seamScore > 0.15) {
          bestC = possibleGrids[0].c;
          bestR = possibleGrids[0].r;
        } else {
          possibleGrids.sort((a, b) => a.total - b.total);
          bestC = possibleGrids[0].c;
          bestR = possibleGrids[0].r;
        }
      } else {
        bestC = Math.max(1, Math.round(box.width / Math.min(box.width, box.height)));
        bestR = Math.max(1, Math.round(box.height / Math.min(box.width, box.height)));
      }

      let curUnitW = box.width / bestC;
      let curUnitH = box.height / bestR;

      for (let r = 0; r < bestR; r++) {
        for (let c = 0; c < bestC; c++) {
          let absCx = Math.floor(box.x + c * curUnitW + curUnitW / 2);
          let absCy = Math.floor(box.y + r * curUnitH + curUnitH / 2);

          let maskVal = pMask.ucharAt(absCy, absCx);
          if (maskVal > 128) {
            cv.circle(piecesDebugMat, new cv.Point(absCx, absCy), 5, [255, 0, 0, 255], -1);
          } else {
            cv.circle(piecesDebugMat, new cv.Point(absCx, absCy), 2, [150, 150, 150, 255], -1);
          }
        }
      }
    }

    let tempCanvas = document.createElement('canvas');
    cv.imshow(tempCanvas, piecesDebugMat);
    piecesPreviewUrl.value = tempCanvas.toDataURL('image/jpeg');

  } catch (e) {
    console.error(e);
    statusText.value = "提取失败";
  } finally {
    matsToRelease.forEach(m => { if (m && !m.isDeleted()) m.delete(); });
  }
};

const executeSingleOCR = async () => {
  currentStep.value = 'processing';
  statusText.value = "正在分析网格详情...";

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
    let roi = src.roi(new cv.Rect(0, 0, roiW, src.rows));
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

    let mask70 = new cv.Mat();
    let mask100 = new cv.Mat();
    cv.threshold(gray, mask70, 70, 255, cv.THRESH_BINARY);
    cv.threshold(gray, mask100, 100, 255, cv.THRESH_BINARY);
    matsToRelease.push(mask70, mask100);

    const rowValues = [];
    const colValues = [];

    for (const box of savedSingleBoxes.value.rows) {
      let digitRoi = gray.roi(box);
      rowValues.push({ val: await recognizeDigit(digitRoi, worker), y: box.y + box.height/2 });
      digitRoi.delete();
    }
    for (const box of savedSingleBoxes.value.cols) {
      let digitRoi = gray.roi(box);
      colValues.push({ val: await recognizeDigit(digitRoi, worker), x: box.x + box.width/2 });
      digitRoi.delete();
    }

    const finalGrid = [];
    for (let r = 0; r < rowValues.length; r++) {
      const rowData = [];
      for (let c = 0; c < colValues.length; c++) {
        const py = Math.floor(rowValues[r].y);
        const px = Math.floor(colValues[c].x);

        const val60 = mask70.ucharAt(py, px);
        const val100 = mask100.ucharAt(py, px);

        let type = 'normal';
        if (val100 > 0) {
          type = 'preset';
        } else if (val60 > 0) {
          type = 'obstacle';
        }
        rowData.push(type);
      }
      finalGrid.push(rowData);
    }

    matrixData.value = {
      rows: rowValues.map(v => v.val),
      cols: colValues.map(v => v.val),
      grid: finalGrid
    };

    currentStep.value = 'result';
    statusText.value = "单色地图深度识别完成";
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

    //组件提取与颜色分类
    let mask100 = new cv.Mat();
    cv.threshold(gray, mask100, 100, 255, cv.THRESH_BINARY);
    matsToRelease.push(mask100);

    const getMeanColor = (box) => {
      let boxRoi = roi.roi(new cv.Rect(box.x, box.y, box.width, box.height));
      let maskRoi = mask100.roi(new cv.Rect(box.x, box.y, box.width, box.height));
      let mean = cv.mean(boxRoi, maskRoi);
      boxRoi.delete();
      maskRoi.delete();
      return [mean[0], mean[1], mean[2]];
    };

    let color1 = finalColPairs.length > 0 ? getMeanColor(finalColPairs[0].left) : [0,255,0];
    let color2 = finalColPairs.length > 0 ? getMeanColor(finalColPairs[0].right) : [0,0,255];

    const colorDist = (c1, c2) => {
      const dist = Math.pow(c1[0] - c2[0], 2) + Math.pow(c1[1] - c2[1], 2) + Math.pow(c1[2] - c2[2], 2);
      const getMaxChannel = (c) => {
        if (c[0] >= c[1] && c[0] >= c[2]) return 0;
        if (c[1] >= c[0] && c[1] >= c[2]) return 1;
        return 2;
      };
      return getMaxChannel(c1) === getMaxChannel(c2) ? dist : dist + 1000000;
    };

    let rightRoiW = src.cols - roiW;
    let piecesRoi = src.roi(new cv.Rect(roiW, 0, rightRoiW, src.rows));
    let piecesDebugMat = piecesRoi.clone();
    matsToRelease.push(piecesRoi, piecesDebugMat);

    let pGray = new cv.Mat();
    cv.cvtColor(piecesRoi, pGray, cv.COLOR_RGBA2GRAY);
    matsToRelease.push(pGray);

    let pMask = new cv.Mat();
    cv.threshold(pGray, pMask, 80, 255, cv.THRESH_BINARY);
    matsToRelease.push(pMask);

    let pContours = new cv.MatVector();
    let pHierarchy = new cv.Mat();
    cv.findContours(pMask, pContours, pHierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    matsToRelease.push(pContours, pHierarchy);

    let tempBoxes = [];
    for (let i = 0; i < pContours.size(); ++i) {
      let box = cv.boundingRect(pContours.get(i));
      if (box.width > 10 && box.height > 10) {
        tempBoxes.push(box);
      }
    }

    let validBoxes = [];
    if (tempBoxes.length > 0) {
      let reasonableBoxes = tempBoxes.filter(b => {
        let ratio = b.width / b.height;
        return ratio > 0.15 && ratio < 6.5;
      });

      if (reasonableBoxes.length > 0) {
        let maxDim = Math.max(...reasonableBoxes.map(b => Math.max(b.width, b.height)));

        for (let box of reasonableBoxes) {
          if (Math.max(box.width, box.height) > maxDim * 0.4) {
            validBoxes.push(box);
          }
        }
      }
    }

    for (let box of validBoxes) {
      cv.rectangle(piecesDebugMat, new cv.Point(box.x, box.y), new cv.Point(box.x + box.width, box.y + box.height), [255, 255, 255, 255], 2);

      let boxGray = pGray.roi(new cv.Rect(box.x, box.y, box.width, box.height));
      let boxEdges = new cv.Mat();
      cv.Canny(boxGray, boxEdges, 40, 120);

      let possibleGrids = [];

      for (let c = 1; c <= 5; c++) {
        for (let r = 1; r <= 5; r++) {
          let cellW = box.width / c;
          let cellH = box.height / r;
          let ratio = cellW / cellH;

          if (ratio > 0.6 && ratio < 1.6) {
            let isPure = true;
            let k = 0;

            for (let i = 0; i < c; i++) {
              for (let j = 0; j < r; j++) {
                let cx = Math.floor(i * cellW);
                let cy = Math.floor(j * cellH);
                let cw = Math.floor(cellW);
                let ch = Math.floor(cellH);

                let shrink = 4;
                let innerW = cw - shrink * 2;
                let innerH = ch - shrink * 2;

                if (innerW > 2 && innerH > 2) {
                  let absCx = box.x + cx;
                  let absCy = box.y + cy;
                  let cellRoi = pMask.roi(new cv.Rect(absCx + shrink, absCy + shrink, innerW, innerH));
                  let whiteCount = cv.countNonZero(cellRoi);
                  let fillRatio = whiteCount / (innerW * innerH);
                  cellRoi.delete();

                  if (fillRatio > 0.80) {
                    k++;
                  } else if (fillRatio < 0.20) {
                  } else {
                    isPure = false;
                    break;
                  }
                } else {
                  isPure = false; break;
                }
              }
              if (!isPure) break;
            }

            if (isPure && k > 0) {
              let edgeCount = 0;
              let totalLen = 0;

              for(let i = 1; i < c; i++){
                let px = Math.floor(i * cellW);
                for(let py = 0; py < box.height; py++){
                  if (boxEdges.ucharAt(py, px) > 0 ||
                      (px > 0 && boxEdges.ucharAt(py, px-1) > 0) ||
                      (px < box.width-1 && boxEdges.ucharAt(py, px+1) > 0)) {
                    edgeCount++;
                  }
                  totalLen++;
                }
              }
              for(let j = 1; j < r; j++){
                let py = Math.floor(j * cellH);
                for(let px = 0; px < box.width; px++){
                  if (boxEdges.ucharAt(py, px) > 0 ||
                      (py > 0 && boxEdges.ucharAt(py-1, px) > 0) ||
                      (py < box.height-1 && boxEdges.ucharAt(py+1, px) > 0)) {
                    edgeCount++;
                  }
                  totalLen++;
                }
              }

              let seamScore = totalLen > 0 ? (edgeCount / totalLen) : 0;
              possibleGrids.push({ c, r, k, total: c * r, seamScore });
            }
          }
        }
      }

      boxGray.delete();
      boxEdges.delete();

      let bestC = 1, bestR = 1;
      if (possibleGrids.length > 0) {
        possibleGrids.sort((a, b) => b.seamScore - a.seamScore);
        if (possibleGrids[0].seamScore > 0.15) {
          bestC = possibleGrids[0].c;
          bestR = possibleGrids[0].r;
        } else {
          possibleGrids.sort((a, b) => a.total - b.total);
          bestC = possibleGrids[0].c;
          bestR = possibleGrids[0].r;
        }
      } else {
        bestC = Math.max(1, Math.round(box.width / Math.min(box.width, box.height)));
        bestR = Math.max(1, Math.round(box.height / Math.min(box.width, box.height)));
      }

      let curUnitW = box.width / bestC;
      let curUnitH = box.height / bestR;

      for (let r = 0; r < bestR; r++) {
        for (let c = 0; c < bestC; c++) {
          let absCx = Math.floor(box.x + c * curUnitW + curUnitW / 2);
          let absCy = Math.floor(box.y + r * curUnitH + curUnitH / 2);

          let maskVal = pMask.ucharAt(absCy, absCx);
          if (maskVal > 128) {
            let cellPixel = piecesRoi.ucharPtr(absCy, absCx);
            let cTarget = [cellPixel[0], cellPixel[1], cellPixel[2]];

            //比较并绘制不同颜色的点
            if (colorDist(cTarget, color1) < colorDist(cTarget, color2)) {
              //属于颜色1
              cv.circle(piecesDebugMat, new cv.Point(absCx, absCy), 5, [0, 255, 0, 255], -1);
            } else {
              //属于颜色2
              cv.circle(piecesDebugMat, new cv.Point(absCx, absCy), 5, [0, 0, 255, 255], -1);
            }
          } else {
            //缺角背景
            cv.circle(piecesDebugMat, new cv.Point(absCx, absCy), 2, [150, 150, 150, 255], -1);
          }
        }
      }
    }
    let tempCanvas = document.createElement('canvas');
    cv.imshow(tempCanvas, piecesDebugMat);
    piecesPreviewUrl.value = tempCanvas.toDataURL('image/jpeg');

    //Debug
    currentStep.value = 'debug';
    await nextTick();
    cv.imshow(debugCanvas.value, debugMat);
    statusText.value = `选区已提取：${finalRowPairs.length}行${finalColPairs.length}列`;

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
    let mask70 = new cv.Mat();
    let mask100 = new cv.Mat();
    cv.threshold(gray, mask, 96, 255, cv.THRESH_BINARY);
    cv.threshold(gray, mask70, 70, 255, cv.THRESH_BINARY);
    cv.threshold(gray, mask100, 100, 255, cv.THRESH_BINARY);
    matsToRelease.push(mask, mask70, mask100);

    let hKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(30, 1));
    let hLines = new cv.Mat();
    cv.morphologyEx(mask, hLines, cv.MORPH_OPEN, hKernel);
    cv.subtract(mask, hLines, mask);

    let vKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(1, 2));
    cv.morphologyEx(mask, mask, cv.MORPH_CLOSE, vKernel);
    matsToRelease.push(hKernel, hLines, vKernel);

    const getMeanColor = (box) => {
      let boxRoi = roi.roi(new cv.Rect(box.x, box.y, box.width, box.height));
      let maskRoi = mask100.roi(new cv.Rect(box.x, box.y, box.width, box.height));
      let mean = cv.mean(boxRoi, maskRoi);
      boxRoi.delete();
      maskRoi.delete();
      return [mean[0], mean[1], mean[2]];
    };

    let color1 = getMeanColor(savedDoubleBoxes.value.cols[0].left);
    let color2 = getMeanColor(savedDoubleBoxes.value.cols[0].right);

    const toHex = (c) => c.toString(16).padStart(2, '0');
    const rgbToHex = (color) => {
      const r = Math.trunc(color[0]);
      const g = Math.trunc(color[1]);
      const b = Math.trunc(color[2]);
      return `#${toHex(r)}${toHex(g)}${toHex(b)}`.toUpperCase();
    };
    console.log(rgbToHex(color1), rgbToHex(color2));

    const colorDist = (c1, c2) => {
      //计算常规的 RGB 欧氏距离
      const dist = Math.pow(c1[0] - c2[0], 2) + Math.pow(c1[1] - c2[1], 2) + Math.pow(c1[2] - c2[2], 2);
      //找到颜色的主通道
      const getMaxChannel = (c) => {
        if (c[0] >= c[1] && c[0] >= c[2]) return 0;
        if (c[1] >= c[0] && c[1] >= c[2]) return 1;
        return 2;
      };
      //如果主通道不同，直接判定为不匹配
      return getMaxChannel(c1) === getMaxChannel(c2) ? dist : dist + 1000000;
    };

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

    const finalGrid = [];
    for (let r = 0; r < savedDoubleBoxes.value.rows.length; r++) {
      const rowData = [];
      for (let c = 0; c < savedDoubleBoxes.value.cols.length; c++) {
        const py = Math.floor(savedDoubleBoxes.value.rows[r].centerY);
        const px = Math.floor(savedDoubleBoxes.value.cols[c].centerX);

        const val60 = mask70.ucharAt(py, px);
        const val100 = mask100.ucharAt(py, px);

        let type = 'normal';
        if (val100 > 0) {
          let cellPixel = roi.ucharPtr(py, px);
          let cTarget = [cellPixel[0], cellPixel[1], cellPixel[2]];
          console.log(`(${r},${c}):`, rgbToHex(cTarget));
          console.log(colorDist(cTarget, color1), colorDist(cTarget, color2))
          if (colorDist(cTarget, color1) < colorDist(cTarget, color2)) {
            type = 'preset-c1';
          } else {
            type = 'preset-c2';
          }
        } else if (val60 > 0) {
          type = 'obstacle';
        }
        rowData.push(type);
      }
      finalGrid.push(rowData);
    }

    matrixData.value = { rows: rowVals, cols: colVals, grid: finalGrid };
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
  piecesPreviewUrl.value = '';
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

        <div style="display: flex; gap: 40px; align-items: flex-start; justify-content: center; width: 100%;">
          <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
            <span style="font-weight: bold; color: #409eff;">左侧网格选区</span>
            <canvas ref="debugCanvas" class="output-canvas" style="max-height: 65vh;"></canvas>
          </div>

          <div v-if="piecesPreviewUrl" style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
            <span style="font-weight: bold; color: #409eff;">右侧组件识别预览</span>
            <img :src="piecesPreviewUrl" style="max-height: 65vh; border: 2px solid #409eff; border-radius: 8px;" />
          </div>
        </div>

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
          <div v-for="(rowType, rIndex) in matrixData.grid" :key="'row-grid-'+rIndex" class="matrix-row">
            <div class="cell header-cell side-header" style="display:flex; justify-content:center; align-items:center;">
              <span v-if="detectMode === 'single'">{{ matrixData.rows[rIndex] }}</span>
              <div v-else style="display:flex; flex-direction:column; align-items:center; line-height:1.1; font-size: 1rem;">
                <span style="color: #409eff;">{{ matrixData.rows[rIndex].c2 }}</span>
                <span style="color: #67c23a;">{{ matrixData.rows[rIndex].c1 }}</span>
              </div>
            </div>
            <div v-for="(cellType, cIndex) in rowType"
                 :key="'cell-'+rIndex+'-'+cIndex"
                 :class="['cell', 'grid-cell', cellType + '-cell']">
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
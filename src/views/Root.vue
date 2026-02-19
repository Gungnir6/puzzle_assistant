<script setup>
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import LockIcon from '@/assets/lock.svg';

const imageUrl = ref('');
const inputImage = ref(null);
const fileInput = ref(null);
const currentFile = ref(null);
const piecesPreviewUrl = ref('');

const detectMode = ref('single');
const currentStep = ref('upload');
const matrixData = ref({ rows: [], cols: [] });
const debugCanvas = ref(null);
const isDragging = ref(false);

const savedSingleBoxes = ref({ rows: [], cols: [] });
const savedDoubleBoxes = ref({ rows: [], cols: [] });
const recognizedPieces = ref([]);

let worker = null;

const calculateDynamicBounds = (src) => {
  let hsv = new cv.Mat();
  cv.cvtColor(src, hsv, cv.COLOR_RGBA2RGB);
  cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);
  let channels = new cv.MatVector();
  cv.split(hsv, channels);
  let sat = channels.get(1);
  let val = channels.get(2);

  let mask = new cv.Mat();
  let satMask = new cv.Mat();
  let valMask = new cv.Mat();
  cv.threshold(sat, satMask, 40, 255, cv.THRESH_BINARY);
  cv.threshold(val, valMask, 80, 255, cv.THRESH_BINARY);
  cv.bitwise_and(satMask, valMask, mask);

  let contours = new cv.MatVector();
  let hierarchy = new cv.Mat();
  cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

  let boxes = [];
  for (let i = 0; i < contours.size(); ++i) {
    let box = cv.boundingRect(contours.get(i));
    if (box.width > 8 && box.height > 8 && box.width < src.cols * 0.3) {
      boxes.push(box);
    }
  }

  satMask.delete(); valMask.delete(); mask.delete();
  contours.delete(); hierarchy.delete(); channels.delete(); hsv.delete(); sat.delete(); val.delete();

  if (boxes.length < 5) return { roiW: Math.floor(src.cols * 0.7), gridMinX: 0 };

  boxes.sort((a, b) => a.x - b.x);
  let maxGap = 0;
  let splitX = Math.floor(src.cols * 0.7);

  for (let i = 1; i < boxes.length; i++) {
    let gap = boxes[i].x - (boxes[i - 1].x + boxes[i - 1].width);
    if (gap > maxGap && gap > src.cols * 0.05) {
      maxGap = gap;
      splitX = boxes[i - 1].x + boxes[i - 1].width + Math.floor(gap / 2);
    }
  }

  let leftBoxes = boxes.filter(b => b.x + b.width < splitX);
  let gridMinX = leftBoxes.length > 0 ? Math.min(...leftBoxes.map(b => b.x)) : 0;

  return { roiW: splitX, gridMinX: gridMinX };
};

const triggerUpload = () => {
  if (fileInput.value) fileInput.value.click()
}

const processFile = (file) => {
  if (file && file.type.startsWith('image/')) {
    currentFile.value = file;
    if (imageUrl.value) URL.revokeObjectURL(imageUrl.value);
    imageUrl.value = URL.createObjectURL(file);

    currentStep.value = 'processing';
    matrixData.value = { rows: [], cols: [] };

    setTimeout(() => {
      runAnalysis();
    }, 50);
  }
};

//点击上传
const handleFileChange = (event) => {
  const file = event.target.files[0];
  if (file) processFile(file);
};

//拖拽上传
const handleDrop = (event) => {
  isDragging.value = false;
  const file = event.dataTransfer.files[0];
  if (file) processFile(file);
};

//Ctrl+V上传
const handlePaste = (event) => {
  //只有在初始上传界面才响应粘贴事件
  if (currentStep.value !== 'upload') return;

  const items = event.clipboardData.items;
  for (let i = 0; i < items.length; i++) {
    if (items[i].type.indexOf('image') !== -1) {
      const file = items[i].getAsFile();
      processFile(file);
      break; //只取第一张图片
    }
  }
};

onMounted(() => {
  window.addEventListener('paste', handlePaste);
});

onUnmounted(async () => {
  window.removeEventListener('paste', handlePaste);
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
    recognizedPieces.value = [];
    const { roiW, gridMinX } = calculateDynamicBounds(src);
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
    let gray = channels.get(2).clone();
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
      if (box.width > 6 && box.height > 6 && box.width < 100 && box.x >= gridMinX - 10) {
        boxes.push(box);
      }
    }

    if (boxes.length < 5) throw new Error("未检测到足够数据");

    let yGroups = [];
    let xGroups = [];
    for (let b of boxes) {
      let yg = yGroups.find(g => Math.abs(g.y - b.y) < 15);
      if (yg) { yg.boxes.push(b); yg.y = (yg.y * (yg.boxes.length - 1) + b.y) / yg.boxes.length; }
      else { yGroups.push({y: b.y, boxes: [b]}); }

      let xg = xGroups.find(g => Math.abs(g.x - b.x) < 15);
      if (xg) { xg.boxes.push(b); xg.x = (xg.x * (xg.boxes.length - 1) + b.x) / xg.boxes.length; }
      else { xGroups.push({x: b.x, boxes: [b]}); }
    }

    yGroups.sort((a, b) => a.y - b.y);
    let topYGroup = yGroups.find(g => g.boxes.length >= 3 && (Math.max(...g.boxes.map(b=>b.x)) - Math.min(...g.boxes.map(b=>b.x)) > 80));
    let colNums = topYGroup ? topYGroup.boxes : [];
    colNums.sort((a, b) => a.x - b.x);

    xGroups.sort((a, b) => a.x - b.x);
    let leftXGroup = xGroups.find(g => g.boxes.length >= 3 && (Math.max(...g.boxes.map(b=>b.y)) - Math.min(...g.boxes.map(b=>b.y)) > 80));
    let rowNums = leftXGroup ? leftXGroup.boxes : [];
    rowNums.sort((a, b) => a.y - b.y);

    if (colNums.length > 2) {
      let lastGap = colNums[colNums.length - 1].x - colNums[colNums.length - 2].x;
      let avgGap = (colNums[colNums.length - 2].x - colNums[0].x) / (colNums.length - 2);
      if (lastGap > avgGap * 2.5) colNums.pop();
    }

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

    let rightRoiW = src.cols - roiW;
    let piecesRoi = src.roi(new cv.Rect(roiW, 0, rightRoiW, src.rows));
    let piecesDebugMat = piecesRoi.clone();
    matsToRelease.push(piecesRoi, piecesDebugMat);

    let pGray = new cv.Mat();
    cv.cvtColor(piecesRoi, pGray, cv.COLOR_RGBA2GRAY);
    matsToRelease.push(pGray);

    let pLightMask = new cv.Mat();
    cv.threshold(pGray, pLightMask, 80, 255, cv.THRESH_BINARY);
    matsToRelease.push(pLightMask);

    let pHsv = new cv.Mat();
    cv.cvtColor(piecesRoi, pHsv, cv.COLOR_RGBA2RGB);
    cv.cvtColor(pHsv, pHsv, cv.COLOR_RGB2HSV);
    matsToRelease.push(pHsv);

    let pChannels = new cv.MatVector();
    cv.split(pHsv, pChannels);
    let pSat = pChannels.get(1);
    pChannels.delete();
    matsToRelease.push(pSat);

    let pSatMask = new cv.Mat();
    cv.threshold(pSat, pSatMask, 40, 255, cv.THRESH_BINARY);
    matsToRelease.push(pSatMask);

    let pMask = new cv.Mat();
    cv.bitwise_and(pLightMask, pSatMask, pMask);
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

    validBoxes.sort((a, b) => {
      if (Math.abs(a.y - b.y) < 20) {
        return a.x - b.x;
      }
      return a.y - b.y;
    });

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

      let pieceGrid = [];
      for (let r = 0; r < bestR; r++) {
        let rowData = [];
        for (let c = 0; c < bestC; c++) {
          let absCx = Math.floor(box.x + c * curUnitW + curUnitW / 2);
          let absCy = Math.floor(box.y + r * curUnitH + curUnitH / 2);

          let maskVal = pMask.ucharAt(absCy, absCx);
          if (maskVal > 128) {
            cv.circle(piecesDebugMat, new cv.Point(absCx, absCy), 5, [255, 0, 0, 255], -1);
            rowData.push(1);
          } else {
            cv.circle(piecesDebugMat, new cv.Point(absCx, absCy), 2, [150, 150, 150, 255], -1);
            rowData.push(0);
          }
        }
        pieceGrid.push(rowData);
      }
      recognizedPieces.value.push({ grid: pieceGrid, type: 'single' });
    }

    let tempCanvas = document.createElement('canvas');
    cv.imshow(tempCanvas, piecesDebugMat);
    piecesPreviewUrl.value = tempCanvas.toDataURL('image/jpeg');

  } catch (e) {
    console.error(e);
  } finally {
    matsToRelease.forEach(m => { if (m && !m.isDeleted()) m.delete(); });
  }
};

const executeSingleOCR = async () => {
  currentStep.value = 'processing';

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

    const { roiW } = calculateDynamicBounds(src);
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
  } catch (e) {
    console.error(e);
  } finally {
    matsToRelease.forEach(m => { if (m && !m.isDeleted()) m.delete(); });
  }
};

const detectDoubleMapDebug = async (imgElement) => {
  let src = cv.imread(imgElement);
  let matsToRelease = [src];


  try {
    recognizedPieces.value = [];
    const { roiW, gridMinX } = calculateDynamicBounds(src);
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
    let sat = channels.get(1);
    let gray = channels.get(2);
    channels.delete();
    matsToRelease.push(sat, gray);

    let mask = new cv.Mat();
    let sMask = new cv.Mat();

    cv.threshold(gray, mask, 80, 255, cv.THRESH_BINARY);
    cv.threshold(sat, sMask, 40, 255, cv.THRESH_BINARY);

    cv.bitwise_and(mask, sMask, mask);

    sMask.delete();
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
      if (box.width > 6 && box.height > 6 && box.width < 100 && box.x >= gridMinX - 10) {
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

    let pLightMask = new cv.Mat();
    cv.threshold(pGray, pLightMask, 80, 255, cv.THRESH_BINARY);
    matsToRelease.push(pLightMask);

    let pHsv = new cv.Mat();
    cv.cvtColor(piecesRoi, pHsv, cv.COLOR_RGBA2RGB);
    cv.cvtColor(pHsv, pHsv, cv.COLOR_RGB2HSV);
    matsToRelease.push(pHsv);

    let pChannels = new cv.MatVector();
    cv.split(pHsv, pChannels);
    let pSat = pChannels.get(1);
    pChannels.delete();
    matsToRelease.push(pSat);

    let pSatMask = new cv.Mat();
    cv.threshold(pSat, pSatMask, 40, 255, cv.THRESH_BINARY);
    matsToRelease.push(pSatMask);

    let pMask = new cv.Mat();
    cv.bitwise_and(pLightMask, pSatMask, pMask);
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

    validBoxes.sort((a, b) => {
      if (Math.abs(a.y - b.y) < 20) {
        return a.x - b.x;
      }
      return a.y - b.y;
    });

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

      let pieceGrid = [];
      let c1Votes = 0;
      let c2Votes = 0;

      for (let r = 0; r < bestR; r++) {
        let rowData = [];
        for (let c = 0; c < bestC; c++) {
          let absCx = Math.floor(box.x + c * curUnitW + curUnitW / 2);
          let absCy = Math.floor(box.y + r * curUnitH + curUnitH / 2);

          let maskVal = pMask.ucharAt(absCy, absCx);
          if (maskVal > 128) {
            let cellPixel = piecesRoi.ucharPtr(absCy, absCx);
            let cTarget = [cellPixel[0], cellPixel[1], cellPixel[2]];

            if (colorDist(cTarget, color1) < colorDist(cTarget, color2)) {
              cv.circle(piecesDebugMat, new cv.Point(absCx, absCy), 5, [0, 255, 0, 255], -1);
              c1Votes++;
            } else {
              cv.circle(piecesDebugMat, new cv.Point(absCx, absCy), 5, [0, 0, 255, 255], -1);
              c2Votes++;
            }
            rowData.push(1);
          } else {
            cv.circle(piecesDebugMat, new cv.Point(absCx, absCy), 2, [150, 150, 150, 255], -1);
            rowData.push(0);
          }
        }
        pieceGrid.push(rowData);
      }
      let pieceType = c1Votes >= c2Votes ? 'c1' : 'c2';
      recognizedPieces.value.push({ grid: pieceGrid, type: pieceType });
    }
    let tempCanvas = document.createElement('canvas');
    cv.imshow(tempCanvas, piecesDebugMat);
    piecesPreviewUrl.value = tempCanvas.toDataURL('image/jpeg');

    //Debug
    currentStep.value = 'debug';
    await nextTick();
    cv.imshow(debugCanvas.value, debugMat);

  } catch (e) {
    console.error(e);
  } finally {
    matsToRelease.forEach(m => { if (m && !m.isDeleted()) m.delete(); });
  }
};

const executeDoubleOCR = async () => {
  currentStep.value = 'processing';

  let src = cv.imread(inputImage.value);
  let matsToRelease = [src];

  try {
    if (!worker) {
      worker = await Tesseract.createWorker('eng', 1, { logger: () => {} });
      await worker.setParameters({ tessedit_char_whitelist: '0123456789Ø', tessedit_pageseg_mode: '10' });
    }

    const { roiW } = calculateDynamicBounds(src);
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
      let topRoi = mask.roi(toRect(pair.top));
      let val2 = await recognizeDigit(topRoi, worker);
      topRoi.delete();

      let bottomRoi = mask.roi(toRect(pair.bottom));
      let val1 = await recognizeDigit(bottomRoi, worker);
      bottomRoi.delete();

      rowVals.push({ c1: val1, c2: val2 });
    }

    for (let pair of savedDoubleBoxes.value.cols) {
      let leftRoi = mask.roi(toRect(pair.left));
      let val1 = await recognizeDigit(leftRoi, worker);
      leftRoi.delete();

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
  } catch (e) {
    console.error(e);
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

const getSolutionCellClass = (cellData) => {
  if (!cellData) return 'normal-cell';
  if (cellData.type === 'obstacle') return 'obstacle-cell';
  if (cellData.type === 'preset' || cellData.type === 'preset-c1') return 'preset-c1-cell';
  if (cellData.type === 'preset-c2') return 'preset-c2-cell';
  if (cellData.type === 'piece') {
    if (detectMode.value === 'single') return 'cell-piece-single';
    if (cellData.colorType === 'c1') return 'cell-piece-c1';
    if (cellData.colorType === 'c2') return 'cell-piece-c2';
  }
  return 'normal-cell';
};

const checkConnection = (r, c, direction) => {
  const solution = matrixData.value.solution;
  const current = solution[r][c];
  if (!current || current.type !== 'piece') return false;

  let nr = r, nc = c;
  if (direction === 'top') nr--;
  else if (direction === 'bottom') nr++;
  else if (direction === 'left') nc--;
  else if (direction === 'right') nc++;
  else if (direction === 'top-left') { nr--; nc--; }
  else if (direction === 'top-right') { nr--; nc++; }
  else if (direction === 'bottom-left') { nr++; nc--; }
  else if (direction === 'bottom-right') { nr++; nc++; }

  if (nr < 0 || nr >= solution.length || nc < 0 || nc >= solution[0].length) return false;
  const neighbor = solution[nr][nc];

  return neighbor && neighbor.type === 'piece' && neighbor.id === current.id;
};

const handleWheel = (event, type, index, colorType = null) => {
  const maxVal = type === 'row' ? matrixData.value.cols.length : matrixData.value.rows.length;
  let dataRef = type === 'row' ? matrixData.value.rows : matrixData.value.cols;

  let currentVal = detectMode.value === 'single' ? dataRef[index] : dataRef[index][colorType];

  if (event.deltaY < 0 && currentVal < maxVal) {
    currentVal++;
  } else if (event.deltaY > 0 && currentVal > 0) {
    currentVal--;
  } else {
    return;
  }

  if (detectMode.value === 'single') {
    dataRef[index] = currentVal;
  } else {
    dataRef[index][colorType] = currentVal;
  }
};

const dragState = {
  isDragging: false,
  hasMoved: false,
  startY: 0,
  startVal: 0,
  type: null,
  index: -1,
  colorType: null,
  maxVal: 0
};

const startNumberDrag = (e, type, index, colorType = null) => {
  dragState.isDragging = true;
  dragState.hasMoved = false;
  dragState.startY = e.type.includes('mouse') ? e.clientY : e.touches[0].clientY;

  dragState.maxVal = type === 'row' ? matrixData.value.cols.length : matrixData.value.rows.length;
  dragState.type = type;
  dragState.index = index;
  dragState.colorType = colorType;

  let dataRef = type === 'row' ? matrixData.value.rows : matrixData.value.cols;
  dragState.startVal = detectMode.value === 'single' ? dataRef[index] : dataRef[index][colorType];

  if (e.type.includes('mouse')) {
    document.addEventListener('mousemove', onNumberDrag);
    document.addEventListener('mouseup', stopNumberDrag);
  }
};

const onNumberDrag = (e) => {
  if (!dragState.isDragging) return;
  const currentY = e.type.includes('mouse') ? e.clientY : e.touches[0].clientY;
  const deltaY = currentY - dragState.startY;

  if (Math.abs(deltaY) > 5) {
    dragState.hasMoved = true;
  }

  if (dragState.hasMoved) {
    const sensitivity = 20;
    let diff = Math.round(-deltaY / sensitivity);
    let newVal = dragState.startVal + diff;

    newVal = Math.max(0, Math.min(dragState.maxVal, newVal));

    let dataRef = dragState.type === 'row' ? matrixData.value.rows : matrixData.value.cols;
    if (detectMode.value === 'single') {
      dataRef[dragState.index] = newVal;
    } else {
      dataRef[dragState.index][dragState.colorType] = newVal;
    }
  }
};

const stopNumberDrag = () => {
  if (!dragState.isDragging) return;
  dragState.isDragging = false;

  if (!dragState.hasMoved) {
    let newVal = dragState.startVal + 1;
    if (newVal > dragState.maxVal) newVal = 0;

    let dataRef = dragState.type === 'row' ? matrixData.value.rows : matrixData.value.cols;
    if (detectMode.value === 'single') {
      dataRef[dragState.index] = newVal;
    } else {
      dataRef[dragState.index][dragState.colorType] = newVal;
    }
  }

  document.removeEventListener('mousemove', onNumberDrag);
  document.removeEventListener('mouseup', stopNumberDrag);
};

const solvePuzzle = async () => {
  currentStep.value = 'processing';
  await nextTick();

  setTimeout(() => {
    const R = matrixData.value.rows.length;
    const C = matrixData.value.cols.length;

    // 1. 约束目标初始化
    let targetRow = [], targetCol = [];
    if (detectMode.value === 'single') {
      targetRow = matrixData.value.rows.map(v => ({c1: v, c2: 0}));
      targetCol = matrixData.value.cols.map(v => ({c1: v, c2: 0}));
    } else {
      targetRow = matrixData.value.rows.map(v => ({c1: v.c1, c2: v.c2}));
      targetCol = matrixData.value.cols.map(v => ({c1: v.c1, c2: v.c2}));
    }

    let curRow = Array.from({length: R}, () => ({c1: 0, c2: 0}));
    let curCol = Array.from({length: C}, () => ({c1: 0, c2: 0}));
    let board = Array.from({length: R}, () => Array(C).fill(null));

    // 2. 障碍物与预设方块入列，预占容量
    for (let r = 0; r < R; r++) {
      for (let c = 0; c < C; c++) {
        let type = matrixData.value.grid[r][c];
        if (type === 'obstacle') {
          board[r][c] = { type: 'obstacle' };
        } else if (type === 'preset' || type === 'preset-c1') {
          board[r][c] = { type: 'preset-c1' };
          curRow[r].c1++; curCol[c].c1++;
        } else if (type === 'preset-c2') {
          board[r][c] = { type: 'preset-c2' };
          curRow[r].c2++; curCol[c].c2++;
        }
      }
    }

    const getRotations = (grid) => {
      const rots = [];
      let current = grid;
      for(let i = 0; i < 4; i++) {
        if (!rots.some(r => JSON.stringify(r) === JSON.stringify(current))) {
          rots.push(current);
        }
        const rows = current.length;
        const cols = current[0].length;
        let next = Array.from({length: cols}, () => Array(rows).fill(0));
        for(let r = 0; r < rows; r++) {
          for(let c = 0; c < cols; c++) {
            next[c][rows - 1 - r] = current[r][c];
          }
        }
        current = next;
      }
      return rots;
    };

    // 3. 提取组件并应用启发式排序（面积大的优先放置，提早触发剪枝）
    let pieces = recognizedPieces.value.map((p, idx) => {
      let type = (detectMode.value === 'single') ? 'c1' : p.type;
      let shapes = getRotations(p.grid);
      let size = p.grid.flat().reduce((a, b) => a + b, 0);
      return { id: idx, type, shapes, size };
    });
    pieces.sort((a, b) => b.size - a.size);

    // 4. DFS 核心
    const dfs = (pIdx) => {
      if (pIdx === pieces.length) {
        // 验证最终行列数量是否严丝合缝
        for(let r=0; r<R; r++) if(curRow[r].c1 !== targetRow[r].c1 || curRow[r].c2 !== targetRow[r].c2) return false;
        for(let c=0; c<C; c++) if(curCol[c].c1 !== targetCol[c].c1 || curCol[c].c2 !== targetCol[c].c2) return false;
        return true;
      }

      let p = pieces[pIdx];
      let pType = p.type;

      for (let shape of p.shapes) {
        let sr = shape.length;
        let sc = shape[0].length;

        for (let r = 0; r <= R - sr; r++) {
          for (let c = 0; c <= C - sc; c++) {
            let canPlace = true;
            let rowAdd = new Array(sr).fill(0);
            let colAdd = new Array(sc).fill(0);

            // 预判一：碰撞检测
            for (let ir = 0; ir < sr; ir++) {
              for (let ic = 0; ic < sc; ic++) {
                if (shape[ir][ic]) {
                  if (board[r+ir][c+ic] !== null) { canPlace = false; break; }
                  rowAdd[ir]++;
                  colAdd[ic]++;
                }
              }
              if (!canPlace) break;
            }
            if (!canPlace) continue;

            // 预判二：行列容量极速剪枝
            for (let ir = 0; ir < sr; ir++) {
              if (pType === 'c1' && curRow[r+ir].c1 + rowAdd[ir] > targetRow[r+ir].c1) { canPlace = false; break; }
              if (pType === 'c2' && curRow[r+ir].c2 + rowAdd[ir] > targetRow[r+ir].c2) { canPlace = false; break; }
            }
            if(!canPlace) continue;

            for (let ic = 0; ic < sc; ic++) {
              if (pType === 'c1' && curCol[c+ic].c1 + colAdd[ic] > targetCol[c+ic].c1) { canPlace = false; break; }
              if (pType === 'c2' && curCol[c+ic].c2 + colAdd[ic] > targetCol[c+ic].c2) { canPlace = false; break; }
            }
            if (!canPlace) continue;

            // 回溯：放置组件
            for (let ir = 0; ir < sr; ir++) {
              for (let ic = 0; ic < sc; ic++) {
                if (shape[ir][ic]) {
                  board[r+ir][c+ic] = { type: 'piece', id: p.id, colorType: pType };
                  if(pType === 'c1') { curRow[r+ir].c1++; curCol[c+ic].c1++; }
                  else { curRow[r+ir].c2++; curCol[c+ic].c2++; }
                }
              }
            }

            if (dfs(pIdx + 1)) return true;

            // 回溯：撤销放置
            for (let ir = 0; ir < sr; ir++) {
              for (let ic = 0; ic < sc; ic++) {
                if (shape[ir][ic]) {
                  board[r+ir][c+ic] = null;
                  if(pType === 'c1') { curRow[r+ir].c1--; curCol[c+ic].c1--; }
                  else { curRow[r+ir].c2--; curCol[c+ic].c2--; }
                }
              }
            }
          }
        }
      }
      return false;
    };

    let success = dfs(0);
    if (success) {
      matrixData.value.solution = board;
      currentStep.value = 'solved';
    } else {
      currentStep.value = 'result';
      alert("未找到可行解，可能组件或网格识别存在误差");
    }
  }, 100);
};

const handleReUpload = () => {
  imageUrl.value = '';
  currentFile.value = null;
  piecesPreviewUrl.value = '';
  currentStep.value = 'upload';
  if (fileInput.value) {
    fileInput.value.value = '';
  }
}
</script>

<template>
  <div class="upload-container">
    <input type="file" ref="fileInput" accept="image/*" @change="handleFileChange" style="display: none" />

    <div v-if="currentStep === 'upload'"
         class="upload-placeholder"
         :class="{ 'is-dragging': isDragging }"
         @dragover.prevent="isDragging = true"
         @dragenter.prevent="isDragging = true"
         @dragleave.prevent="isDragging = false"
         @drop.prevent="handleDrop">

      <div class="tip">故障机器人修复工具</div>

      <div class="mode-toggle-wrapper">
        <span class="mode-label" :class="{ 'active': detectMode === 'single' }" @click="detectMode = 'single'">单色解密</span>

        <div class="toggle-switch" :class="detectMode" @click="detectMode = detectMode === 'single' ? 'double' : 'single'">
          <div class="toggle-circle"></div>
        </div>

        <span class="mode-label" :class="{ 'active': detectMode === 'double' }" @click="detectMode = 'double'">双色解密</span>
      </div>

      <div class="display-block" @click.stop="triggerUpload">
        <span class="corner t-l"></span><span class="corner t-r"></span>
        <span class="corner b-l"></span><span class="corner b-r"></span>
      </div>
      <div class="tip-sub">点击、拖拽或按 Ctrl+V 粘贴图片</div>
    </div>

    <div v-else-if="currentStep === 'processing'" class="preview-wrapper">
      <img :src="imageUrl" ref="inputImage" style="display:none" alt=""/>
      <div class="processing-content">
        <div class="tip-sub" style="font-size: 1.5rem; color: #409eff; margin-bottom: 20px;">
          准备分析...
        </div>
        <div class="button-group">
          <button class="btn btn-secondary" @click="handleReUpload">取消操作</button>
        </div>
      </div>
    </div>

    <div v-else-if="currentStep === 'debug'" class="preview-wrapper">
      <img :src="imageUrl" ref="inputImage" style="display:none" alt=""/>
      <div class="result-container">
        <div class="tip-sub" style="font-size: 1.2rem; color: #67c23a; margin-bottom: 15px; font-weight: bold;">
          提取选区
        </div>

        <div style="display: flex; gap: 40px; align-items: flex-start; justify-content: center; width: 100%;">
          <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
            <canvas ref="debugCanvas" class="output-canvas" style="max-height: 65vh;"></canvas>
          </div>

          <div v-if="piecesPreviewUrl" style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
            <img :src="piecesPreviewUrl" style="max-height: 65vh; border: 2px solid #409eff; border-radius: 8px;" alt=""/>
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
          识别结果
        </div>

        <div style="display: flex; gap: 40px; align-items: flex-start; justify-content: center; width: 100%;">
          <div class="matrix-board">
            <div class="matrix-row header-row">
              <div class="cell corner-cell"></div>
              <div v-for="(colVal, i) in matrixData.cols" :key="'col-sol-'+i" class="cell header-cell" style="display:flex; justify-content:center; align-items:center;">
                <span v-if="detectMode === 'single'" class="adjustable-number"
                      @wheel.prevent="handleWheel($event, 'col', i)"
                      @touchstart="startNumberDrag($event, 'col', i)"
                      @touchmove.prevent="onNumberDrag"
                      @touchend="stopNumberDrag"
                      @mousedown="startNumberDrag($event, 'col', i)">
                  {{ matrixData.cols[i] }}
                </span>
                <div v-else style="display:flex; align-items:center; font-size: 1.1rem; gap: 4px;">
                  <span style="color: #67c23a;" class="adjustable-number"
                        @wheel.prevent="handleWheel($event, 'col', i, 'c1')"
                        @touchstart="startNumberDrag($event, 'col', i, 'c1')"
                        @touchmove.prevent="onNumberDrag"
                        @touchend="stopNumberDrag"
                        @mousedown="startNumberDrag($event, 'col', i, 'c1')">{{ matrixData.cols[i].c1 }}</span>
                  <span style="color: #ccc; font-size: 0.9rem;">|</span>
                  <span style="color: #409eff;" class="adjustable-number"
                        @wheel.prevent="handleWheel($event, 'col', i, 'c2')"
                        @touchstart="startNumberDrag($event, 'col', i, 'c2')"
                        @touchmove.prevent="onNumberDrag"
                        @touchend="stopNumberDrag"
                        @mousedown="startNumberDrag($event, 'col', i, 'c2')">{{ matrixData.cols[i].c2 }}</span>
                </div>
              </div>
            </div>
            <div v-for="(rowType, rIndex) in matrixData.grid" :key="'row-grid-'+rIndex" class="matrix-row">
              <div class="cell header-cell side-header" style="display:flex; justify-content:center; align-items:center;">
                <span v-if="detectMode === 'single'" class="adjustable-number"
                      @wheel.prevent="handleWheel($event, 'row', rIndex)"
                      @touchstart="startNumberDrag($event, 'row', rIndex)"
                      @touchmove.prevent="onNumberDrag"
                      @touchend="stopNumberDrag"
                      @mousedown="startNumberDrag($event, 'row', rIndex)">
                  {{ matrixData.rows[rIndex] }}
                </span>
                <div v-else style="display:flex; flex-direction:column; align-items:center; line-height:1.1; font-size: 1rem;">
                  <span style="color: #409eff;" class="adjustable-number"
                        @wheel.prevent="handleWheel($event, 'row', rIndex, 'c2')"
                        @touchstart="startNumberDrag($event, 'row', rIndex, 'c2')"
                        @touchmove.prevent="onNumberDrag"
                        @touchend="stopNumberDrag"
                        @mousedown="startNumberDrag($event, 'row', rIndex, 'c2')">{{ matrixData.rows[rIndex].c2 }}</span>
                  <span style="color: #67c23a;" class="adjustable-number"
                        @wheel.prevent="handleWheel($event, 'row', rIndex, 'c1')"
                        @touchstart="startNumberDrag($event, 'row', rIndex, 'c1')"
                        @touchmove.prevent="onNumberDrag"
                        @touchend="stopNumberDrag"
                        @mousedown="startNumberDrag($event, 'row', rIndex, 'c1')">{{ matrixData.rows[rIndex].c1 }}</span>
                </div>
              </div>
              <div v-for="(cellType, cIndex) in rowType"
                   :key="'cell-'+rIndex+'-'+cIndex"
                   :class="['cell', 'grid-cell', cellType + '-cell']">
                <LockIcon v-if="cellType.startsWith('preset')" class="lock-icon-svg" />
              </div>
            </div>
          </div>

          <div class="pieces-board">
            <div class="pieces-grid">
              <div v-for="(piece, pIndex) in recognizedPieces" :key="'piece-'+pIndex" class="piece-box">
                <div class="piece-render">
                  <div v-for="(rData, rIdx) in piece.grid" :key="'pr-'+rIdx" class="piece-row">
                    <div v-for="(cData, cIdx) in rData" :key="'pc-'+cIdx"
                         :class="['piece-cell', cData ? (detectMode === 'single' ? 'piece-single' : 'piece-' + piece.type) : 'piece-empty']">
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="button-group" style="margin-top: 20px;">
          <button class="btn btn-primary" @click="solvePuzzle">开始求解</button>
          <button class="btn btn-secondary" @click="handleReUpload">重新上传</button>
        </div>
      </div>
    </div>

    <div v-else-if="currentStep === 'solved'" class="preview-wrapper">
      <div class="result-container">
        <div class="tip-sub" style="font-size: 1.2rem; color: #67c23a; margin-bottom: 15px; font-weight: bold;">
          求解成功
        </div>

        <div class="matrix-board">
          <div class="matrix-row header-row">
            <div class="cell corner-cell"></div>
            <div v-for="(colVal, i) in matrixData.cols" :key="'col-sol-'+i" class="cell header-cell" style="display:flex; justify-content:center; align-items:center;">
                <span v-if="detectMode === 'single'" class="adjustable-number"
                      @wheel.prevent="handleWheel($event, 'col', i)"
                      @touchstart="startNumberDrag($event, 'col', i)"
                      @touchmove.prevent="onNumberDrag"
                      @touchend="stopNumberDrag"
                      @mousedown="startNumberDrag($event, 'col', i)">
                  {{ matrixData.cols[i] }}
                </span>
              <div v-else style="display:flex; align-items:center; font-size: 1.1rem; gap: 4px;">
                <span style="color: #67c23a;" class="adjustable-number"
                      @wheel.prevent="handleWheel($event, 'col', i, 'c1')"
                      @touchstart="startNumberDrag($event, 'col', i, 'c1')"
                      @touchmove.prevent="onNumberDrag"
                      @touchend="stopNumberDrag"
                      @mousedown="startNumberDrag($event, 'col', i, 'c1')">{{ matrixData.cols[i].c1 }}</span>
                <span style="color: #ccc; font-size: 0.9rem;">|</span>
                <span style="color: #409eff;" class="adjustable-number"
                      @wheel.prevent="handleWheel($event, 'col', i, 'c2')"
                      @touchstart="startNumberDrag($event, 'col', i, 'c2')"
                      @touchmove.prevent="onNumberDrag"
                      @touchend="stopNumberDrag"
                      @mousedown="startNumberDrag($event, 'col', i, 'c2')">{{ matrixData.cols[i].c2 }}</span>
              </div>
            </div>
          </div>
          <div v-for="(row, rIndex) in matrixData.solution" :key="'sol-row-'+rIndex" class="matrix-row">
            <div class="cell header-cell side-header" style="display:flex; justify-content:center; align-items:center;">
              <span v-if="detectMode === 'single'" class="adjustable-number"
                    @wheel.prevent="handleWheel($event, 'row', rIndex)"
                    @touchstart="startNumberDrag($event, 'row', rIndex)"
                    @touchmove.prevent="onNumberDrag"
                    @touchend="stopNumberDrag"
                    @mousedown="startNumberDrag($event, 'row', rIndex)">
                {{ matrixData.rows[rIndex] }}
              </span>
              <div v-else style="display:flex; flex-direction:column; align-items:center; line-height:1.1; font-size: 1rem;">
                <span style="color: #409eff;" class="adjustable-number"
                      @wheel.prevent="handleWheel($event, 'row', rIndex, 'c2')"
                      @touchstart="startNumberDrag($event, 'row', rIndex, 'c2')"
                      @touchmove.prevent="onNumberDrag"
                      @touchend="stopNumberDrag"
                      @mousedown="startNumberDrag($event, 'row', rIndex, 'c2')">{{ matrixData.rows[rIndex].c2 }}</span>
                <span style="color: #67c23a;" class="adjustable-number"
                      @wheel.prevent="handleWheel($event, 'row', rIndex, 'c1')"
                      @touchstart="startNumberDrag($event, 'row', rIndex, 'c1')"
                      @touchmove.prevent="onNumberDrag"
                      @touchend="stopNumberDrag"
                      @mousedown="startNumberDrag($event, 'row', rIndex, 'c1')">{{ matrixData.rows[rIndex].c1 }}</span>
              </div>
            </div>
            <div v-for="(cellData, cIndex) in row"
                 :key="'sol-cell-'+rIndex+'-'+cIndex"
                 :class="[
                   'cell', 'grid-cell', getSolutionCellClass(cellData),
                   {
                     'conn-t': checkConnection(rIndex, cIndex, 'top'),
                     'conn-b': checkConnection(rIndex, cIndex, 'bottom'),
                     'conn-l': checkConnection(rIndex, cIndex, 'left'),
                     'conn-r': checkConnection(rIndex, cIndex, 'right'),
                     'conn-tr': checkConnection(rIndex, cIndex, 'top-right'),
                     'conn-bl': checkConnection(rIndex, cIndex, 'bottom-left'),
                     'conn-br': checkConnection(rIndex, cIndex, 'bottom-right')
                   }
                 ]">

              <LockIcon v-if="cellData && (cellData.type === 'preset-c1' || cellData.type === 'preset-c2')"
                        class="lock-icon-svg" />

              <div v-if="checkConnection(rIndex, cIndex, 'right') && checkConnection(rIndex, cIndex, 'bottom') && checkConnection(rIndex, cIndex, 'bottom-right')"
                   class="corner-bridge">
              </div>
            </div>
          </div>
        </div>

        <div class="button-group" style="margin-top: 20px;">
          <button class="btn btn-secondary" @click="currentStep = 'result'">查看原选区</button>
          <button class="btn btn-primary" @click="handleReUpload">完成并重置</button>
        </div>
      </div>
    </div>

  </div>
</template>

<style>
@import "Root.css";
</style>